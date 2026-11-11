"""Dataset download, manifest, frozen speaker-disjoint splits, 16 kHz cache and PyTorch datasets.

Folder layout of CSALT/deepfake_detection_dataset_urdu (Hugging Face release):
    Bonafide/Speaker_XX/Part 1/*.wav      genuine speech
    Bonafide/Speaker_XX/Part 2/*.wav      genuine speech, same sentence IDs as the spoofs
    Spoofed_TTS/Speaker_XX/*.wav          VITS clone of Speaker_XX
    Spoofed_Tacotron/Speaker_XX/*.wav     Tacotron clone of Speaker_XX
Every clip of a speaker (genuine and both clones) goes to the same split, so
the test set contains only voices the model never heard.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from .audio_io import load_audio, normalize, repeat_pad, trim_silence, window_starts
from .augment import augment
from .config import Config

LABELS = {0: "bona_fide", 1: "synthetic"}


# ---------------------------------------------------------------- download
def download_dataset(cfg: Config) -> Path:
    """Resumable download. Re-running continues partial downloads and skips finished files."""
    if cfg.local_dataset_dir:
        return Path(cfg.local_dataset_dir)
    from huggingface_hub import snapshot_download

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    marker = cfg.raw_dir / ".download_complete"
    if marker.exists():
        return cfg.raw_dir
    snapshot_download(
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        local_dir=str(cfg.raw_dir),
        allow_patterns=["Bonafide/**", "Spoofed_*/**", "README.md"],
        token=os.environ.get("HF_TOKEN") or None,
        max_workers=8,
    )
    marker.write_text("ok")
    return cfg.raw_dir


# ---------------------------------------------------------------- manifest
def _generator_from_path(parts: list[str]) -> str | None:
    top = parts[0].lower()
    if top.startswith("bonafide") or top.startswith("bona"):
        return "bonafide"
    if "tacotron" in top:
        return "tacotron"
    if "tts" in top or "vits" in top:
        return "vits"
    return None


def build_manifest(cfg: Config) -> pd.DataFrame:
    root = cfg.raw_dir
    rows = []
    for p in sorted(root.rglob("*.wav")):
        rel = p.relative_to(root)
        parts = list(rel.parts)
        gen = _generator_from_path(parts)
        spk = next((m.group(1) for s in parts if (m := re.match(r"(?i)speaker[_ ]?(\d+)", s))), None)
        if gen is None or spk is None:
            continue
        part_m = next((re.match(r"(?i)part\s*(\d+)", s) for s in parts if re.match(r"(?i)part\s*\d+", s)), None)
        part = int(part_m.group(1)) if part_m else 0
        utt = p.stem
        group = f"S{spk}_p1_{utt}" if (gen == "bonafide" and part == 1) else f"S{spk}_{utt}"
        try:
            info = sf.info(str(p))
            file_sr, file_dur = int(info.samplerate), float(info.duration)
        except Exception:
            file_sr, file_dur = -1, -1.0
        rows.append(dict(path=rel.as_posix(), speaker=f"S{int(spk):02d}", generator=gen,
                         label=0 if gen == "bonafide" else 1, part=part, utt_id=utt,
                         group=group, file_sr=file_sr, file_dur=file_dur))
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No wav files found under {root}")
    return df


def make_splits(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Frozen speaker-disjoint split. Written once to data/splits.json and reused forever."""
    split_file = cfg.root / "data" / "splits.json"
    speakers = sorted(df.speaker.unique())
    if split_file.exists():
        sp = json.loads(split_file.read_text())
    else:
        rng = np.random.default_rng(cfg.split_seed)
        perm = list(rng.permutation(speakers))
        test = sorted(perm[: cfg.n_test_speakers])
        dev = sorted(perm[cfg.n_test_speakers: cfg.n_test_speakers + cfg.n_dev_speakers])
        train = sorted(s for s in speakers if s not in test and s not in dev)
        sp = {"seed": cfg.split_seed, "train": train, "dev": dev, "test": test}
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(sp, indent=2))
    lookup = {s: k for k in ("train", "dev", "test") for s in sp[k]}
    df = df.copy()
    df["split"] = df.speaker.map(lookup).fillna("train")
    return df


# ---------------------------------------------------------------- 16 kHz cache
def cache_path(cfg: Config, rel: str) -> Path:
    return cfg.cache_dir / (rel.rsplit(".", 1)[0] + ".npy")


def _process_one(src: str, dst: str, sr: int, top_db: float) -> dict:
    dst_p = Path(dst)
    if dst_p.exists():
        y = np.load(dst_p, mmap_mode="r")
        meta_p = dst_p.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        meta["n"] = int(y.shape[0])
        return meta
    y, sr_in = load_audio(src, sr)
    raw_n = y.size
    yt, s, e = trim_silence(y, top_db)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(yt).max()) if yt.size else 0.0
    rms = float(np.sqrt(np.mean(yt ** 2))) if yt.size else 0.0
    meta = {"orig_sr": int(sr_in), "raw_sec": raw_n / sr, "lead_sil_sec": s / sr,
            "trail_sil_sec": (raw_n - e) / sr if raw_n else 0.0,
            "rms_dbfs": float(20 * np.log10(rms + 1e-10)), "peak": peak, "n": int(yt.size)}
    tmp = dst_p.with_name(dst_p.stem + ".tmp.npy")
    np.save(tmp, (np.clip(yt, -1, 1) * 32767).astype(np.int16))
    dst_p.with_suffix(".json").write_text(json.dumps(meta))   # metadata first, then the atomic rename
    os.replace(tmp, dst_p)
    return meta


def preprocess_all(df: pd.DataFrame, cfg: Config, n_jobs: int = -1) -> pd.DataFrame:
    """Decode, resample to 16 kHz, trim, cache as int16 .npy. Resumable: finished files are skipped."""
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    jobs = [(str(cfg.raw_dir / r), str(cache_path(cfg, r))) for r in df.path]
    metas = Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(_process_one)(s, d, cfg.sample_rate, cfg.trim_top_db) for s, d in tqdm(jobs, desc="cache 16k"))
    meta = pd.DataFrame(metas)
    out = pd.concat([df.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    out["sec"] = out["n"] / cfg.sample_rate
    out["cache"] = [str(cache_path(cfg, r)) for r in out.path]
    out["keep"] = out["sec"] >= cfg.min_clip_sec
    out.to_csv(cfg.root / "data" / "manifest.csv", index=False)
    return out


def manifest_hash(df: pd.DataFrame) -> str:
    key = "\n".join(sorted(df.path + ":" + df.split))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def load_clip(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32) / 32767.0


# ---------------------------------------------------------------- datasets
class ResumableBatchSampler(Sampler):
    """Deterministic batches per epoch, so training can resume at the exact batch after a power cut.

    Yields lists of (index, epoch, crop_samples). One crop length per batch.
    """

    def __init__(self, n: int, batch_size: int, grad_accum: int, seed: int, crop_samples: list[int]):
        self.n, self.bs, self.accum, self.seed, self.crops = n, batch_size, grad_accum, seed, crop_samples
        self.epoch, self.start_batch = 0, 0

    def batches_per_epoch(self) -> int:
        nb = self.n // self.bs
        return nb - nb % self.accum

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        perm = rng.permutation(self.n)
        crops = rng.choice(self.crops, size=self.batches_per_epoch())
        for b in range(self.start_batch, self.batches_per_epoch()):
            idx = perm[b * self.bs: (b + 1) * self.bs]
            yield [(int(i), self.epoch, int(crops[b])) for i in idx]

    def __len__(self) -> int:
        return self.batches_per_epoch() - self.start_batch


class TrainDataset(Dataset):
    def __init__(self, cache_paths: list[str], labels: list[int], cfg: Config, train: bool = True):
        self.paths, self.labels, self.cfg, self.train = list(cache_paths), list(labels), cfg, train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, key):
        idx, epoch, crop_n = key
        rng = np.random.default_rng([self.cfg.seed, epoch, idx])
        y = load_clip(self.paths[idx])
        if y.size > crop_n:
            s = int(rng.integers(0, y.size - crop_n + 1))
            y = y[s: s + crop_n]
        else:
            y = repeat_pad(y, crop_n)
        if self.train and rng.random() < self.cfg.aug_prob:
            y = augment(y, self.cfg.sample_rate, rng)
        return torch.from_numpy(normalize(y)), torch.tensor(self.labels[idx], dtype=torch.long)


class WindowDataset(Dataset):
    """Fixed-length windows over whole clips, the exact procedure the product uses at inference."""

    def __init__(self, clips: list, win: int, hop: int, transform=None):
        self.clips, self.win, self.transform = clips, win, transform
        self.items = []
        self._cache = {}
        for ci, c in enumerate(clips):
            n = c.size if isinstance(c, np.ndarray) else int(np.load(c, mmap_mode="r").shape[0])
            self.items += [(ci, s) for s in window_starts(n, win, hop)]

    def _get(self, ci: int) -> np.ndarray:
        c = self.clips[ci]
        if isinstance(c, np.ndarray):
            return c
        if ci not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            y = load_clip(c)
            self._cache[ci] = self.transform(y, ci) if self.transform else y
        return self._cache[ci]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        ci, s = self.items[i]
        y = self._get(ci)
        w = repeat_pad(y[s: s + self.win], self.win)
        return torch.from_numpy(normalize(w)), ci, s


# ---------------------------------------------------------------- shortcut audit
def audit(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-class recording properties, plus a leakage check: how well do metadata alone predict the label?

    A high metadata-only AUC means a model could cheat on duration, silence or loudness. Trimming, random
    crops and per-window normalisation are there to remove those cues from what the detector sees.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    df = manifest[manifest.keep]
    tbl = df.groupby("generator").agg(
        clips=("path", "size"), speakers=("speaker", "nunique"),
        source_sr=("orig_sr", lambda s: ",".join(str(int(v)) for v in sorted(s.unique()))),
        raw_sec_median=("raw_sec", "median"), trimmed_sec_median=("sec", "median"),
        lead_silence_median=("lead_sil_sec", "median"), trail_silence_median=("trail_sil_sec", "median"),
        rms_dbfs_median=("rms_dbfs", "median")).reset_index()
    feats = ["raw_sec", "lead_sil_sec", "trail_sil_sec", "rms_dbfs"]
    tr, dv = df[df.split == "train"], df[df.split == "dev"]
    out = {}
    for name, cols in (("raw_metadata", feats), ("after_trim", ["sec"])):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[cols], tr.label)
        out[f"metadata_auc_{name}"] = float(roc_auc_score(dv.label, clf.predict_proba(dv[cols])[:, 1]))
    out["excluded_short_clips"] = int((~manifest.keep).sum())
    return tbl, out
