"""Product inference: load once, analyse any audio file, return a verdict with its measured basis."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import metrics as M
from .audio_io import load_audio, normalize, probe, quality_report, repeat_pad, trim_silence, window_starts
from .config import Config
from .evaluate import mahalanobis_min
from .evidence import acoustic_descriptors, compare
from .model import SachaiDetector
from .train import amp_settings, autocast_ctx, pick_device

LABELS_EN = {"synthetic": "Likely synthetic", "bona_fide": "Likely genuine", "inconclusive": "Inconclusive"}
LABELS_UR = {"synthetic": "غالباً مصنوعی آواز", "bona_fide": "غالباً اصلی آواز", "inconclusive": "نتیجہ غیر واضح"}


def resolve_model_dir(model: str | Path) -> Path:
    """Local export folder, or a Hugging Face repo id (downloaded once, then served from the local cache)."""
    p = Path(model)
    if (p / "policy.json").exists():
        return p
    from huggingface_hub import snapshot_download
    import os
    return Path(snapshot_download(repo_id=str(model), token=os.environ.get("HF_TOKEN") or None))


class Detector:
    def __init__(self, model: str | Path, device: torch.device | str | None = None):
        self.dir = resolve_model_dir(model)
        self.cfg = Config.load(self.dir / "sachai_config.json")
        self.policy = json.loads((self.dir / "policy.json").read_text())
        self.device = torch.device(device) if device else pick_device()
        self.model = SachaiDetector.from_pretrained(self.dir).to(self.device).eval()
        self.amp, _ = amp_settings(self.device)
        ood = np.load(self.dir / "ood.npz")
        self.ood = {k: ood[k] for k in ood.files if k != "model_id"}
        ref_p = self.dir / "reference_stats.json"
        self.ref = json.loads(ref_p.read_text(encoding="utf-8")) if ref_p.exists() else None
        self.version = {"model_dir": str(self.dir), "preprocess": self.policy.get("preprocess_version"),
                        "policy": self.policy.get("policy_version"), "backbone": self.cfg.backbone,
                        "run": self.cfg.run_name}

    @torch.no_grad()
    def _windows(self, y: np.ndarray, win_sec: float, hop_sec: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
        sr = self.cfg.sample_rate
        win, hop = int(win_sec * sr), int(hop_sec * sr)
        starts = window_starts(y.size, win, hop)
        zs, embs = [], []
        for i in range(0, len(starts), self.cfg.infer_batch):
            batch = np.stack([normalize(repeat_pad(y[s: s + win], win)) for s in starts[i: i + self.cfg.infer_batch]])
            with autocast_ctx(self.device, self.amp):
                out = self.model(torch.from_numpy(batch).to(self.device))
            zs.append(SachaiDetector.spoof_logit(out["logits"]).float().cpu().numpy())
            embs.append(out["embedding"].float().cpu().numpy())
        return np.concatenate(zs), np.concatenate(embs), starts

    def score_array(self, y: np.ndarray) -> dict:
        """y: trimmed mono float32 at the model sample rate."""
        sr, T = self.cfg.sample_rate, self.policy["temperature"]
        z, emb, _ = self._windows(y, self.cfg.verdict_win_sec, self.cfg.verdict_hop_sec)
        clip_z = float(z.mean())
        p = float(M.sigmoid(clip_z / T))
        zs, _, starts = self._windows(y, self.cfg.segment_win_sec, self.cfg.segment_hop_sec)
        seg_len = self.cfg.segment_win_sec
        segments = [{"start": round(s / sr, 2), "end": round(min(s / sr + seg_len, y.size / sr), 2),
                     "score": round(float(M.sigmoid(v / T)), 4),
                     "flagged": bool(M.sigmoid(v / T) >= self.policy["high"])} for s, v in zip(starts, zs)]
        d = float(mahalanobis_min(emb.mean(0, keepdims=True), self.ood["means"], self.ood["precision"])[0])
        thr = float(self.ood["threshold"])
        return {"spoof_logit": clip_z, "spoof_score": round(p, 4), "verdict": M.decide(p, self.policy),
                "segments": segments,
                "ood": {"flag": bool(d > thr), "distance": round(d, 3), "threshold": round(thr, 3),
                        "ratio": round(d / max(thr, 1e-9), 3)}}

    def decode_file(self, path: str | Path) -> tuple[np.ndarray, dict]:
        """Decode any format ffmpeg reads (wav, mp3, m4a, ogg/opus, webm) to 16 kHz mono."""
        meta = probe(path)
        y, sr_in = load_audio(path, self.cfg.sample_rate, prefer_ffmpeg=True)
        meta["source_sample_rate"] = meta.get("sample_rate") or sr_in
        meta["decoded_sec"] = round(y.size / self.cfg.sample_rate, 2)
        return y, meta

    def analyze_file(self, path: str | Path, with_descriptors: bool = True) -> dict:
        t0 = time.time()
        y, meta = self.decode_file(path)
        res = self.analyze_array(y, meta, with_descriptors)
        res["timing_ms"]["decode"] = int((time.time() - t0) * 1000) - sum(res["timing_ms"].values())
        return res

    def analyze_array(self, y: np.ndarray, meta: dict | None = None, with_descriptors: bool = True) -> dict:
        meta = meta or {}
        sr = self.cfg.sample_rate
        t0 = time.time()
        yt, s, e = trim_silence(y, self.cfg.trim_top_db)
        q = quality_report(yt, sr)
        q.update(decoded_sec=round(y.size / sr, 2), trimmed_sec=round(yt.size / sr, 2), codec=meta.get("codec"),
                 source_sample_rate=meta.get("source_sample_rate", sr))
        flags = []
        if q["clipping_pct"] > 1.0:
            flags.append("clipping")
        if q["snr_db"] is not None and q["snr_db"] < 12:
            flags.append("noisy")
        if q["trimmed_sec"] < 1.5:
            flags.append("too_little_speech")
        q["flags"] = flags
        if yt.size < int(0.5 * sr):
            res = {"verdict": "inconclusive", "spoof_score": None, "segments": [], "ood": {"flag": False},
                   "reason": "not_enough_speech"}
        else:
            res = self.score_array(yt)
            if "too_little_speech" in flags or (q["snr_db"] is not None and q["snr_db"] < 6):
                res["verdict"], res["reason"] = "inconclusive", "quality_gate"
        t_score = time.time()
        res["quality"] = q
        res["trim_offset_sec"] = round(s / sr, 3)
        res["descriptors"] = compare(acoustic_descriptors(yt, sr), self.ref) if (with_descriptors and self.ref and yt.size) else []
        res["label_en"], res["label_ur"] = LABELS_EN[res["verdict"]], LABELS_UR[res["verdict"]]
        res["policy"] = {k: self.policy[k] for k in ("low", "high", "eer_threshold", "temperature")}
        res["versions"] = self.version
        res["timing_ms"] = {"score": int((t_score - t0) * 1000), "descriptors": int((time.time() - t_score) * 1000)}
        return res
