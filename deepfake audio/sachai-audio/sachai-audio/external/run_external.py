"""Step 1: zero-shot test of the frozen Sachai detector on new Urdu data. Nothing is retrained or refitted.

Real: Kathbath Urdu (AI4Bharat), clean valid / test-known / test-unknown and noisy test-known / test-unknown,
      from the public IndicSUPERB archives (no token needed).
Fake: IndicSynth Urdu (ACL 2025): XTTS-v2 voice cloning and FreeVC24 voice conversion, stratified per target speaker.

The existing pipeline is only read, never written: data/splits.json, data/manifest.csv, runs/<run>/export and
runs/<run>/report stay as they are. New files go to data/external/, runs/<run>/external/ and external/frozen/.

    .venv/Scripts/python external/run_external.py download   # ~14 GB of tar streamed (only Urdu kept) + ~7.5 GB fakes
    .venv/Scripts/python external/run_external.py prepare    # codec-matched fakes, 16 kHz cache, parity + leakage checks
    .venv/Scripts/python external/run_external.py freeze     # speaker split + step-2 rule, written once, before scoring
    .venv/Scripts/python external/run_external.py score      # frozen v1 model; checks nothing frozen changed
    .venv/Scripts/python external/run_external.py report     # tables with the v1 policy, unchanged
    .venv/Scripts/python external/run_external.py all

Every stage is safe to re-run: finished downloads, files and scores are skipped.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

EXT = ROOT / "data" / "external"
RAW, CACHE, MANIFEST = EXT / "raw", EXT / "cache_16k", EXT / "manifest.csv"
FROZEN = HERE / "frozen"
SPLIT_FILE, RULE_FILE = FROZEN / "speaker_split.json", FROZEN / "step2_rule.json"
INDEX = HERE / "indicsynth_urdu_index.csv.gz"

KB_URL = "https://objectstore.e2enetworks.net/indic-superb/kathbath/"
KB_TARS = {"clean_valid": "clean/valid_audio.tar", "clean_testkn": "clean/testkn_audio.tar",
           "clean_testunk": "clean/testunk_audio.tar", "noisy_testkn": "noisy/testkn_audio.tar",
           "noisy_testunk": "noisy/testunk_audio.tar"}
IS_REPO, IS_REV = "vdivyasharma/IndicSynth", "c0a10386b723717aff682f757bd67f72983f269f"
GENERATORS = ("xtts_v2", "freevc24")
N_FAKES = {"xtts_v2": 3500, "freevc24": 800}   # FreeVC24 is smaller because its clips repeat source recordings
SEED = 20261116
AAC_SR, AAC_KBPS = 44100, 128     # measured on Kathbath m4a: AAC-LC, 44.1 kHz mono, about 128 kb/s (prepare re-checks)
NAME_RE = re.compile(r"(\d+)-(\d+)-([mf])\.m4a$")
VIEWS = ("aac_matched", "as_distributed")   # headline first


# ---------------------------------------------------------------- download
def fetch(url: str, dst: Path) -> None:
    """Resumable HTTP download: after a power cut or network drop, re-running continues the partial file."""
    from tqdm.auto import tqdm
    if dst.exists():
        return
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        total = int(r.headers["Content-Length"])
    part = dst.with_name(dst.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    done = part.stat().st_size if part.exists() else 0
    if done < total:
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f, \
                tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=dst.name) as bar:
            if done and r.status != 206:
                raise RuntimeError(f"Server ignored the resume request. Delete {part} and re-run.")
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                bar.update(len(chunk))
    if part.stat().st_size != total:
        raise RuntimeError(f"Incomplete download {part}. Re-run to continue.")
    os.replace(part, dst)


def _write_atomic(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


def extract_urdu(tar_path: Path, out_dir: Path) -> int:
    """Copies only Urdu clips out of an all-language archive. Uses the file name only, never the archive's own
    paths, so a downloaded archive cannot write outside out_dir."""
    n = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not (m.isfile() and "/urdu/" in m.name and NAME_RE.search(m.name)):
                continue
            dst = out_dir / Path(m.name).name
            if not dst.exists():
                _write_atomic(dst, tf.extractfile(m).read())
            n += 1
    return n


def download_kathbath(keep_tars: bool = False) -> None:
    for name, rel in KB_TARS.items():
        out = RAW / "kathbath" / name
        if (out / ".done").exists():
            print(f"kathbath {name}: done ({(out / '.done').read_text()} Urdu clips)")
            continue
        tar = EXT / "downloads" / f"{name}.tar"
        fetch(KB_URL + rel, tar)
        out.mkdir(parents=True, exist_ok=True)
        n = extract_urdu(tar, out)
        (out / ".done").write_text(str(n))
        if not keep_tars:
            tar.unlink()
        print(f"kathbath {name}: {n} Urdu clips")


def select_fakes(idx: pd.DataFrame, n_fakes: dict, seed: int = SEED, max_per_source: int = 2) -> pd.DataFrame:
    """Per generator, an equal number of clips per target speaker. Row groups (100 clips, one ~40 MB read each) are
    visited in random order and every useful row of a visited group is used. FreeVC24 converts each source
    recording 20 to 30 times, so at most `max_per_source` clips per (target, source recording) are kept."""
    rng = np.random.default_rng(seed)
    picks = []
    for g in GENERATORS:
        d = idx[idx.generator == g]
        need = {t: math.ceil(n_fakes[g] / d.target.nunique()) for t in d.target.unique()}
        used = {}
        groups = [x for _, x in d.groupby(["file", "rg"])]
        for i in rng.permutation(len(groups)):
            take = []
            for r in groups[i].itertuples():
                key = (r.target, r.source_clip)
                if need[r.target] > 0 and (r.source_clip != r.source_clip or used.get(key, 0) < max_per_source):
                    take.append(r.Index)
                    need[r.target] -= 1
                    used[key] = used.get(key, 0) + 1
            if take:
                picks.append(d.loc[take])
            if not any(v > 0 for v in need.values()):
                break
    return pd.concat(picks, ignore_index=True)


def fake_path(r) -> str:
    return f"indicsynth/{r.generator}/{Path(r.file).stem}_{r.rg}_{r.row}.wav"


def download_indicsynth(n_fakes: dict = N_FAKES) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from tqdm.auto import tqdm
    pick = select_fakes(pd.read_csv(INDEX), n_fakes)
    pick["path"] = [fake_path(r) for r in pick.itertuples()]
    todo = pick[[not (RAW / p).exists() for p in pick.path]]
    fs = HfFileSystem()
    for f, df in tqdm(list(todo.groupby("file")), desc="indicsynth shards"):
        with fs.open(f"datasets/{IS_REPO}@{IS_REV}/Urdu/{f}", "rb", block_size=1 << 20) as h:
            pf = pq.ParquetFile(h)
            for rg, d in df.groupby("rg"):
                col = pf.read_row_group(int(rg), columns=["audio"]).column(0)
                for r, p in zip(d.row, d.path):
                    _write_atomic(RAW / p, col[int(r)].as_py()["bytes"])
    pick.to_csv(EXT / "indicsynth_pick.csv", index=False)
    print(pick.groupby("generator").agg(clips=("path", "size"), targets=("target", "nunique"),
                                        source_recordings=("source_clip", "nunique")))


def build_index() -> None:
    """How external/indicsynth_urdu_index.csv.gz was made (metadata columns only, no audio). Takes ~1 hour."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    cols = ["Generative Model", "Target Speaker ID", "Source Speaker_ID", "Gender", "Source Reference Audio"]
    rows = []
    for f in sorted(fs.glob(f"datasets/{IS_REPO}@{IS_REV}/Urdu/*.parquet")):
        with fs.open(f, "rb", block_size=1 << 18) as h:
            pf = pq.ParquetFile(h)
            for rg in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(rg, columns=cols)
                for row, v in enumerate(zip(*(t.column(k).to_pylist() for k in range(len(cols))))):
                    rows.append((f.rsplit("/", 1)[1], rg, row, *v))
        print(f, len(rows), flush=True)
    pd.DataFrame(rows, columns=["file", "rg", "row", "generator", "target", "source", "gender", "source_clip"]).to_csv(
        INDEX, index=False)


# ---------------------------------------------------------------- prepare
def kathbath_format(n: int = 20) -> pd.DataFrame:
    """Codec, sample rate and bitrate of a random sample of the real clips."""
    from sachai_audio.audio_io import ffmpeg_exe, probe
    files = sorted((RAW / "kathbath").rglob("*.m4a"))
    rows = []
    for p in np.random.default_rng(SEED).choice(files, min(n, len(files)), replace=False):
        err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(p)], capture_output=True, text=True).stderr
        m = re.search(r"bitrate:\s*(\d+)\s*kb/s", err)
        rows.append({**probe(p), "kbps": int(m.group(1)) if m else None})
    return pd.DataFrame(rows)


def aac_match(src: str, dst: str) -> None:
    """Fake wav -> the same codec settings as the real Kathbath clips."""
    from sachai_audio.audio_io import ffmpeg_exe
    d = Path(dst)
    if d.exists():
        return
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.with_name(d.stem + ".part.m4a")
    subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", str(AAC_SR),
                    "-c:a", "aac", "-b:a", f"{AAC_KBPS}k", str(tmp)], check=True, capture_output=True, timeout=120)
    os.replace(tmp, d)


def build_manifest() -> pd.DataFrame:
    rows = []
    for name in KB_TARS:
        for p in sorted((RAW / "kathbath" / name).glob("*.m4a")):
            _, spk, g = NAME_RE.search(p.name).groups()
            rows.append(dict(path=p.relative_to(RAW).as_posix(), source=f"kathbath_{name.split('_')[0]}", kb_split=name,
                             label=0, speaker=f"K{int(spk)}", source_speaker="", gender=g, variant="real"))
    for r in pd.read_csv(EXT / "indicsynth_pick.csv").itertuples():
        base = dict(source=r.generator, kb_split="", label=1, speaker=f"K{int(r.target)}",
                    source_speaker=f"K{int(r.source)}" if r.source == r.source else "", gender=str(r.gender)[0].lower())
        rows.append({**base, "path": r.path, "variant": "as_distributed"})
        rows.append({**base, "path": r.path.replace("indicsynth/", "indicsynth_aac/").replace(".wav", ".m4a"),
                     "variant": "aac_matched"})
    df = pd.DataFrame(rows)
    df["generator"], df["group"] = df.source, df.speaker      # the column names sachai_audio's evaluation code expects
    return df


def cache_of(rel: str) -> str:
    return str(CACHE / (rel.rsplit(".", 1)[0] + ".npy"))


def parity_check(cfg, n: int = 5) -> None:
    """Re-caches a few CSALT test clips through this script's path; they must match the existing cache exactly."""
    import tempfile
    from sachai_audio.data import _process_one, cache_path
    m = pd.read_csv(cfg.root / "data" / "manifest.csv")
    te = m[(m.split == "test") & m.keep].sample(n, random_state=0)
    with tempfile.TemporaryDirectory() as td:
        for i, r in enumerate(te.itertuples()):
            dst = Path(td) / f"{i}.npy"
            _process_one(str(cfg.raw_dir / r.path), str(dst), cfg.sample_rate, cfg.trim_top_db)
            if not np.array_equal(np.load(dst), np.load(cache_path(cfg, r.path))):
                sys.exit(f"Parity check failed on {r.path}: preprocessing differs from the training cache.")
    print(f"parity check: {n} CSALT test clips re-cached identically")


def split_halves(speakers: list[str], seed: int = SEED) -> dict[str, str]:
    perm = np.random.default_rng(seed).permutation(sorted(speakers))
    return {s: ("a" if i % 2 == 0 else "b") for i, s in enumerate(perm)}


def prepare(cfg) -> pd.DataFrame:
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from sachai_audio.data import _process_one, audit
    fmt = kathbath_format()
    print("Kathbath sample format:\n", fmt.groupby(["codec", "sample_rate", "channels"]).kbps.describe()[["count", "50%"]])
    if set(fmt.codec) != {"aac"} or set(fmt.sample_rate) != {AAC_SR} or abs(fmt.kbps.median() - AAC_KBPS) > 16:
        sys.exit(f"Kathbath format differs from AAC {AAC_SR} Hz {AAC_KBPS} kb/s. Update AAC_SR / AAC_KBPS first.")
    man = build_manifest()
    aac = man[man.variant == "aac_matched"]
    Parallel(n_jobs=-1, batch_size=16)(delayed(aac_match)(str(RAW / s.replace("indicsynth_aac/", "indicsynth/").replace(".m4a", ".wav")),
                                                          str(RAW / s)) for s in tqdm(aac.path, desc="aac match"))
    metas = Parallel(n_jobs=-1, batch_size=16)(delayed(_process_one)(str(RAW / p), cache_of(p), cfg.sample_rate, cfg.trim_top_db)
                                               for p in tqdm(man.path, desc="cache 16k"))
    man = pd.concat([man.reset_index(drop=True), pd.DataFrame(metas)], axis=1)
    man["sec"] = man.n / cfg.sample_rate
    man["keep"] = man.sec >= cfg.min_clip_sec
    man["cache"] = [cache_of(p) for p in man.path]
    man.to_csv(MANIFEST, index=False)
    print(man[man.keep].groupby(["source", "variant"]).agg(clips=("path", "size"), speakers=("speaker", "nunique"),
                                                           hours=("sec", lambda s: round(s.sum() / 3600, 1))))
    parity_check(cfg)
    out = cfg.run_dir / "external"
    out.mkdir(parents=True, exist_ok=True)
    half = split_halves(man.speaker.unique().tolist())
    for view in VIEWS:       # metadata-only leakage check: fit on half the speakers, score the other half
        d = man[man.variant.isin(["real", view])].copy()
        d["split"] = d.speaker.map(lambda s: "train" if half[s] == "a" else "dev")
        tbl, aucs = audit(d)
        tbl.to_csv(out / f"audit_{view}.csv", index=False)
        (out / f"audit_{view}.json").write_text(json.dumps(aucs, indent=2))
        flag = "  <-- above 0.75, report it" if max(aucs["metadata_auc_raw_metadata"], aucs["metadata_auc_after_trim"]) > 0.75 else ""
        print(f"leakage audit ({view}): {aucs}{flag}")
    return man


# ---------------------------------------------------------------- freeze (before any scoring)
def split_speakers(man: pd.DataFrame, seed: int = SEED) -> dict[str, list[str]]:
    """Calibration vs evaluation speakers, balanced within each kind of speaker (real only / fake only / both)."""
    has_real = set(man.speaker[man.label == 0])
    has_fake = set(man.speaker[man.label == 1])
    out = {"calibration": [], "evaluation": []}
    for kind in (has_real - has_fake, has_fake - has_real, has_real & has_fake):
        for s, h in split_halves(sorted(kind), seed).items():
            out["calibration" if h == "a" else "evaluation"].append(s)
    return {k: sorted(v) for k, v in out.items()}


STEP2_RULE = {
    "decided": "2026-11-16, by the team, before any external clip was scored",
    "applies_to": "product policy only; the paper keeps the v1 result on the original CSALT test speakers",
    "keep": "v1 temperature and v1 genuine line (low)",
    "synthetic_line": "smallest value such that at most 1% of calibration-speaker Kathbath real clips (clean and noisy, "
                      "as distributed) have a calibrated spoof score at or above it",
    "max_real_above_line": 0.01,
    "evaluation": "evaluation speakers are scored once with the resulting policy, codec-matched view as headline",
    "freevc24": "a clip belongs to its target speaker; in evaluation, clips whose source speaker is a calibration "
                "speaker are dropped",
    "stop_condition": "if the new synthetic line is at or below the genuine line, stop; the team decides before "
                      "anything else is run",
}


def freeze(cfg, man: pd.DataFrame) -> None:
    if SPLIT_FILE.exists() and RULE_FILE.exists():
        print(f"Frozen files already exist, not rewritten: {SPLIT_FILE.name}, {RULE_FILE.name}")
        return
    if any((cfg.run_dir / "external").glob("scores_*.csv")):
        sys.exit("External scores already exist, so a split frozen now would not be blind. Ask the team.")
    sp = split_speakers(man)
    meta = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "seed": SEED,
            "indicsynth_revision": IS_REV, "kathbath_archives": [KB_URL + r for r in KB_TARS.values()],
            "n_speakers": {k: len(v) for k, v in sp.items()}}
    FROZEN.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps({**meta, **sp}, indent=2))
    RULE_FILE.write_text(json.dumps(STEP2_RULE, indent=2))
    print(f"Frozen: {meta['n_speakers']}. Commit external/frozen/ now, so the date is on record.")


# ---------------------------------------------------------------- score
def _sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest() if p.exists() else "missing"


def frozen_state(cfg) -> dict:
    from sachai_audio.evaluate import model_fingerprint
    files = [cfg.root / "data" / "splits.json", cfg.root / "data" / "manifest.csv", cfg.export_dir / "policy.json",
             cfg.report_dir / "test_scores.csv"]
    return {**{str(f.relative_to(cfg.root)): _sha1(f) for f in files}, "model": model_fingerprint(cfg.export_dir)}


def score(cfg) -> None:
    from sachai_audio.config import PREPROCESS_VERSION
    from sachai_audio.data import cache_path
    from sachai_audio.evaluate import _cached_scores, load_policy, model_fingerprint
    from sachai_audio.model import SachaiDetector
    from sachai_audio.train import pick_device, score_clips
    if not (SPLIT_FILE.exists() and RULE_FILE.exists()):
        sys.exit("Run the freeze stage first (speaker split and step-2 rule must exist before scoring).")
    pol = load_policy(cfg.export_dir)
    if pol.get("preprocess_version") != PREPROCESS_VERSION or "temperature" not in pol:
        sys.exit(f"Export does not match this code ({pol.get('preprocess_version')} vs {PREPROCESS_VERSION}).")
    before = frozen_state(cfg)
    device = pick_device()
    model = SachaiDetector.from_pretrained(cfg.export_dir).to(device).eval()
    mid = model_fingerprint(cfg.export_dir)
    ts = pd.read_csv(cfg.report_dir / "test_scores.csv")
    if not (ts.model_id.astype(str) == mid).all():
        sys.exit("report/test_scores.csv was made by different weights; the sanity check cannot run.")
    ts = ts.sample(8, random_state=0)
    z = score_clips(model, [str(cache_path(cfg, p)) for p in ts.path], cfg, device)["z"]
    diff = float(np.abs(z - ts.z.values).max())
    print(f"sanity: 8 CSALT test clips rescored, max |z difference| {diff:.4f}")
    if diff > 0.05:
        sys.exit("Rescored CSALT clips do not match test_scores.csv: model or preprocessing mismatch.")
    out = cfg.run_dir / "external"
    man = pd.read_csv(MANIFEST)
    man = man[man.keep]
    for name, sel in {"real": man.variant == "real", "fake_aac_matched": man.variant == "aac_matched",
                      "fake_as_distributed": man.variant == "as_distributed"}.items():
        df = man[sel].reset_index(drop=True)
        print(f"scoring {name}: {len(df)} clips")
        _cached_scores(out / f"scores_{name}.csv", df,
                       lambda: score_clips(model, df.cache.tolist(), cfg, device, progress=True), mid)
    after = frozen_state(cfg)
    if before != after:
        sys.exit(f"A frozen file changed during scoring: {before} -> {after}")
    print("frozen files unchanged:", after)


# ---------------------------------------------------------------- report
def report(cfg) -> None:
    from sachai_audio import metrics as M
    from sachai_audio.evaluate import latex_table, load_policy
    pol = load_policy(cfg.export_dir)
    out = cfg.run_dir / "external"
    man = pd.read_csv(MANIFEST)[["path", "source", "kb_split", "gender", "source_speaker"]]
    real = pd.read_csv(out / "scores_real.csv").merge(man, on="path")
    csalt = pd.read_csv(cfg.report_dir / "test_scores.csv")
    csalt_real_p = M.sigmoid(csalt.z[csalt.label == 0] / pol["temperature"])
    print(f"policy v1: T {pol['temperature']:.3f} | genuine <= {pol['low']:.3f} | synthetic >= {pol['high']:.3f} "
          f"| forced cut {pol['eer_threshold']:.3f}")
    for view in VIEWS:
        fake = pd.read_csv(out / f"scores_fake_{view}.csv").merge(man, on="path")
        df = pd.concat([real, fake], ignore_index=True)
        df["p"] = M.sigmoid(df.z / pol["temperature"])
        df["decision"] = [M.decide(v, pol) for v in df.p]
        df.to_csv(out / f"scored_{view}.csv", index=False)

        det = []
        for real_set in (["kathbath_clean"], ["kathbath_clean", "kathbath_noisy"]):
            for atk in (["xtts_v2"], ["freevc24"], list(GENERATORS)):
                d = df[((df.label == 0) & df.source.isin(real_set)) | ((df.label == 1) & df.source.isin(atk))]
                r = M.report(d.label.values, d.p.values, pol)
                ci = M.grouped_bootstrap(d.label.values, d.p.values, d.speaker.values, pol)    # resamples speakers
                det.append({"real": "+".join(s.split("_")[1] for s in real_set), "attack": "+".join(atk),
                            "n_real": r["n_bona_fide"], "n_fake": r["n_synthetic"], "eer": r["eer"],
                            "eer_lo": ci["eer"][0], "eer_hi": ci["eer"][1], "auc": r["auc"],
                            "auc_lo": ci["auc"][0], "auc_hi": ci["auc"][1]})
        det = pd.DataFrame(det)

        lab = []
        groups = [(s, "all", d) for s, d in df.groupby("source")] + \
                 [(s, g, d) for (s, g), d in df.groupby(["source", "gender"])]
        for src, g, d in groups:
            truth = int(d.label.iloc[0])
            lab.append({"population": src, "gender": g, "n": len(d), "speakers": d.speaker.nunique(),
                        "forced_correct": float(((d.p >= pol["eer_threshold"]).astype(int) == truth).mean()),
                        "said_genuine": float((d.decision == "bona_fide").mean()),
                        "inconclusive": float((d.decision == "inconclusive").mean()),
                        "said_synthetic": float((d.decision == "synthetic").mean()), "median_p": float(d.p.median())})
        lab = pd.DataFrame(lab).sort_values(["population", "gender"])

        for name, t, cap in (("detection", det, "EER and AUC (95% CI, speakers resampled)"),
                             ("per_label", lab, "Outcomes per true label with the frozen v1 policy")):
            t.to_csv(out / f"table_{name}_{view}.csv", index=False)
            (out / f"table_{name}_{view}.tex").write_text(latex_table(
                t, f"Zero-shot Kathbath + IndicSynth Urdu, {view.replace('_', ' ')}: {cap}.", f"tab:ext-{name}-{view}"))
        print(f"\n===== {view}{' (headline)' if view == VIEWS[0] else ''} =====")
        print(det.round(4).to_string(index=False))
        print(lab.round(4).to_string(index=False))

    rp = real.assign(p=M.sigmoid(real.z / pol["temperature"]))
    spk = rp.groupby(["source", "speaker", "gender"]).agg(n=("p", "size"), median_p=("p", "median"),
                                                          called_synthetic=("p", lambda s: float((s >= pol["high"]).mean())))
    spk = spk.reset_index()
    spk.to_csv(out / "table_real_per_speaker.csv", index=False)
    print(f"\nreal median score: Kathbath clean {rp.p[rp.source == 'kathbath_clean'].median():.3f} | "
          f"Kathbath noisy {rp.p[rp.source == 'kathbath_noisy'].median():.3f} | CSALT test {np.median(csalt_real_p):.3f}")
    for src, d in spk.groupby("source"):
        print(f"{src}: {len(d)} speakers | called synthetic on >10% of clips: {(d.called_synthetic > 0.10).sum()} | "
              f">50%: {(d.called_synthetic > 0.50).sum()} (S17 on CSALT test was 78%)")
    print(f"\nsaved to {out}")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["download", "prepare", "freeze", "score", "report", "all", "index"])
    ap.add_argument("--run-name", default="xlsr300m-sls-v1")
    ap.add_argument("--keep-tars", action="store_true", help="keep the downloaded all-language archives")
    a = ap.parse_args()
    from sachai_audio.config import Config
    cfg = Config.load(ROOT / "runs" / a.run_name / "export" / "sachai_config.json")   # the model's own settings
    cfg.project_dir, cfg.run_name = str(ROOT), a.run_name
    stages = ["download", "prepare", "freeze", "score", "report"] if a.stage == "all" else [a.stage]
    for s in stages:
        print(f"\n===== external {s} =====", flush=True)
        if s == "index":
            build_index()
        elif s == "download":
            download_kathbath(a.keep_tars)
            download_indicsynth()
        elif s == "prepare":
            prepare(cfg)
        elif s == "freeze":
            freeze(cfg, pd.read_csv(MANIFEST))
        elif s == "score":
            score(cfg)
        elif s == "report":
            report(cfg)


if __name__ == "__main__":   # required on Windows: joblib workers re-import this file
    main()
