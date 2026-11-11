"""Measured acoustic descriptors and their reference distributions.

These numbers are context for the explanation, never the decision. The decision is the detector's.
Every descriptor is compared with what the training data looks like for genuine and synthetic Urdu
speech, so an explanation can say "lower than 9 in 10 genuine clips" instead of guessing.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import librosa
import numpy as np

DESCRIPTORS = {
    "pitch_var_st": "Pitch variation (std of pitch, semitones)",
    "pitch_step_st": "Frame-to-frame pitch movement (median, semitones)",
    "voiced_ratio": "Share of frames that are voiced",
    "pause_ratio": "Share of the clip spent in pauses",
    "pause_floor_db": "Background level inside pauses (dB below speech)",
    "hf_ratio_db": "Energy in 4 to 8 kHz relative to 0 to 4 kHz (dB)",
    "flatness": "Spectral flatness during speech",
}


def acoustic_descriptors(y: np.ndarray, sr: int = 16000) -> dict:
    """Seven interpretable descriptors on trimmed 16 kHz audio. NaN when a descriptor is undefined."""
    out = {k: float("nan") for k in DESCRIPTORS}
    if y.size < sr // 2:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y8 = librosa.resample(y, orig_sr=sr, target_sr=8000, res_type="soxr_hq")
        f0, voiced, _ = librosa.pyin(y8, fmin=70, fmax=400, sr=8000, frame_length=512, hop_length=128)
    voiced = voiced & np.isfinite(f0)
    out["voiced_ratio"] = float(voiced.mean())
    if voiced.sum() >= 10:
        st = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
        out["pitch_var_st"] = float(np.std(st))
        both = voiced[1:] & voiced[:-1]
        if both.sum() >= 5:
            d = np.abs(np.diff(12 * np.log2(np.where(voiced, f0, 1.0))))[both]
            out["pitch_step_st"] = float(np.median(d))

    hop = 160
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-10)
    speech_level = np.percentile(db, 90)
    pause = db < speech_level - 35
    out["pause_ratio"] = float(pause.mean())
    if pause.sum() >= 5:
        out["pause_floor_db"] = float(speech_level - np.median(db[pause]))

    spec = np.abs(librosa.stft(y, n_fft=512, hop_length=hop)) ** 2
    m = min(spec.shape[1], pause.size)
    speech_cols = ~pause[:m]
    if speech_cols.sum() >= 5:
        s = spec[:, :m][:, speech_cols]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        lo, hi = s[(freqs > 50) & (freqs < 4000)].sum(), s[freqs >= 4000].sum()
        out["hf_ratio_db"] = float(10 * np.log10((hi + 1e-12) / (lo + 1e-12)))
        flat = librosa.feature.spectral_flatness(S=np.sqrt(s))[0]
        out["flatness"] = float(np.median(flat))
    return out


def build_reference(clips: list[np.ndarray], labels: list[int], sr: int = 16000, n_jobs: int = -1) -> dict:
    """Stores the sorted descriptor values per class (from training speakers only)."""
    from joblib import Parallel, delayed

    feats = Parallel(n_jobs=n_jobs)(delayed(acoustic_descriptors)(c, sr) for c in clips)
    ref = {"bona_fide": {}, "synthetic": {}, "labels": DESCRIPTORS}
    labels = np.asarray(labels)
    for k in DESCRIPTORS:
        vals = np.array([f[k] for f in feats])
        for lab, name in ((0, "bona_fide"), (1, "synthetic")):
            v = vals[labels == lab]
            ref[name][k] = sorted(float(x) for x in v[np.isfinite(v)])
    return ref


def percentile_rank(value: float, sorted_vals: list) -> float | None:
    if not sorted_vals or not np.isfinite(value):
        return None
    return float(np.searchsorted(sorted_vals, value, side="right") / len(sorted_vals) * 100)


def compare(desc: dict, ref: dict) -> list[dict]:
    """For each descriptor: value, genuine percentile, class medians, and whether it is unusual for genuine speech."""
    rows = []
    for k, label in DESCRIPTORS.items():
        v = desc.get(k, float("nan"))
        gb, gs = ref["bona_fide"].get(k, []), ref["synthetic"].get(k, [])
        if not np.isfinite(v) or len(gb) < 10 or len(gs) < 10:
            continue
        pb = percentile_rank(v, gb)
        mb, ms = float(np.median(gb)), float(np.median(gs))
        closer = "synthetic" if abs(v - ms) < abs(v - mb) else "genuine"
        rows.append({"key": k, "label": label, "value": float(f"{v:.4g}"), "genuine_pct": round(pb, 1),
                     "genuine_median": float(f"{mb:.4g}"), "synthetic_median": float(f"{ms:.4g}"), "closer_to": closer,
                     "unusual_for_genuine": bool(pb < 10 or pb > 90),
                     "separation": round(abs(ms - mb) / (np.std(gb) + np.std(gs) + 1e-9), 3)})
    return rows


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (a.size * b.size))


def save_reference(ref: dict, path: Path) -> None:
    path.write_text(json.dumps(ref), encoding="utf-8")
