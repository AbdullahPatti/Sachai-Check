"""Audio decoding, resampling, silence trimming and codec round trips.

The same functions run on training data and on user uploads, so the model
always sees audio prepared the same way.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def ffmpeg_encoders() -> set:
    exe = ffmpeg_exe()
    if not exe:
        return set()
    out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
    return {name for name in ("libmp3lame", "libopus", "aac", "libvorbis") if name in out}


def to_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=1) if y.ndim == 2 else y


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False)
    return librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=sr_out, res_type="soxr_hq").astype(np.float32)


def probe(path: str | Path) -> dict:
    """Codec, sample rate and channels of any file ffmpeg can open."""
    exe = ffmpeg_exe()
    info = {"codec": None, "sample_rate": None, "channels": None}
    if not exe:
        return info
    r = subprocess.run([exe, "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, text=True, timeout=30)
    m = re.search(r"Audio:\s*([\w\-]+)[^,]*,\s*(\d+)\s*Hz,\s*([\w\.\(\)]+)", r.stderr)
    if m:
        info.update(codec=m.group(1), sample_rate=int(m.group(2)), channels=m.group(3))
    return info


def decode_ffmpeg(path: str | Path, sr: int, timeout: float = 60.0) -> np.ndarray:
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg not available. Install imageio-ffmpeg.")
    cmd = [exe, "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise ValueError(f"Could not decode audio: {r.stderr.decode(errors='ignore')[-300:]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def load_audio(path: str | Path, sr: int = 16000, prefer_ffmpeg: bool = False) -> tuple[np.ndarray, int]:
    """Returns (mono float32 at `sr`, original sample rate)."""
    path = Path(path)
    if not prefer_ffmpeg:
        try:
            y, sr_in = sf.read(str(path), dtype="float32", always_2d=False)
            return resample(to_mono(y), sr_in, sr), sr_in
        except Exception:
            pass
    meta = probe(path)
    return decode_ffmpeg(path, sr), meta.get("sample_rate") or sr


def trim_silence(y: np.ndarray, top_db: float = 40.0) -> tuple[np.ndarray, int, int]:
    """Trims leading and trailing silence. Returns (trimmed, start, end) in samples."""
    if y.size == 0 or not np.any(np.abs(y) > 1e-6):
        return y[:0], 0, 0
    yt, (start, end) = librosa.effects.trim(y, top_db=top_db, frame_length=512, hop_length=128)
    return yt.astype(np.float32, copy=False), int(start), int(end)


def normalize(x: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per window (the XLS-R input convention). Removes loudness as a cue."""
    x = x.astype(np.float32, copy=False)
    return (x - x.mean()) / np.sqrt(x.var() + 1e-7)


def repeat_pad(x: np.ndarray, n: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros(n, dtype=np.float32)
    if x.size >= n:
        return x[:n]
    reps = int(np.ceil(n / x.size))
    return np.tile(x, reps)[:n]


def window_starts(n: int, win: int, hop: int) -> list[int]:
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] != n - win:
        starts.append(n - win)
    return starts


def codec_roundtrip(y: np.ndarray, sr: int, codec: str, bitrate: str) -> np.ndarray:
    """Encodes with a real codec (mp3 / opus / aac) and decodes back to `sr`."""
    exe = ffmpeg_exe()
    enc = {"mp3": ("libmp3lame", ".mp3"), "opus": ("libopus", ".ogg"), "aac": ("aac", ".m4a")}[codec]
    if enc[0] not in ffmpeg_encoders():
        raise RuntimeError(f"ffmpeg build has no {enc[0]} encoder")
    tmp = Path(tempfile.gettempdir()) / f"sachai_{uuid.uuid4().hex}"
    src, dst = tmp.with_suffix(".wav"), tmp.with_suffix(enc[1])
    try:
        sf.write(str(src), np.clip(y, -1, 1), sr, subtype="PCM_16")
        r = subprocess.run([exe, "-nostdin", "-v", "error", "-y", "-i", str(src), "-c:a", enc[0], "-b:a", bitrate, str(dst)],
                           capture_output=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="ignore")[-300:])
        out = decode_ffmpeg(dst, sr)
    finally:
        for p in (src, dst):
            try:
                os.remove(p)
            except OSError:
                pass
    n = y.size
    return out[:n] if out.size >= n else np.pad(out, (0, n - out.size))


def quality_report(y: np.ndarray, sr: int) -> dict:
    """Rough recording quality indicators used by the quality gate and the explanation."""
    if y.size == 0:
        return {"snr_db": None, "clipping_pct": 0.0, "speech_sec": 0.0}
    frame, hop = 512, 160
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-10)
    loud, quiet = np.percentile(db, 90), np.percentile(db, 10)
    speech_frames = db > (loud - 30)
    clipping = float(np.mean(np.abs(y) >= 0.999) * 100)
    return {
        "snr_db": float(round(loud - quiet, 1)),
        "clipping_pct": round(clipping, 3),
        "speech_sec": round(float(speech_frames.sum() * hop / sr), 2),
    }
