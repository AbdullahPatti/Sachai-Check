"""Training-time channel augmentation, applied identically to both classes.

Goal: the detector should learn generation artefacts, not the recording
channel. Every transform here is label-blind, so it cannot leak the label.
Ideas follow RawBoost (Tak et al., 2022): linear convolutive coloration,
impulsive signal-dependent noise and stationary additive noise, plus
band-limiting, mu-law quantisation and synthetic reverberation to imitate
phone lines, messaging apps and rooms. Real codecs (MP3, Opus) are kept out
of training on purpose and used only in the robustness test.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def colored_noise(n: int, rng: np.random.Generator, color: str) -> np.ndarray:
    white = rng.standard_normal(n)
    if color == "white":
        return white
    spec = np.fft.rfft(white)
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    spec = spec / (np.sqrt(f) if color == "pink" else f)
    out = np.fft.irfft(spec, n)
    return out / (out.std() + 1e-12)


def add_noise(y: np.ndarray, rng: np.random.Generator, snr_db: float, color: str = "pink") -> np.ndarray:
    noise = colored_noise(y.size, rng, color)
    scale = _rms(y) / (10 ** (snr_db / 20) * _rms(noise))
    return y + scale * noise


def synthetic_rir(sr: int, rt60: float, rng: np.random.Generator) -> np.ndarray:
    n = max(int(rt60 * sr), 32)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)
    h[0] = 1.0 / max(np.abs(h).max(), 1e-6)
    return h / np.sqrt(np.sum(h ** 2))


def reverb(y: np.ndarray, sr: int, rng: np.random.Generator, rt60: float) -> np.ndarray:
    wet = signal.fftconvolve(y, synthetic_rir(sr, rt60, rng))[: y.size]
    return wet * (_rms(y) / (_rms(wet) + 1e-12))


def bandlimit(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    hi = float(rng.choice([3400.0, 4000.0, 5500.0, 7000.0]))
    lo = float(rng.choice([0.0, 100.0, 300.0]))
    if lo > 0:
        sos = signal.butter(6, [lo, hi], btype="band", fs=sr, output="sos")
    else:
        sos = signal.butter(6, hi, btype="low", fs=sr, output="sos")
    return signal.sosfilt(sos, y)


def mulaw(y: np.ndarray, bits: int = 8) -> np.ndarray:
    peak = np.abs(y).max() + 1e-9
    x = y / peak
    mu = 2 ** bits - 1
    c = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((c + 1) / 2 * mu) / mu * 2 - 1
    return np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu * peak


def peaking_eq(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """RawBoost-style linear convolutive coloration: a few random peaking filters."""
    for _ in range(int(rng.integers(1, 4))):
        f0 = rng.uniform(100, 0.47 * sr)
        gain_db = rng.uniform(3, 12) * rng.choice([-1, 1])
        q = rng.uniform(0.5, 3.0)
        a = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * f0 / sr
        alpha = np.sin(w0) / (2 * q)
        b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
        den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
        y = signal.lfilter(b, den, y)
    return y


def impulsive(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """RawBoost-style impulsive signal-dependent noise."""
    out = y.copy()
    p = rng.uniform(0.01, 0.08)
    idx = rng.random(y.size) < p
    out[idx] += y[idx] * rng.uniform(-1, 1, idx.sum()) * rng.uniform(0.2, 1.0)
    return out


OPS = ("noise", "reverb", "bandlimit", "mulaw", "eq", "impulsive")


def augment(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Applies 1 or 2 random label-blind transforms."""
    x = y.astype(np.float64)
    for op in rng.choice(OPS, size=int(rng.integers(1, 3)), replace=False):
        if op == "noise":
            x = add_noise(x, rng, rng.uniform(5, 30), rng.choice(["white", "pink", "brown"]))
        elif op == "reverb":
            x = reverb(x, sr, rng, rng.uniform(0.15, 0.8))
        elif op == "bandlimit":
            x = bandlimit(x, sr, rng)
        elif op == "mulaw":
            x = mulaw(x)
        elif op == "eq":
            x = peaking_eq(x, sr, rng)
        elif op == "impulsive":
            x = impulsive(x, rng)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)
