"""Metrics, calibration and the decision policy. Synthetic is the positive class everywhere."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def eer(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the score threshold where it occurs."""
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def fit_temperature(z: np.ndarray, y: np.ndarray, lo: float = 0.2, hi: float = 10.0) -> float:
    """Temperature scaling (Guo et al., 2017) on spoof log-odds, fitted on dev only."""
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def nll(log_t):
        p = np.clip(sigmoid(z / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    r = minimize_scalar(nll, bounds=(np.log(lo), np.log(hi)), method="bounded")
    return float(np.exp(r.x))


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def decision_policy(y_dev, p_dev, target_precision: float = 0.99, min_band: float = 0.10, min_support: int = 20) -> dict:
    """Three-way policy fitted on calibrated dev probabilities.

    bona fide if p <= low, synthetic if p >= high, inconclusive in between.
    low / high are the widest cut-offs that keep each decided side at >= target precision on dev,
    and the band is never narrower than +-min_band around the EER threshold.
    """
    y, p = np.asarray(y_dev), np.asarray(p_dev)
    e, t_eer = eer(y, p)
    order = np.sort(np.unique(p))
    low, high = None, None
    for t in order:                         # largest t keeping bona fide side clean
        m = p <= t
        if m.sum() >= min_support and (y[m] == 0).mean() >= target_precision:
            low = float(t)
    for t in order[::-1]:                   # smallest t keeping synthetic side clean
        m = p >= t
        if m.sum() >= min_support and (y[m] == 1).mean() >= target_precision:
            high = float(t)
    low = t_eer - min_band if low is None else min(low, t_eer - min_band)
    high = t_eer + min_band if high is None else max(high, t_eer + min_band)
    low, high = float(np.clip(low, 0.01, t_eer)), float(np.clip(high, t_eer, 0.99))
    return {"eer_threshold": float(t_eer), "low": low, "high": high, "dev_eer": e,
            "target_precision": target_precision, "min_band": min_band}


# ---------------------------------------------------------------- v1.1 policy (scale + shift, error-limit band)
def fit_platt(z: np.ndarray, y: np.ndarray, a_bounds: tuple = (0.1, 5.0)) -> tuple[float, float]:
    """Platt scaling p = sigmoid(a * z + b) on dev only. Unlike temperature scaling it can shift the scores.

    Uses Platt's smoothed targets, and `a` stays inside the same range the temperature fit allows (T 0.2 to 10),
    so near-separable dev data cannot push every score to 0 or 1.
    """
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    t = np.where(y == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))

    def nll(w):
        q = np.clip(sigmoid(w[0] * z + w[1]), 1e-7, 1 - 1e-7)
        return -np.mean(t * np.log(q) + (1 - t) * np.log(1 - q))

    a0 = 1.0 / fit_temperature(z, y)
    r = minimize(nll, x0=np.array([a0, 0.0]), bounds=[a_bounds, (-50.0, 50.0)], method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


def calibrate(z, policy: dict) -> np.ndarray:
    """Raw spoof log-odds to a calibrated spoof probability. v1 policies hold `temperature`, v1.1 hold `platt_a/b`."""
    z = np.asarray(z, dtype=np.float64)
    if "platt_a" in policy:
        return sigmoid(policy["platt_a"] * z + policy["platt_b"])
    return sigmoid(z / policy["temperature"])


def error_limit_lines(y, z, fake_as_real: float, real_as_fake: float, margin: float) -> dict:
    """Three-way lines on raw dev log-odds, set from error limits instead of a fixed band width.

    genuine line: as high as possible while at most `fake_as_real` of dev fakes fall at or below it.
    synthetic line: as low as possible while at most `real_as_fake` of dev genuine clips fall at or above it.
    If the two lines cross they meet at the EER threshold. Then both move out by `margin` (raw log-odds units),
    a safety gap for speaker shift, which dev (2 speakers) cannot measure well.
    """
    y, z = np.asarray(y).astype(int), np.asarray(z, dtype=np.float64)
    zf, zr = np.sort(z[y == 1]), np.sort(z[y == 0])[::-1]
    _, t = eer(y, z)
    kf, kr = int(np.floor(fake_as_real * len(zf))), int(np.floor(real_as_fake * len(zr)))
    low = float(np.nextafter(zf[min(kf, len(zf) - 1)], -np.inf))
    high = float(np.nextafter(zr[min(kr, len(zr) - 1)], np.inf))
    low, high = min(low, t), max(high, t)
    return {"eer_z": float(t), "low_z": low - margin, "high_z": high + margin}


def decide_z(z, lines: dict) -> np.ndarray:
    """Same rule as `decide`, on raw log-odds (any monotone calibration gives identical verdicts)."""
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= lines["high_z"], "synthetic", np.where(z <= lines["low_z"], "bona_fide", "inconclusive"))


def decide(p: float, policy: dict) -> str:
    if p >= policy["high"]:
        return "synthetic"
    if p <= policy["low"]:
        return "bona_fide"
    return "inconclusive"


def report(y, p, policy: dict) -> dict:
    """Full metric set for one population. p are calibrated spoof probabilities."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=np.float64)
    e, _ = eer(y, p)
    hard = (p >= policy["eer_threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, hard, labels=[0, 1]).ravel()
    dec = np.array([decide(v, policy) for v in p])
    decided = dec != "inconclusive"
    pred_dec = (dec[decided] == "synthetic").astype(int)
    n_dec = int(decided.sum())
    return {
        "n": int(len(y)), "n_bona_fide": int((y == 0).sum()), "n_synthetic": int((y == 1).sum()),
        "eer": e, "auc": auc(y, p),
        "accuracy": float((hard == y).mean()),
        "macro_f1": float(f1_score(y, hard, average="macro", zero_division=0)),
        "far_bona_fide_called_synthetic": float(fp / max(fp + tn, 1)),
        "frr_synthetic_missed": float(fn / max(fn + tp, 1)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "coverage": float(decided.mean()),
        "selective_risk": float((pred_dec != y[decided]).mean()) if n_dec else float("nan"),
        "n_inconclusive": int((~decided).sum()),
        "ece": ece(y, p), "brier": brier(y, p),
    }


def grouped_bootstrap(y, p, groups, policy: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% CIs, resampling utterance groups (a genuine sentence and its clones move together)."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = {"eer": [], "auc": [], "macro_f1": []}
    thr = policy["eer_threshold"]
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        stats["eer"].append(eer(yy, pp)[0])
        stats["auc"].append(auc(yy, pp))
        stats["macro_f1"].append(f1_score(yy, (pp >= thr).astype(int), average="macro", zero_division=0))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items() if v}
