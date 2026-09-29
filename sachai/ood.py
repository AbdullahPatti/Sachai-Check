"""Out-of-scope / out-of-distribution handling.

Three layers, cheapest first:
  1. input_gate  - rule-based bounds: empty, too short, not Urdu script, Roman Urdu (-> transliterate), too long.
  2. distribution - Mahalanobis distance on the fine-tuned encoder's [CLS] (Lee et al., 2018), plus MSP and
                    energy (Liu et al., 2020) as baselines. Threshold set so 95% of in-distribution val passes.
  3. topical scope - "is this about Pakistan/South Asia news?" asked to the LLM judge (judge.py).
"""
import re

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from . import config as C
from .data import normalize, urdu_ratio

_LATIN = re.compile(r"[A-Za-z]")
# Frequent Roman-Urdu function words; English text rarely contains several of them.
_ROMAN_URDU = re.compile(r"\b(hai|hain|ka|ki|ke|ko|se|mein|main|nahi|nahin|aur|ne|ho|kya|tha|thi|ye|yeh|woh|wo|bhi|par|pe|kar|raha|rahi)\b", re.I)
MAX_WORDS = 1000


def input_gate(text: str) -> dict:
    t = normalize(text or "")
    words = t.split()
    latin = len(_LATIN.findall(t)) / max(sum(c.isalpha() for c in t), 1)
    if not words:
        status = "empty"
    elif len(words) < C.MIN_WORDS:
        status = "too_short"
    elif urdu_ratio(t) >= C.MIN_URDU_RATIO:
        status = "ok"
    elif latin > 0.8 and len(_ROMAN_URDU.findall(t)) >= max(2, 0.15 * len(words)):
        status = "roman_urdu"
    else:
        status = "not_urdu"
    return {"status": status, "text": " ".join(words[:MAX_WORDS]), "truncated": len(words) > MAX_WORDS,
            "urdu_ratio": round(urdu_ratio(t), 3)}


def fit(emb: np.ndarray, y: np.ndarray) -> dict:
    means = np.stack([emb[y == c].mean(0) for c in (0, 1)])
    centered = emb - means[y]
    return {"means": means, "precision": LedoitWolf().fit(centered).precision_}


def mahalanobis(emb: np.ndarray, stats: dict) -> np.ndarray:
    d = [np.einsum("ij,jk,ik->i", emb - m, stats["precision"], emb - m) for m in stats["means"]]
    return np.min(d, axis=0)


def scores(emb, logits, stats) -> dict:
    """Higher = more out-of-distribution, for every method."""
    return {"mahalanobis": mahalanobis(emb, stats), "energy": -logsumexp(logits, axis=1), "msp": 1 - softmax(logits, axis=1).max(1)}


def threshold(val_scores: np.ndarray) -> float:
    return float(np.quantile(val_scores, C.OOD_TPR))


def evaluate(id_scores: dict, ood_scores: dict, thr: float) -> dict:
    """AUROC and FPR@95%TPR for each (OOD set, method); plus how often the deployed Mahalanobis gate fires."""
    out = {}
    for name, sc in ood_scores.items():
        out[name] = {}
        for m in id_scores:
            a, b = id_scores[m], sc[m]
            t95 = np.quantile(a, C.OOD_TPR)
            out[name][m] = {"auroc": round(float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])), 4),
                            "fpr_at_95tpr": round(float((b <= t95).mean()), 4)}
        out[name]["gate_flag_rate"] = round(float((sc["mahalanobis"] > thr).mean()), 4)
    return out
