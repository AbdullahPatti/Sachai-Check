"""Metrics shared by baselines, transformers and the judge. p = probability of FAKE (label 1)."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from . import config as C


def ece(y, p, bins=15):
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == y
    idx = np.minimum((conf - 0.5) * 2 * bins, bins - 1).astype(int)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def aurc(y, p):
    """Area under risk-coverage: error rate when abstaining on the least confident first (lower = better)."""
    order = np.argsort(-np.maximum(p, 1 - p))
    err = ((p >= 0.5) != y)[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def metrics(y, p, ci=False) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    yhat = (p >= 0.5).astype(int)
    pr, rc, f, _ = precision_recall_fscore_support(y, yhat, labels=[0, 1], zero_division=0)
    two = len(np.unique(y)) == 2
    m = dict(n=len(y), fake_rate=float(y.mean()), acc=accuracy_score(y, yhat), macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
             f1_fake=f[1], f1_real=f[0], prec_fake=pr[1], rec_fake=rc[1], prec_real=pr[0], rec_real=rc[0],
             mcc=matthews_corrcoef(y, yhat) if two else float("nan"), auroc=roc_auc_score(y, p) if two else float("nan"),
             ece=ece(y, p), brier=brier_score_loss(y, p, pos_label=1) if two else float("nan"), aurc=aurc(y, p))
    if ci and two:
        rng = np.random.default_rng(0)
        boots = [(f1_score(y[i], yhat[i], average="macro", zero_division=0), roc_auc_score(y[i], p[i]) if len(np.unique(y[i])) == 2 else np.nan)
                 for i in (rng.integers(0, len(y), len(y)) for _ in range(C.BOOTSTRAP))]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        m.update(macro_f1_ci=[lo[0], hi[0]], auroc_ci=[lo[1], hi[1]])
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def sliced(df: pd.DataFrame, p) -> dict:
    """Headline metrics + the slices that expose shortcuts: per source, per domain, per length quintile, minimal pairs."""
    df = df.assign(p=np.asarray(p), nw=df["text"].str.split().str.len())
    out = {"all": metrics(df.label, df.p, ci=True)}
    out["by_source"] = {k: metrics(g.label, g.p) for k, g in df.groupby("source")}
    out["by_domain"] = {k: metrics(g.label, g.p) for k, g in df.groupby("domain") if len(g) >= 20 and g.label.nunique() == 2}
    q = pd.qcut(df.nw.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    out["by_length_quintile"] = {str(k): metrics(g.label, g.p) | {"words_range": [int(g.nw.min()), int(g.nw.max())]} for k, g in df.groupby(q, observed=True)}
    if "pair" in df and df.pair.any():
        out["minimal_pairs"] = metrics(df[df.pair].label, df[df.pair].p)
    if "debunk" in df and df.debunk.any():
        out["claim_style_only"] = metrics(df[~df.debunk].label, df[~df.debunk].p)
    # Topic shortcut check: HnB fakes are mostly translated US fact-checks, reals mostly Pakistani outlets.
    geo = np.select([df.text.str.contains(_PK), df.text.str.contains(_IN)], ["pakistan", "india"], "elsewhere")
    out["by_geo"] = {k: metrics(g.label, g.p) for k, g in df.groupby(geo) if g.label.nunique() == 2}
    return out


_PK = "پاکستان|لاہور|کراچی|اسلام آباد|پنجاب|سندھ|پشاور|بلوچستان|خیبر|راولپنڈی|کوئٹہ|ملتان|فیصل آباد"
_IN = "بھارت|ہندوستان|بھارتی|دہلی|ممبئی|مودی|کشمیر|بی جے پی|اتر پردیش|بنگال"


def dump(obj, path):
    """Atomic JSON write (tmp + rename): a power cut can never leave a half-written results file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
