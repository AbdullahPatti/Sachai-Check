"""Paper-ready figures (PNG + PDF). Genuine is blue, synthetic is orange, everywhere."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e2dc"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out / f"{name}.png"


def learning_curves(history_csv: Path, out: Path) -> Path:
    h = pd.read_csv(history_csv)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    a.plot(h.epoch, h.train_loss_aug, color=MUTED, label="train (augmented)", linestyle="--")
    a.plot(h.epoch, h.train_probe_loss, color=BLUE, label="train (clean probe)")
    a.plot(h.epoch, h.dev_loss, color=ORANGE, label="dev (unseen speakers)")
    a.set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    a.legend(loc="best")
    b.plot(h.epoch, h.train_probe_eer * 100, color=BLUE, label="train (clean probe)")
    b.plot(h.epoch, h.dev_eer * 100, color=ORANGE, label="dev (unseen speakers)")
    best = h.loc[h.dev_eer.idxmin()]
    b.scatter([best.epoch], [best.dev_eer * 100], s=40, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
    b.annotate("selected", (best.epoch, best.dev_eer * 100), textcoords="offset points", xytext=(6, 6), color=INK)
    b.set(title="Equal error rate", xlabel="epoch", ylabel="EER (%)")
    b.legend(loc="best")
    for ax in (a, b):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return _save(fig, out, "fig_learning_curves")


def fit_diagnosis(history_csv: Path) -> str:
    h = pd.read_csv(history_csv)
    best = h.loc[h.dev_eer.idxmin()]
    gap = (best.dev_eer - best.train_probe_eer) * 100
    last = h.iloc[-1]
    msgs = [f"Selected epoch {int(best.epoch)}: train-probe EER {best.train_probe_eer*100:.2f}%, dev EER {best.dev_eer*100:.2f}% "
            f"(gap {gap:.2f} points)."]
    if best.train_probe_eer > 0.05:
        msgs.append("Train EER is still high: the model is underfitting. Train longer or raise lr_backbone.")
    elif gap > 5:
        msgs.append("Dev is much worse than train: overfitting to training speakers. Increase aug_prob or dropout.")
    else:
        msgs.append("Train and dev agree closely: no sign of under- or overfitting.")
    if last.dev_loss > h.dev_loss.min() * 1.3 and len(h) > 3:
        msgs.append("Dev loss rose after the best epoch; early stopping kept the best weights.")
    return " ".join(msgs)


def score_hist(scores_csv: Path, policy: dict, out: Path) -> Path:
    s = pd.read_csv(scores_csv)
    p = s["p"] if "p" in s else 1 / (1 + np.exp(-s.z / policy["temperature"]))
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.axvspan(policy["low"], policy["high"], color=GRID, alpha=0.7, lw=0, label="inconclusive band")
    bins = np.linspace(0, 1, 41)
    ax.hist(p[s.label == 0], bins=bins, color=BLUE, alpha=0.85, label="genuine", rwidth=0.9)
    ax.hist(p[s.label == 1], bins=bins, color=ORANGE, alpha=0.85, label="synthetic", rwidth=0.9)
    ax.set(xlabel="calibrated spoof score", ylabel="clips", title="Test score distribution", yscale="log")
    ax.legend(loc="upper center")
    return _save(fig, out, "fig_score_hist")


def roc_plot(scores_csv: Path, out: Path) -> Path:
    s = pd.read_csv(scores_csv)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    fpr, tpr, _ = roc_curve(s.label, s.z)
    ax.plot(fpr * 100, tpr * 100, color=ORANGE)
    i = np.argmin(np.abs(1 - tpr - fpr))
    ax.scatter([fpr[i] * 100], [tpr[i] * 100], s=40, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(f"EER {((fpr[i] + 1 - tpr[i]) / 2) * 100:.2f}%", (fpr[i] * 100, tpr[i] * 100), xytext=(8, -14),
                textcoords="offset points", color=INK)
    ax.plot([0, 100], [0, 100], color=GRID, lw=1)
    ax.set(xlabel="genuine called synthetic (%)", ylabel="synthetic detected (%)", title="ROC, test speakers")
    return _save(fig, out, "fig_roc")


def reliability(scores_csv: Path, out: Path, bins: int = 10) -> Path:
    s = pd.read_csv(scores_csv)
    edges = np.linspace(0, 1, bins + 1)
    mids, acc, cnt = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (s.p >= a) & (s.p < b if b < 1 else s.p <= b)
        if m.sum():
            mids.append(s.p[m].mean())
            acc.append(s.label[m].mean())
            cnt.append(m.sum())
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.plot(mids, acc, color=ORANGE, marker="o", markersize=5, markeredgecolor="white")
    ax.set(xlabel="predicted spoof probability", ylabel="observed share synthetic", title="Reliability, test")
    return _save(fig, out, "fig_reliability")


def robustness_plot(table_csv: Path, out: Path) -> Path:
    d = pd.read_csv(table_csv)
    fig, ax = plt.subplots(figsize=(4.8, 0.32 * len(d) + 0.8))
    ax.barh(d.condition[::-1], d.eer[::-1] * 100, color=ORANGE, height=0.6)
    for y, v in enumerate(d.eer[::-1] * 100):
        ax.text(v, y, f" {v:.2f}", va="center", color=INK, fontsize=8)
    ax.set(xlabel="EER (%)", title="Robustness on test speakers")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "fig_robustness")


def layer_weights_plot(export_dir: Path, out: Path) -> Path:
    from safetensors.torch import load_file
    w = load_file(str(Path(export_dir) / "model.safetensors"))["layer_logits"].float()
    w = (w.softmax(0) * 100).numpy()
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.bar(np.arange(len(w)), w, color=BLUE, width=0.7)
    ax.axhline(100 / len(w), color=MUTED, lw=1, linestyle="--")
    ax.text(len(w) - 0.5, 100 / len(w), "uniform", va="bottom", ha="right", color=MUTED, fontsize=8)
    ax.set(xlabel="XLS-R layer (0 = CNN output)", ylabel="weight (%)", title="Learned layer weights")
    ax.grid(axis="x", visible=False)
    return _save(fig, out, "fig_layer_weights")
