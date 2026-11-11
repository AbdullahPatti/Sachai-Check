"""Calibration, decision policy, OOD statistics, export, held-out test evaluation and robustness tests.

Order matters for honest numbers:
  1. finalize(): everything that is *fitted* (temperature, thresholds, OOD threshold, reference stats)
     uses train/dev speakers only.
  2. evaluate_test(): test speakers are scored once with the frozen policy.
  3. robustness(): the same frozen policy under degraded audio. No re-tuning per condition.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import metrics as M
from .audio_io import codec_roundtrip, ffmpeg_encoders, resample
from .augment import add_noise, reverb
from .config import POLICY_VERSION, PREPROCESS_VERSION, Config
from .data import load_clip
from .evidence import DESCRIPTORS, acoustic_descriptors, build_reference, cliffs_delta, save_reference
from .model import SachaiDetector
from .train import pick_device, score_clips


def _kept(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    return manifest[(manifest.split == split) & manifest.keep].reset_index(drop=True)


def model_fingerprint(model_dir: Path) -> str:
    """Cheap identity of a weights file, so caches are never reused across different models."""
    p = Path(model_dir) / "model.safetensors"
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 22))
        f.seek(max(0, size - (1 << 22)))
        h.update(f.read())
    return h.hexdigest()[:16]


def _cached_scores(path: Path, df: pd.DataFrame, fn, model_id: str) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        if (len(old) == len(df) and "model_id" in old and (old.model_id.astype(str) == model_id).all()
                and (old.path.values == df.path.values).all()):
            return old
    res = fn()
    out = df[["path", "speaker", "generator", "label", "group"]].copy()
    out["z"] = res["z"]
    out["model_id"] = model_id
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def latex_table(df: pd.DataFrame, caption: str, label: str, fmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{'l' + 'r' * (len(cols) - 1)}}}",
             "\\toprule", " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v).replace("_", "\\_") for v in r.values]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- finalize / export
def finalize(cfg: Config, manifest: pd.DataFrame, device: torch.device | None = None, n_ref: int = 400,
             force: bool = False) -> Path:
    """Skips if export/ already holds the current best weights; re-exports automatically after a retrain."""
    device = device or pick_device()
    best, exp, rep = cfg.run_dir / "best", cfg.export_dir, cfg.report_dir
    fp = model_fingerprint(best)
    if not force and (exp / "policy.json").exists() and (exp / "model.safetensors").exists() and model_fingerprint(exp) == fp:
        print(f"Export is up to date with the best weights ({fp}).")
        return exp
    exp.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    model = SachaiDetector.from_pretrained(best).to(device).eval()
    tr, dv = _kept(manifest, "train"), _kept(manifest, "dev")

    # 1. calibration + three-way policy on dev
    dev = _cached_scores(rep / "dev_scores.csv", dv, lambda: score_clips(model, dv.cache.tolist(), cfg, device, progress=True), fp)
    T = M.fit_temperature(dev.z.values, dev.label.values)
    p_dev = M.sigmoid(dev.z.values / T)
    policy = M.decision_policy(dev.label.values, p_dev, cfg.target_precision, cfg.min_band)
    policy.update(temperature=T, preprocess_version=PREPROCESS_VERSION, policy_version=POLICY_VERSION, model_id=fp,
                  dev_raw_ece=M.ece(dev.label.values, M.sigmoid(dev.z.values)),
                  dev_calibrated_ece=M.ece(dev.label.values, p_dev))
    dev_report = M.report(dev.label.values, p_dev, policy)

    # 2. OOD statistics: class-conditional Gaussians on clean train embeddings (Lee et al., 2018)
    ood_path = exp / "ood.npz"
    if not ood_path.exists() or str(np.load(ood_path).get("model_id", "")) != fp:
        from sklearn.covariance import LedoitWolf
        tr_s = score_clips(model, tr.cache.tolist(), cfg, device, return_embeddings=True, progress=True)
        dv_s = score_clips(model, dv.cache.tolist(), cfg, device, return_embeddings=True, progress=True)
        E, y = tr_s["embedding"], tr.label.values
        means = np.stack([E[y == c].mean(0) for c in (0, 1)])
        lw = LedoitWolf().fit(np.concatenate([E[y == c] - means[c] for c in (0, 1)]))
        prec = lw.precision_
        d_dev = mahalanobis_min(dv_s["embedding"], means, prec)
        np.savez(ood_path, means=means, precision=prec, threshold=np.percentile(d_dev, 99),
                 dev_median=np.median(d_dev), model_id=np.array(fp))

    # 3. reference descriptor distributions from train speakers
    ref_path = exp / "reference_stats.json"
    if not ref_path.exists():
        sub = pd.concat([g.sample(min(len(g), n_ref // 2), random_state=cfg.seed) for _, g in tr.groupby("label")])
        print(f"Measuring acoustic descriptors on {len(sub)} train clips ...")
        ref = build_reference([load_clip(c) for c in sub.cache], sub.label.tolist(), cfg.sample_rate)
        save_reference(ref, ref_path)

    # 4. copy weights and write config files
    for f in ("model.safetensors", "ssl_config.json", "head_config.json"):
        shutil.copy2(best / f, exp / f)
    cfg.save(exp / "sachai_config.json")
    (exp / "policy.json").write_text(json.dumps(policy, indent=2))
    meta = {"dev": dev_report, "best_epoch": json.loads((best / "best_epoch.json").read_text()),
            "splits": json.loads((cfg.root / "data" / "splits.json").read_text()),
            "n_clips": {s: int(len(_kept(manifest, s))) for s in ("train", "dev", "test")}}
    old = json.loads((exp / "metrics.json").read_text()) if (exp / "metrics.json").exists() else {}
    old.update(meta)
    (exp / "metrics.json").write_text(json.dumps(old, indent=2))
    print(f"Exported to {exp}\nTemperature {T:.3f} | thresholds low {policy['low']:.3f} "
          f"eer {policy['eer_threshold']:.3f} high {policy['high']:.3f} | dev EER {dev_report['eer']:.4f}")
    return exp


# ---------------------------------------------------------------- v1.1 policy options (dev only, no test)
POLICY_OPTIONS = {               # name: (max dev fakes called genuine, max dev reals called synthetic, safety margin)
    "strict": (0.005, 0.005, 0.30),
    "balanced": (0.01, 0.01, 0.15),
    "lenient": (0.02, 0.02, 0.05),
}


def _outcomes(dev: pd.DataFrame, verdicts: np.ndarray, p: np.ndarray) -> dict:
    real, fake, vits = (dev.label == 0).values, (dev.label == 1).values, (dev.generator == "vits").values
    pct = lambda m, v: round(100 * float((verdicts[m] == v).mean()), 1)
    return {"real_genuine": pct(real, "bona_fide"), "real_unsure": pct(real, "inconclusive"),
            "real_fake": pct(real, "synthetic"), "fake_genuine": pct(fake, "bona_fide"),
            "fake_unsure": pct(fake, "inconclusive"), "fake_fake": pct(fake, "synthetic"),
            "vits_genuine": pct(vits, "bona_fide"), "ece": round(M.ece(dev.label.values, p), 3)}


def _v1_rule(fit: pd.DataFrame, cfg: Config) -> dict:
    T = M.fit_temperature(fit.z.values, fit.label.values)
    pol = M.decision_policy(fit.label.values, M.sigmoid(fit.z.values / T), cfg.target_precision, cfg.min_band)
    return {**pol, "temperature": T}


def _v11_rule(fit: pd.DataFrame, opt: tuple) -> dict:
    a, b = M.fit_platt(fit.z.values, fit.label.values)
    lines = M.error_limit_lines(fit.label.values, fit.z.values, *opt)
    return {"platt_a": a, "platt_b": b, **lines,
            "low": float(M.sigmoid(a * lines["low_z"] + b)), "high": float(M.sigmoid(a * lines["high_z"] + b)),
            "eer_threshold": float(M.sigmoid(a * lines["eer_z"] + b))}


def _apply(rule: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p = M.calibrate(df.z.values, rule)
    return np.array([M.decide(v, rule) for v in p]), p


def policy_options(cfg: Config) -> pd.DataFrame:
    """Compare the v1 rule with v1.1 candidates on the dev speakers only. Never reads test or robustness scores.

    'dev fit' rows: fitted and measured on both dev speakers (optimistic).
    'unseen speaker' rows: fitted on one dev speaker, measured on the other, both ways, pooled. This is the
    closer guide to how a rule behaves on a new voice.
    """
    rep = cfg.report_dir
    dev = pd.read_csv(rep / "dev_scores.csv")
    mid = model_fingerprint(cfg.export_dir)
    if not (dev.model_id.astype(str) == mid).all():
        raise RuntimeError("dev_scores.csv does not match the exported weights. Run: python run.py finalize")
    spk = sorted(dev.speaker.unique())
    rules = {"v1 (current)": lambda f: _v1_rule(f, cfg)} | {
        f"v1.1 {k}": (lambda f, o=o: _v11_rule(f, o)) for k, o in POLICY_OPTIONS.items()}
    rows = []
    for name, make in rules.items():
        r = make(dev)
        v, p = _apply(r, dev)
        rows.append({"option": name, "measured_on": "dev fit", **_outcomes(dev, v, p),
                     "genuine_line": round(r["low"], 3), "synthetic_line": round(r["high"], 3)})
        parts = []
        for s in spk:
            held = dev[dev.speaker == s].reset_index(drop=True)
            rr = make(dev[dev.speaker != s].reset_index(drop=True))
            vv, pp = _apply(rr, held)
            parts.append((held, vv, pp))
        held = pd.concat([h for h, _, _ in parts], ignore_index=True)
        rows.append({"option": name, "measured_on": "unseen speaker",
                     **_outcomes(held, np.concatenate([v for _, v, _ in parts]), np.concatenate([p for _, _, p in parts]))})
    out = pd.DataFrame(rows)
    out.to_csv(rep / "policy_options_dev.csv", index=False)
    full = {k: _v11_rule(dev, o) for k, o in POLICY_OPTIONS.items()}
    (rep / "policy_options_dev.json").write_text(json.dumps(
        {"model_id": mid, "dev_speakers": spk, "options": {k: {"limits": dict(zip(("fake_as_real", "real_as_fake", "margin"), o)),
                                                               "rule": full[k]} for k, o in POLICY_OPTIONS.items()}}, indent=2))
    return out


def mahalanobis_min(E: np.ndarray, means: np.ndarray, prec: np.ndarray) -> np.ndarray:
    E = np.atleast_2d(E)
    d = [np.einsum("nd,dk,nk->n", E - m, prec, E - m) for m in means]
    return np.sqrt(np.maximum(np.min(np.stack(d), axis=0), 0))


def load_policy(export_dir: Path) -> dict:
    return json.loads((Path(export_dir) / "policy.json").read_text())


# ---------------------------------------------------------------- test
def evaluate_test(cfg: Config, manifest: pd.DataFrame, device: torch.device | None = None, n_boot: int = 1000) -> dict:
    device = device or pick_device()
    exp, rep = cfg.export_dir, cfg.report_dir
    policy = load_policy(exp)
    te = _kept(manifest, "test")
    model = SachaiDetector.from_pretrained(exp).to(device).eval()
    t0 = time.time()
    mid = model_fingerprint(exp)
    sc = _cached_scores(rep / "test_scores.csv", te, lambda: score_clips(model, te.cache.tolist(), cfg, device, progress=True), mid)
    p = M.sigmoid(sc.z.values / policy["temperature"])
    sc["p"] = p
    sc["decision"] = [M.decide(v, policy) for v in p]
    sc.to_csv(rep / "test_scores.csv", index=False)
    y = sc.label.values

    main = M.report(y, p, policy)
    main["ci95"] = M.grouped_bootstrap(y, p, sc.group.values, policy, n_boot=n_boot)

    gen_rows = []
    for g in ("tacotron", "vits"):
        m = (sc.generator == "bonafide") | (sc.generator == g)
        if (sc.generator == g).any():
            r = M.report(y[m], p[m], policy)
            gen_rows.append({"attack": g, "n_spoof": int((sc.generator == g).sum()), "eer": r["eer"], "auc": r["auc"],
                             "spoof_detected": float((sc.decision[sc.generator == g] == "synthetic").mean()),
                             "spoof_inconclusive": float((sc.decision[sc.generator == g] == "inconclusive").mean())})
    spk_rows = []
    for s, g in sc.groupby("speaker"):
        r = M.report(g.label.values, g.p.values, policy)
        spk_rows.append({"speaker": s, "n": len(g), "eer": r["eer"], "auc": r["auc"], "accuracy": r["accuracy"],
                         "coverage": r["coverage"]})
    gen_df, spk_df = pd.DataFrame(gen_rows), pd.DataFrame(spk_rows)
    gen_df.to_csv(rep / "test_per_attack.csv", index=False)
    spk_df.to_csv(rep / "test_per_speaker.csv", index=False)

    main_tbl = pd.DataFrame([{
        "split": "dev", **{k: json.loads((exp / "metrics.json").read_text())["dev"][k] for k in ("eer", "auc", "accuracy", "macro_f1", "coverage", "ece")}},
        {"split": "test", **{k: main[k] for k in ("eer", "auc", "accuracy", "macro_f1", "coverage", "ece")}}])
    main_tbl.to_csv(rep / "table_main.csv", index=False)
    (rep / "table_main.tex").write_text(latex_table(main_tbl, "Detection results on speaker-disjoint dev and test sets "
                                                    "(synthetic is the positive class; coverage is the share of clips "
                                                    "outside the inconclusive band).", "tab:audio-main"))
    (rep / "table_per_attack.tex").write_text(latex_table(gen_df, "Test results per spoofing attack.", "tab:audio-attack"))

    out = {"test": main, "test_per_attack": gen_rows, "test_per_speaker": spk_rows, "policy": policy,
           "scored_in_sec": round(time.time() - t0, 1)}
    allm = json.loads((exp / "metrics.json").read_text())
    allm.update(out)
    (exp / "metrics.json").write_text(json.dumps(allm, indent=2))
    (rep / "metrics_test.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- robustness
def robustness_conditions(sr: int) -> dict:
    enc = ffmpeg_encoders()

    def noise(snr):
        return lambda y, ci: add_noise(y.astype(np.float64), np.random.default_rng([777, ci]), snr, "pink").astype(np.float32)

    def rev(rt):
        return lambda y, ci: reverb(y.astype(np.float64), sr, np.random.default_rng([778, ci]), rt).astype(np.float32)

    conds = {"clean": None, "noise_20db": noise(20), "noise_10db": noise(10), "noise_5db": noise(5),
             "reverb_rt60_0.3s": rev(0.3), "reverb_rt60_0.8s": rev(0.8),
             "telephone_8khz": lambda y, ci: resample(resample(y, sr, 8000), 8000, sr)[: y.size]}
    if "libmp3lame" in enc:
        conds["mp3_64kbps"] = lambda y, ci: codec_roundtrip(y, sr, "mp3", "64k")
        conds["mp3_32kbps"] = lambda y, ci: codec_roundtrip(y, sr, "mp3", "32k")
    if "libopus" in enc:
        conds["opus_16kbps_voice_note"] = lambda y, ci: codec_roundtrip(y, sr, "opus", "16k")
    return conds


def robustness(cfg: Config, manifest: pd.DataFrame, device: torch.device | None = None, only: list | None = None) -> pd.DataFrame:
    """Every condition is cached separately, so an interrupted run continues where it stopped."""
    device = device or pick_device()
    exp, rep = cfg.export_dir, cfg.report_dir / "robustness"
    rep.mkdir(parents=True, exist_ok=True)
    policy = load_policy(exp)
    te = _kept(manifest, "test")
    model = SachaiDetector.from_pretrained(exp).to(device).eval()
    mid = model_fingerprint(exp)
    rows = []
    for name, tf in robustness_conditions(cfg.sample_rate).items():
        if only and name not in only:
            continue
        print(f"robustness: {name}")
        sc = _cached_scores(rep / f"{name}.csv", te,
                            lambda: score_clips(model, te.cache.tolist(), cfg, device, transform=tf, progress=True), mid)
        p = M.sigmoid(sc.z.values / policy["temperature"])
        r = M.report(sc.label.values, p, policy)
        rows.append({"condition": name, "eer": r["eer"], "auc": r["auc"], "accuracy": r["accuracy"],
                     "far": r["far_bona_fide_called_synthetic"], "frr": r["frr_synthetic_missed"],
                     "coverage": r["coverage"]})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.report_dir / "table_robustness.csv", index=False)
    (cfg.report_dir / "table_robustness.tex").write_text(latex_table(
        df, "Test results under degraded audio with the frozen dev policy. Codecs were never used in training.",
        "tab:audio-robust"))
    return df


# ---------------------------------------------------------------- research analyses
def descriptor_analysis(cfg: Config, manifest: pd.DataFrame, n_per_group: int = 150) -> pd.DataFrame:
    """Which measurable properties separate genuine from synthetic Urdu speech (test speakers, per attack)."""
    te = _kept(manifest, "test")
    parts = [g.sample(min(len(g), n_per_group), random_state=cfg.seed) for _, g in te.groupby("generator")]
    sub = pd.concat(parts).reset_index(drop=True)
    from joblib import Parallel, delayed
    feats = Parallel(n_jobs=-1)(delayed(acoustic_descriptors)(load_clip(c), cfg.sample_rate) for c in sub.cache)
    F = pd.concat([sub[["generator", "label"]], pd.DataFrame(feats)], axis=1)
    rows = []
    for k, label in DESCRIPTORS.items():
        b = F.loc[F.generator == "bonafide", k].dropna().values
        row = {"descriptor": label, "genuine_median": float(np.median(b)) if b.size else np.nan}
        for g in ("tacotron", "vits"):
            s = F.loc[F.generator == g, k].dropna().values
            row[f"{g}_median"] = float(np.median(s)) if s.size else np.nan
            row[f"cliffs_delta_{g}"] = cliffs_delta(s, b)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.report_dir / "table_descriptors.csv", index=False)
    (cfg.report_dir / "table_descriptors.tex").write_text(latex_table(
        df, "Acoustic descriptors on test speakers. Cliff's delta compares each attack with genuine speech "
            "(positive means higher in synthetic).", "tab:audio-desc", fmt="{:.3f}"))
    F.to_csv(cfg.report_dir / "descriptors_test_raw.csv", index=False)
    return df
