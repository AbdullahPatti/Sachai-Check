"""Sachai Check - Urdu fake-news text pipeline. Open this file and press F5.

Stages (config.STAGES): data -> audit -> baselines -> transformers -> judge -> report
Everything lands in outputs/runs/<timestamp>/ . To resume a crashed run: set env SACHAI_RUN=<that folder name>.
Finished experiments are skipped on resume, so a crash at hour 5 costs minutes, not hours.
"""
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def ensure_deps():
    need = {"pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy", "openpyxl": "openpyxl", "matplotlib": "matplotlib",
            "transformers": "transformers", "sentencepiece": "sentencepiece", "google.genai": "google-genai>=2.3.0"}
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod.split(".")[0]) is None]
    try:
        if tuple(map(int, importlib.metadata.version("google-genai").split(".")[:2])) < (2, 3):
            missing.append(need["google.genai"])  # Interactions API needs >= 2.3
    except importlib.metadata.PackageNotFoundError:
        missing.append(need["google.genai"])
    if missing:
        print("installing:", missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    if importlib.util.find_spec("torch") is None:
        sys.exit("PyTorch is not installed. Install the CUDA build first, e.g.\n"
                 "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n(see README.md)")


ensure_deps()

import pandas as pd  # noqa: E402

from sachai import config as C  # noqa: E402
from sachai import audit, data  # noqa: E402
from sachai.evaluate import dump  # noqa: E402

log = logging.getLogger("sachai")


def setup_run() -> Path:
    """Power-cut safe: with no SACHAI_RUN given, F5 continues the newest run that has no DONE marker."""
    runs = C.OUTPUTS / "runs"
    unfinished = sorted(d for d in runs.glob("*") if d.is_dir() and not (d / "logs" / "DONE").exists()
                        and d.name.endswith("-smoke") == C.SMOKE) if runs.exists() else []
    name = C.RUN_NAME or (unfinished[-1].name if unfinished and not C.FRESH else time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if C.SMOKE else ""))
    run = runs / name
    resuming = (run / "logs" / "pipeline.log").exists()
    for d in ("logs", "data", "audit", "baselines", "experiments", "judge", "report"):
        (run / d).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(run / "logs" / "pipeline.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    cfg = {k: v for k, v in vars(C).items() if k.isupper() and k != "GEMINI_API_KEY"}
    dump(cfg, run / "logs" / "config_snapshot.json")
    env = {"python": sys.version, "platform": platform.platform(), "gemini_key_set": bool(C.GEMINI_API_KEY)}
    try:
        import torch
        env |= {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if not torch.cuda.is_available() and not C.SMOKE:
            log.warning("NO GPU VISIBLE - transformer stage will be extremely slow. Check the CUDA build of torch.")
    except ImportError:
        pass
    env["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    dump(env, run / "logs" / "environment.json")
    log.info("%s run folder: %s", "RESUMING" if resuming else "NEW", run)
    return run


# ------------------------------------------------------------------ stages
def stage_data(run):
    f = run / "data" / "corpus.csv"
    if f.exists():
        return pd.read_csv(f, keep_default_na=False, na_values=[""])
    data.download_all()
    rep = {}
    df = data.split(data.build_corpus(rep))
    from sachai import judge
    unk = (df.split == "test") & (df.domain == "unknown")
    if judge.available() and not C.SMOKE and unk.any():
        log.info("tagging domains of %d test rows with Gemini (cached)", unk.sum())
        try:
            df.loc[unk, "domain"] = judge.annotate_domains(df.loc[unk, "text"].tolist())
        except Exception:
            log.error("domain tagging failed, continuing with 'unknown':\n%s", traceback.format_exc())
    rep["split_sizes"] = df.groupby(["source", "split", "label"]).size().unstack(fill_value=0).reset_index().to_dict("records")
    dump(rep, run / "data" / "data_report.json")
    df.to_csv(f.with_suffix(".tmp"), index=False)
    os.replace(f.with_suffix(".tmp"), f)  # atomic: a half-written corpus is never mistaken for a finished one
    return df


def stage_transformers(run, df):
    from sachai import train
    ood_sets = data.load_ood_sets()
    for exp in C.EXPERIMENTS:
        if exp["protocol"].startswith("lodo:") and exp["protocol"][5:] not in set(df.source):
            log.info("skip %s: source not present", exp["name"])
            continue
        try:
            train.run_experiment(exp, df, run / "experiments", ood_sets)
        except Exception:
            log.error("experiment %s FAILED, moving on:\n%s", exp["name"], traceback.format_exc())


def stage_judge(run, df):
    from sachai import infer
    infer.evaluate_system(run, df)


# ------------------------------------------------------------------ report
def _md(rows: list, cols: list) -> str:
    if not rows:
        return "_no results yet_\n"
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]) + "\n"


def stage_report(run, df):
    L = [f"# Sachai Check - text pipeline results ({run.name})\n"]
    dr = json.loads((run / "data" / "data_report.json").read_text(encoding="utf-8")) if (run / "data" / "data_report.json").exists() else {}
    L += ["## 1. Data\n", f"Raw rows: {dr.get('raw_rows')}\n\nClean rows (source -> {{label: n}}): {dr.get('clean_rows')}\n",
          f"Dropped: {dr.get('dropped')}\n\nCross-source story overlap (clusters): {dr.get('cross_source_overlap')}\n",
          f"Real/fake minimal pairs kept together: {dr.get('minimal_pair_clusters')}\n\nDebunk-framed rate (source -> label -> share): {dr.get('debunk_framed_rate')}\n"]
    au = run / "audit" / "audit.json"
    if au.exists():
        a = json.loads(au.read_text(encoding="utf-8"))
        L += ["## 2. Shortcut audit\n", "Length-only classifier AUROC (row = trained on, col = tested on). 0.5 = no signal; <0.5 = the length cue flips direction across datasets.\n\n",
              _md([{"train": k} | v for k, v in a["length_only_auroc_train_row_test_col"].items()], ["train"] + sorted(df.source.unique())),
              "\nTop class-indicative tokens per source are in audit/audit.json; length histogram in audit/length_by_class.png.\n"]

    rows = []
    for f in sorted((run / "baselines").glob("*.json")):
        b = json.loads(f.read_text(encoding="utf-8"))
        for t, r in b["results"].items():
            rows.append({"model": b["model"], "protocol": b["protocol"], "debunk": b["debunk"], "view": b["view"], "test": t,
                         "macro_f1": r["all"]["macro_f1"], "auroc": r["all"]["auroc"], "n_test": r["all"]["n"]})
    L += ["## 3. Classical baselines\n", _md(rows, ["model", "protocol", "debunk", "view", "test", "macro_f1", "auroc", "n_test"])]

    exps = {}
    for f in (run / "experiments").glob("*/metrics.json"):
        exps[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
    main = [e for n, e in exps.items() if n.startswith("main_s")]
    L.append("## 4. Transformers\n")
    if main:
        agg = {}
        for m in ("macro_f1", "auroc", "acc", "ece", "aurc"):
            v = [e["tests"]["pooled"]["all"][m] for e in main]
            agg[m] = f"{pd.Series(v).mean():.4f} ± {pd.Series(v).std(ddof=0):.4f}"
        L += [f"**Main model ({C.MAIN_MODEL}), pooled test, {len(main)} seeds (mean ± std):** {agg}\n\n"]
    rows = []
    for n, e in sorted(exps.items()):
        for t, r in e["tests"].items():
            rows.append({"experiment": n, "model": e["experiment"]["model"], "test": t, "macro_f1": r["all"]["macro_f1"],
                         "ci95": r["all"].get("macro_f1_ci"), "auroc": r["all"]["auroc"], "ece": r["all"]["ece"],
                         "minimal_pairs_f1": r.get("minimal_pairs", {}).get("macro_f1", ""), "mins": e["minutes"]})
    L.append(_md(rows, ["experiment", "model", "test", "macro_f1", "ci95", "auroc", "ece", "minimal_pairs_f1", "mins"]))

    if main:
        ref = next(n for n in sorted(exps) if n.startswith("main_s"))
        t = exps[ref]["tests"]["pooled"]
        L += [f"\n### Slices of {ref} (pooled test)\n", "**Per source**\n\n", _md([{"slice": k} | v for k, v in t["by_source"].items()], ["slice", "n", "macro_f1", "auroc"]),
              f"\n**Per domain (gate: macro-F1 >= {C.DOMAIN_F1_GATE})**\n\n",
              _md([{"slice": k, "gate": "PASS" if v["macro_f1"] >= C.DOMAIN_F1_GATE else "FAIL"} | v for k, v in t["by_domain"].items()], ["slice", "n", "macro_f1", "auroc", "gate"]),
              "\n**Per length quintile** (flat = no length shortcut)\n\n", _md([{"slice": k} | v for k, v in t["by_length_quintile"].items()], ["slice", "words_range", "n", "macro_f1"]),
              "\n**Per geography** (flat = no topic shortcut)\n\n", _md([{"slice": k} | v for k, v in t["by_geo"].items()], ["slice", "n", "fake_rate", "macro_f1"]),
              f"\n**Selective prediction** (only texts the OOD gate accepts): {t.get('in_gate_only')}\n",
              "\n### OOD detection (AUROC, higher = better; FPR@95TPR lower = better)\n\n",
              _md([{"ood_set": s, "method": m} | v for s, d in exps[ref]["ood"].items() for m, v in d.items() if isinstance(v, dict)], ["ood_set", "method", "auroc", "fpr_at_95tpr"])]
        near = {n: e["ood"] for n, e in exps.items() if n.startswith("lodo_")}
        if near:
            L += ["\nNear-OOD (whole unseen dataset, from LODO runs):\n\n",
                  _md([{"run": n, "set": s} | d["mahalanobis"] for n, o in near.items() for s, d in o.items() if s.startswith("near_")], ["run", "set", "auroc", "fpr_at_95tpr"])]

    jm = run / "judge" / "judge_metrics.json"
    L.append("\n## 5. LLM judge + full system\n")
    if jm.exists():
        j = json.loads(jm.read_text(encoding="utf-8"))
        for k in ("all", "manual", "test"):
            if k in j:
                v = j[k]
                L.append(f"**{k}** (n={v['n']}): classifier macro-F1 {v['classifier_only'].get('macro_f1')}, judge (decisive only) "
                         f"{v['judge_decisive'].get('macro_f1')} @ coverage {v['judge_decisive'].get('coverage')}, full system "
                         f"{v['system_final'].get('macro_f1')} @ coverage {v['system_final'].get('coverage')}; verdicts {v['judge_verdict_dist']}; "
                         f"meta-judge {v['meta_judge_mean']}\n\n")
        L.append(f"Roman-Urdu round trip: { {k: v for k, v in j.get('roman_roundtrip', {}).items() if k != 'examples'} }\n")
    else:
        L.append("_skipped (no GEMINI_API_KEY or no checkpoint)_\n")
    (run / "report" / "summary.md").write_text("\n".join(L), encoding="utf-8")
    log.info("report written: %s", run / "report" / "summary.md")


def main():
    run = setup_run()
    t0 = time.time()
    df = None
    for st in C.STAGES:
        ts = time.time()
        log.info("===== stage: %s =====", st)
        try:
            if st == "data":
                df = stage_data(run)
            elif st == "audit":
                audit.audit(df, run / "audit")
            elif st == "baselines":
                audit.baselines(df if not C.SMOKE else df.groupby("source").sample(frac=0.05, random_state=0), run / "baselines")
            elif st == "transformers":
                stage_transformers(run, df)
            elif st == "judge":
                stage_judge(run, df)
            elif st == "report":
                stage_report(run, df)
        except Exception:
            log.error("stage %s crashed:\n%s", st, traceback.format_exc())
            if st == "data":
                raise
        log.info("stage %s finished in %.1f min", st, (time.time() - ts) / 60)
    (run / "logs" / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("ALL DONE in %.1f h -> %s", (time.time() - t0) / 3600, run / "report" / "summary.md")


if __name__ == "__main__":
    main()
