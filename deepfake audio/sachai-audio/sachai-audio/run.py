"""Command-line runner for the whole audio pipeline (same steps as the notebook).

    python run.py check          environment, GPU, ffmpeg codecs, keys
    python run.py data           download + manifest + frozen split + 16 kHz cache   (resumable)
    python run.py audit          shortcut / leakage audit
    python run.py train          fine-tune XLS-R                                       (resumable)
    python run.py fit            learning curves + under/overfitting diagnosis
    python run.py finalize       calibration, decision band, OOD stats, reference stats, export
    python run.py options        v1 vs v1.1 decision rules on dev speakers only (no test)
    python run.py test           held-out test speakers, frozen policy                 (cached)
    python run.py robustness     noise / reverb / phone / MP3 / Opus                   (cached per condition)
    python run.py analysis       layer weights + acoustic descriptor table
    python run.py demo           one test clip through detector + explanation, latency
    python run.py push           upload export/ to the Hugging Face Hub (HF_TOKEN in .env)
    python run.py all            data -> audit -> train -> fit -> finalize -> test -> robustness -> analysis -> demo

Every stage is safe to re-run: finished work is skipped, interrupted work resumes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STAGES = ["check", "data", "audit", "train", "fit", "finalize", "options", "test", "robustness", "analysis", "demo", "push"]


def build_cfg(a):
    from sachai_audio.config import Config
    over = {k: v for k, v in {"batch_size": a.batch_size, "grad_accum": a.grad_accum, "num_workers": a.workers,
                              "max_epochs": a.max_epochs}.items() if v is not None}
    return Config(project_dir=str(ROOT), run_name=a.run_name, **over)


def load_manifest(cfg):
    import pandas as pd
    p = cfg.root / "data" / "manifest.csv"
    if not p.exists():
        sys.exit("data/manifest.csv missing. Run: python run.py data")
    return pd.read_csv(p)


def stage_check(cfg, a):
    import os
    import torch
    from sachai_audio.audio_io import ffmpeg_encoders, ffmpeg_exe
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 1024**3:.1f} GB, bf16: {torch.cuda.is_bf16_supported()}")
    else:
        print("WARNING: no CUDA GPU. Reinstall torch from https://download.pytorch.org/whl/cu132 and update the NVIDIA driver.")
    print(f"ffmpeg: {ffmpeg_exe()} | encoders: {sorted(ffmpeg_encoders())}")
    free = shutil.disk_usage(ROOT).free / 1024**3
    print(f"free disk: {free:.0f} GB (need about 15 GB)")
    for k in ("HF_TOKEN", "LLM_API_KEY"):
        print(f"{k}: {'set' if os.environ.get(k) else 'not set'}")
    print(f"config: batch {cfg.batch_size} x accum {cfg.grad_accum}, workers {cfg.num_workers}, run '{cfg.run_name}'")


def stage_data(cfg, a):
    from sachai_audio import data
    p = cfg.root / "data" / "manifest.csv"
    if p.exists() and not a.force:
        print("manifest exists, data stage already done")
    else:
        data.download_dataset(cfg)
        data.preprocess_all(data.make_splits(data.build_manifest(cfg), cfg), cfg)
    m = load_manifest(cfg)
    print(json.loads((cfg.root / "data" / "splits.json").read_text()))
    print(m[m.keep].groupby(["split", "generator"]).size().unstack())


def stage_audit(cfg, a):
    from sachai_audio import data
    tbl, leak = data.audit(load_manifest(cfg))
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(cfg.report_dir / "data_audit.csv", index=False)
    (cfg.report_dir / "data_audit.json").write_text(json.dumps(leak, indent=2))
    print(tbl.round(3).to_string())
    print(leak)


def stage_train(cfg, a):
    from sachai_audio import train as T
    print(T.train(cfg, load_manifest(cfg), force_restart=a.force_retrain))


def stage_fit(cfg, a):
    from sachai_audio import plots as P
    print(P.learning_curves(cfg.run_dir / "history.csv", cfg.report_dir))
    print(P.fit_diagnosis(cfg.run_dir / "history.csv"))


def stage_finalize(cfg, a):
    from sachai_audio import evaluate as E
    E.finalize(cfg, load_manifest(cfg), force=a.force)      # re-exports automatically if the best weights changed
    print(json.loads((cfg.export_dir / "policy.json").read_text()))


def stage_options(cfg, a):
    import pandas as pd
    from sachai_audio import evaluate as E
    pd.set_option("display.width", 220)
    print(E.policy_options(cfg).to_string(index=False))
    print(f"saved {cfg.report_dir / 'policy_options_dev.csv'} (dev speakers only; test not read)")


def stage_test(cfg, a):
    import pandas as pd
    from sachai_audio import evaluate as E, plots as P
    out = E.evaluate_test(cfg, load_manifest(cfg))
    r = cfg.report_dir
    print(pd.read_csv(r / "table_main.csv").round(4).to_string())
    print(pd.read_csv(r / "test_per_attack.csv").round(4).to_string())
    print(pd.read_csv(r / "test_per_speaker.csv").round(4).to_string())
    print("95% CI:", out["test"]["ci95"])
    pol = E.load_policy(cfg.export_dir)
    P.score_hist(r / "test_scores.csv", pol, r)
    P.roc_plot(r / "test_scores.csv", r)
    P.reliability(r / "test_scores.csv", r)


def stage_robustness(cfg, a):
    from sachai_audio import evaluate as E, plots as P
    print(E.robustness(cfg, load_manifest(cfg)).round(4).to_string())
    P.robustness_plot(cfg.report_dir / "table_robustness.csv", cfg.report_dir)


def stage_analysis(cfg, a):
    from sachai_audio import evaluate as E, plots as P
    P.layer_weights_plot(cfg.export_dir, cfg.report_dir)
    print(E.descriptor_analysis(cfg, load_manifest(cfg)).round(3).to_string())


def stage_demo(cfg, a):
    from sachai_audio.explain import explain
    from sachai_audio.infer import Detector
    m = load_manifest(cfg)
    det = Detector(cfg.export_dir)
    n_spk = len(json.loads((cfg.root / "data" / "splits.json").read_text())["train"])
    for label in (1, 0):
        row = m[(m.split == "test") & (m.label == label) & m.keep].iloc[0]
        res = det.analyze_file(cfg.raw_dir / row.path)
        ex = explain(res, n_spk)
        print(f"\n{row.path} (truth: {'synthetic' if label else 'genuine'}) -> {res['label_en']} "
              f"score {res['spoof_score']} | OOD {res['ood']['flag']} | explanation: {ex['source']}")
        for b in ex["bullets"]:
            print("  -", b)
    t = time.time()
    for _ in range(5):
        det.analyze_file(cfg.raw_dir / row.path, with_descriptors=False)
    print(f"\nlatency: {(time.time() - t) / 5 * 1000:.0f} ms per clip ({row.sec:.1f} s of audio) on {det.device}")


def stage_push(cfg, a):
    from sachai_audio.hub import push_to_hub
    print(push_to_hub(cfg.export_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--run-name", default="xlsr300m-sls-v1")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--force", action="store_true", help="redo a finished data/finalize stage")
    ap.add_argument("--force-retrain", action="store_true", help="start training from scratch (overwrites the run)")
    a = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    cfg = build_cfg(a)
    order = ["data", "audit", "train", "fit", "finalize", "test", "robustness", "analysis", "demo"] if a.stage == "all" else [a.stage]
    for s in order:
        print(f"\n===== {s} =====", flush=True)
        globals()[f"stage_{s}"](cfg, a)


if __name__ == "__main__":   # required on Windows: DataLoader workers re-import this file
    main()
