"""Push the exported detector to the Hugging Face Hub. Token comes from HF_TOKEN in .env."""
from __future__ import annotations

import json
import os
from pathlib import Path

CARD = """---
license: cc-by-nc-4.0
language:
- ur
tags:
- audio-classification
- anti-spoofing
- deepfake-detection
- urdu
- wav2vec2
- xls-r
datasets:
- CSALT/deepfake_detection_dataset_urdu
base_model: {backbone}
pipeline_tag: audio-classification
---

# Sachai Check: Urdu audio deepfake detector

Part of **Sachai Check** (FAST-NUCES Lahore final year project). It decides whether an Urdu speech clip is
genuine (bona fide) or synthetic (text-to-speech / voice clone). It reports authenticity only; it says nothing
about whether the spoken content is true.

## Model
- Backbone: `{backbone}` fine-tuned end to end (CNN feature encoder frozen).
- Head: softmax-weighted sum of all {n_states} hidden layers, attentive statistics pooling, 2-layer MLP.
- Input: 16 kHz mono, leading and trailing silence trimmed (40 dB), per-window zero-mean unit-variance.
- Inference: 4 s windows with 2 s hop, mean log-odds per clip, temperature-scaled (T = {temperature:.3f}).
- Decision policy (fitted on dev speakers only): synthetic if score >= {high:.3f}, genuine if score <= {low:.3f},
  otherwise inconclusive.

## Data and protocol
CSALT Urdu deepfake dataset (Munir et al., Findings of ACL 2024): genuine speech plus Tacotron and VITS clones.
Split by speaker, so test voices are never heard in training: train {n_train} clips, dev {n_dev}, test {n_test}.

## Results (test speakers, frozen dev policy)
| metric | value |
|---|---|
| EER | {eer:.4f} |
| ROC-AUC | {auc:.4f} |
| Accuracy at EER threshold | {acc:.4f} |
| Macro-F1 | {f1:.4f} |
| Coverage (not inconclusive) | {cov:.4f} |

Robustness and per-attack tables are in `metrics.json`.

## Limits
Trained on two TTS families and a small set of speakers. Commercial voice cloning, other languages, music and
heavily processed audio are outside the tested conditions; the API flags such inputs as out of distribution.
Treat every verdict as a signal, not proof.

## Use
```python
from sachai_audio.infer import Detector
det = Detector("{repo_id}")
print(det.analyze_file("clip.wav")["verdict"])
```
License follows the dataset (CC BY-NC 4.0): non-commercial use only.
"""


def default_repo_id(name: str = "sachai-check-urdu-audio-deepfake") -> str:
    from huggingface_hub import whoami
    return f"{whoami(token=os.environ['HF_TOKEN'])['name']}/{name}"


def write_card(export_dir: Path, repo_id: str) -> None:
    exp = Path(export_dir)
    m = json.loads((exp / "metrics.json").read_text())
    pol = json.loads((exp / "policy.json").read_text())
    cfg = json.loads((exp / "sachai_config.json").read_text())
    ssl = json.loads((exp / "ssl_config.json").read_text())
    t = m.get("test", {})
    card = CARD.format(backbone=cfg["backbone"], n_states=ssl["num_hidden_layers"] + 1, temperature=pol["temperature"],
                       high=pol["high"], low=pol["low"], n_train=m["n_clips"]["train"], n_dev=m["n_clips"]["dev"],
                       n_test=m["n_clips"]["test"], eer=t.get("eer", float("nan")), auc=t.get("auc", float("nan")),
                       acc=t.get("accuracy", float("nan")), f1=t.get("macro_f1", float("nan")),
                       cov=t.get("coverage", float("nan")), repo_id=repo_id)
    (exp / "README.md").write_text(card, encoding="utf-8")


def push_to_hub(export_dir: Path, repo_id: str | None = None, private: bool | None = None) -> str:
    from huggingface_hub import create_repo, upload_folder

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in your .env file first.")
    repo_id = repo_id or os.environ.get("HF_REPO_ID") or default_repo_id()
    if private is None:
        private = os.environ.get("HF_PRIVATE", "true").lower() != "false"
    write_card(export_dir, repo_id)
    create_repo(repo_id, token=token, private=private, exist_ok=True, repo_type="model")
    upload_folder(repo_id=repo_id, folder_path=str(export_dir), token=token, repo_type="model",
                  commit_message="Sachai Check audio detector", ignore_patterns=["*.tmp"])
    return f"https://huggingface.co/{repo_id}"
