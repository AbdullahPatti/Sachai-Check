"""Single source of truth for every setting in the Sachai Check audio pipeline.

Every value that changes model behaviour is saved next to the model, so a run
can always be reproduced and the API always uses the settings the model was
trained and calibrated with.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PREPROCESS_VERSION = "v1-16k-mono-trim40"
POLICY_VERSION = "v1-calibrated-band"


@dataclass
class Config:
    # where everything lives
    project_dir: str = field(default_factory=lambda: os.environ.get("SACHAI_HOME", "."))
    run_name: str = "xlsr300m-sls-v1"

    # data
    dataset_repo: str = "CSALT/deepfake_detection_dataset_urdu"
    local_dataset_dir: str = ""          # set this to use a local copy (same folder layout)
    sample_rate: int = 16000
    trim_top_db: float = 40.0            # leading/trailing silence trim, same for both classes
    min_clip_sec: float = 0.5            # clips shorter than this after trimming are excluded
    n_dev_speakers: int = 2
    n_test_speakers: int = 3
    split_seed: int = 1234

    # model
    backbone: str = "facebook/wav2vec2-xls-r-300m"
    proj_dim: int = 256
    dropout: float = 0.2
    mask_time_prob: float = 0.05         # SpecAugment-style time masking inside XLS-R
    grad_checkpointing: bool = True

    # training
    seed: int = 42
    crop_secs: tuple = (2.0, 3.0, 4.0)   # a crop length is drawn per batch
    batch_size: int = 16                 # RTX A4000 (16 GB). On an 8 GB GPU use batch_size=8, grad_accum=4
    grad_accum: int = 2                  # effective batch = 32 either way
    max_epochs: int = 20
    patience: int = 5                    # early stopping on dev EER
    lr_backbone: float = 1e-5
    lr_head: float = 5e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    label_smoothing: float = 0.05
    aug_prob: float = 0.75               # chance a training crop gets channel augmentation
    num_workers: int = 4                 # set 0 if DataLoader workers misbehave on Windows
    ckpt_minutes: float = 8.0            # mid-epoch checkpoint interval
    train_probe_clips: int = 600         # clean train subset scored each epoch (fit diagnosis)

    # inference / product
    verdict_win_sec: float = 4.0
    verdict_hop_sec: float = 2.0
    segment_win_sec: float = 2.0
    segment_hop_sec: float = 1.0
    infer_batch: int = 16
    target_precision: float = 0.99       # sets the inconclusive band on dev
    min_band: float = 0.02               # band never narrower than +-0.02 around the EER threshold (chosen on dev, before test; +-0.10 gave no genuine verdicts)
    api_min_sec: float = 3.0
    api_max_sec: float = 60.0
    api_max_mb: float = 20.0

    # ---------- derived paths ----------
    @property
    def root(self) -> Path:
        return Path(self.project_dir).resolve()

    @property
    def raw_dir(self) -> Path:
        return Path(self.local_dataset_dir) if self.local_dataset_dir else self.root / "data" / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.root / "data" / "cache_16k"

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.run_name

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def export_dir(self) -> Path:
        return self.run_dir / "export"

    @property
    def report_dir(self) -> Path:
        return self.run_dir / "report"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["crop_secs"] = list(self.crop_secs)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Config":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["crop_secs"] = tuple(d.get("crop_secs", (2.0, 3.0, 4.0)))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
