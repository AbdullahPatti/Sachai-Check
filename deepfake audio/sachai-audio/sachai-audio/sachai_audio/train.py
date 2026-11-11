"""Resumable fine-tuning.

Power-cut safety:
  * A full checkpoint (model, optimizer, scheduler, scaler, sampler position, history, early-stopping state)
    is written atomically at the end of every epoch and every `ckpt_minutes` inside an epoch.
  * Writes go to a temp file, are fsynced, then renamed. The previous checkpoint is kept as a fallback,
    so a crash during a write can never destroy the last good state.
  * Batches are a pure function of (seed, epoch, batch index), so resuming continues from the exact batch.
  * Ctrl+C (KeyboardInterrupt) also saves before stopping.
Re-running `train(cfg)` resumes automatically. If training already finished it returns immediately.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import ResumableBatchSampler, TrainDataset, WindowDataset
from .metrics import auc, eer
from .model import SachaiDetector, soft_ce


# ---------------------------------------------------------------- helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_settings(device: torch.device) -> tuple[torch.dtype | None, bool]:
    """bf16 on GPUs that support it (RTX 30/40/50, A4000), fp16 + GradScaler otherwise (e.g. Colab T4)."""
    if device.type != "cuda":
        return None, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def autocast_ctx(device: torch.device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    prev = path.with_name(path.stem + "_prev" + path.suffix)
    if path.exists():
        os.replace(path, prev)
    os.replace(tmp, path)


def load_latest_checkpoint(ckpt_dir: Path):
    last = ckpt_dir / "last.pt"
    for p in (last, last.with_suffix(".pt.tmp"), ckpt_dir / "last_prev.pt"):
        if p.exists():
            try:
                return torch.load(p, map_location="cpu", weights_only=False), p
            except Exception as e:  # truncated by a power cut
                print(f"[resume] {p.name} unreadable ({type(e).__name__}), trying older checkpoint")
    return None, None


def build_optimizer(model: SachaiDetector, cfg: Config) -> torch.optim.Optimizer:
    groups = {"bb_decay": [], "bb_no": [], "head_decay": [], "head_no": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        side = "bb" if name.startswith("ssl.") else "head"
        no_decay = p.ndim < 2 or "norm" in name.lower() or name == "layer_logits"
        groups[f"{side}_{'no' if no_decay else 'decay'}"].append(p)
    params = [
        {"params": groups["bb_decay"], "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
        {"params": groups["bb_no"], "lr": cfg.lr_backbone, "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
        {"params": groups["head_no"], "lr": cfg.lr_head, "weight_decay": 0.0},
    ]
    kw = {"fused": True} if torch.cuda.is_available() else {}
    return torch.optim.AdamW([g for g in params if g["params"]], betas=(0.9, 0.98), eps=1e-6, **kw)


def build_scheduler(opt, total_steps: int, warmup_frac: float):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def score_clips(model: SachaiDetector, clips: list, cfg: Config, device: torch.device, win_sec: float | None = None,
                hop_sec: float | None = None, transform=None, return_embeddings: bool = False, num_workers: int = 0,
                progress: bool = False) -> dict:
    """Window-level scoring aggregated per clip (mean log-odds), identical to the product's procedure."""
    sr = cfg.sample_rate
    win = int((win_sec or cfg.verdict_win_sec) * sr)
    hop = int((hop_sec or cfg.verdict_hop_sec) * sr)
    ds = WindowDataset(clips, win, hop, transform=transform)
    dl = DataLoader(ds, batch_size=cfg.infer_batch, shuffle=False, num_workers=num_workers)
    dtype, _ = amp_settings(device)
    was_training = model.training
    model.eval()
    z_sum = np.zeros(len(clips))
    n_win = np.zeros(len(clips))
    emb_sum = None
    it = dl
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(dl, desc="scoring", leave=False)
    for x, ci, _ in it:
        with autocast_ctx(device, dtype):
            out = model(x.to(device, non_blocking=True))
        z = SachaiDetector.spoof_logit(out["logits"]).float().cpu().numpy()
        ci = ci.numpy()
        np.add.at(z_sum, ci, z)
        np.add.at(n_win, ci, 1)
        if return_embeddings:
            e = out["embedding"].float().cpu().numpy()
            if emb_sum is None:
                emb_sum = np.zeros((len(clips), e.shape[1]))
            np.add.at(emb_sum, ci, e)
    model.train(was_training)
    res = {"z": z_sum / np.maximum(n_win, 1), "n_windows": n_win}
    if return_embeddings:
        res["embedding"] = emb_sum / np.maximum(n_win, 1)[:, None]
    return res


def _clip_metrics(y, z) -> dict:
    y = np.asarray(y)
    p = 1 / (1 + np.exp(-z))
    loss = float(-np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))))
    return {"eer": eer(y, z)[0], "auc": auc(y, z), "loss": loss, "acc@0": float(((z > 0).astype(int) == y).mean())}


# ---------------------------------------------------------------- training
LOCKED_ON_RESUME = ("backbone", "proj_dim", "dropout", "seed", "crop_secs", "batch_size", "grad_accum", "max_epochs",
                    "lr_backbone", "lr_head", "weight_decay", "warmup_frac", "label_smoothing", "aug_prob")


def train(cfg: Config, manifest: pd.DataFrame, device: torch.device | None = None, max_steps: int | None = None,
          force_restart: bool = False) -> dict:
    device = device or pick_device()
    ckpt_dir, best_dir = cfg.ckpt_dir, cfg.run_dir / "best"
    done_flag = cfg.run_dir / "train_done.json"
    if done_flag.exists() and not force_restart:
        print(f"Training already finished: {json.loads(done_flag.read_text())}")
        return json.loads(done_flag.read_text())
    state, src = (None, None) if force_restart else load_latest_checkpoint(ckpt_dir)
    if state is not None:
        # A resumed run must keep the settings it started with, or batch order and LR schedule would change.
        import dataclasses
        saved = state.get("config", {})
        changed = {k: (getattr(cfg, k), saved[k]) for k in LOCKED_ON_RESUME
                   if k in saved and list(np.atleast_1d(getattr(cfg, k))) != list(np.atleast_1d(saved[k]))}
        if changed:
            print(f"[resume] keeping the original settings of this run: {changed} (current -> original)")
            cfg = dataclasses.replace(cfg, **{k: (tuple(v[1]) if k == "crop_secs" else v[1]) for k, v in changed.items()})
    cfg.save(cfg.run_dir / "config.json")

    df = manifest[manifest.keep]
    tr, dv = df[df.split == "train"].reset_index(drop=True), df[df.split == "dev"].reset_index(drop=True)
    probe = pd.concat([g.sample(min(len(g), cfg.train_probe_clips // 2), random_state=cfg.seed)
                       for _, g in tr.groupby("label")])

    set_seed(cfg.seed)
    if state is None:
        model = SachaiDetector.from_backbone(cfg.backbone, cfg.proj_dim, cfg.dropout, cfg.mask_time_prob,
                                             cfg.grad_checkpointing)
    else:
        ssl_cfg = state["ssl_config"]
        from transformers import Wav2Vec2Config
        model = SachaiDetector(Wav2Vec2Config.from_dict(ssl_cfg), **state["head_config"])
        model.load_state_dict(state["model"])
        model.prepare_for_training(cfg.grad_checkpointing)
        print(f"[resume] from {src.name}: epoch {state['epoch']} batch {state['batch_in_epoch']} "
              f"(step {state['global_step']})")
    model.to(device)

    crops = [int(s * cfg.sample_rate) for s in cfg.crop_secs]
    sampler = ResumableBatchSampler(len(tr), cfg.batch_size, cfg.grad_accum, cfg.seed, crops)
    steps_per_epoch = sampler.batches_per_epoch() // cfg.grad_accum
    total_steps = steps_per_epoch * cfg.max_epochs
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, total_steps, cfg.warmup_frac)
    amp_dtype, use_scaler = amp_settings(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    st = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0, "best": None, "bad_epochs": 0, "history": []}
    if state is not None:
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        st = {k: state[k] for k in st}
        if "rng" in state:
            torch.set_rng_state(state["rng"]["torch"])
            np.random.set_state(state["rng"]["numpy"])
            random.setstate(state["rng"]["python"])

    def save_ckpt(batch_in_epoch: int, epoch: int):
        payload = {
            "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(), "ssl_config": model.ssl.config.to_dict(), "head_config": model.head_config,
            "epoch": epoch, "batch_in_epoch": batch_in_epoch, "global_step": st["global_step"], "best": st["best"],
            "bad_epochs": st["bad_epochs"], "history": st["history"], "config": cfg.to_dict(),
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
            "saved_at": time.time(),
        }
        atomic_torch_save(payload, ckpt_dir / "last.pt")

    ds = TrainDataset(tr.cache.tolist(), tr.label.tolist(), cfg, train=True)
    from tqdm.auto import tqdm

    last_save = time.time()
    stop = st["bad_epochs"] >= cfg.patience
    for epoch in range(st["epoch"], cfg.max_epochs):
        if stop:
            break
        start_b = st["batch_in_epoch"] if epoch == st["epoch"] else 0
        sampler.set_epoch(epoch, start_b)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                            persistent_workers=False)
        model.train()
        run_loss, run_n, boundary = 0.0, 0, start_b
        bar = tqdm(loader, total=len(sampler), desc=f"epoch {epoch + 1}/{cfg.max_epochs}", leave=False)
        try:
            for b_off, (x, y) in enumerate(bar):
                b = start_b + b_off
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast_ctx(device, amp_dtype):
                    out = model(x)
                loss = soft_ce(out["logits"], y, cfg.label_smoothing)
                scaler.scale(loss / cfg.grad_accum).backward()
                run_loss += loss.item() * len(y)
                run_n += len(y)
                if (b + 1) % cfg.grad_accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    st["global_step"] += 1
                    boundary = b + 1
                    bar.set_postfix(loss=f"{run_loss / max(run_n, 1):.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
                    if (time.time() - last_save) / 60 > cfg.ckpt_minutes:
                        save_ckpt(boundary, epoch)
                        last_save = time.time()
                    if max_steps and st["global_step"] >= max_steps:
                        stop = True
                        break
        except KeyboardInterrupt:
            save_ckpt(boundary, epoch)
            print(f"\nInterrupted. Saved at epoch {epoch + 1}, batch {boundary}. Run train() again to resume.")
            raise

        if stop and boundary < sampler.batches_per_epoch():
            save_ckpt(boundary, epoch)
            print(f"Stopped early by max_steps at epoch {epoch + 1}, batch {boundary}.")
            return {"stopped": True, "global_step": st["global_step"]}

        # ---- end of epoch: dev + clean train probe
        dev_s = score_clips(model, dv.cache.tolist(), cfg, device)
        prb_s = score_clips(model, probe.cache.tolist(), cfg, device)
        dm, pm = _clip_metrics(dv.label.values, dev_s["z"]), _clip_metrics(probe.label.values, prb_s["z"])
        row = {"epoch": epoch + 1, "step": st["global_step"], "train_loss_aug": run_loss / max(run_n, 1),
               "train_probe_loss": pm["loss"], "train_probe_eer": pm["eer"],
               "dev_loss": dm["loss"], "dev_eer": dm["eer"], "dev_auc": dm["auc"], "lr": sched.get_last_lr()[0],
               "layer_weights": model.layer_weights().detach().cpu().tolist()}
        st["history"].append(row)
        key = (round(dm["eer"], 4), dm["loss"])
        improved = st["best"] is None or key < tuple(st["best"])
        if improved:
            st["best"], st["bad_epochs"] = list(key), 0
            model.save_pretrained(best_dir)
            (best_dir / "best_epoch.json").write_text(json.dumps(row, indent=2))
        else:
            st["bad_epochs"] += 1
        print(f"epoch {epoch + 1}: train(aug) loss {row['train_loss_aug']:.4f} | train-probe EER {pm['eer']:.4f} "
              f"loss {pm['loss']:.4f} | dev EER {dm['eer']:.4f} AUC {dm['auc']:.4f} loss {dm['loss']:.4f}"
              f"{'  *best*' if improved else ''}")
        pd.DataFrame([{k: v for k, v in r.items() if k != "layer_weights"} for r in st["history"]]).to_csv(
            cfg.run_dir / "history.csv", index=False)
        st["epoch"], st["batch_in_epoch"] = epoch + 1, 0
        save_ckpt(0, epoch + 1)
        last_save = time.time()
        if st["bad_epochs"] >= cfg.patience:
            print(f"Early stopping: dev did not improve for {cfg.patience} epochs.")
            break
        if stop:
            return {"stopped": True, "global_step": st["global_step"]}

    if st["best"] is None:
        raise RuntimeError("No completed epoch yet.")
    summary = {"epochs_run": st["epoch"], "global_step": st["global_step"], "best_dev_eer": st["best"][0],
               "best_dev_loss": st["best"][1], "best_dir": str(best_dir)}
    done_flag.write_text(json.dumps(summary, indent=2))
    return summary
