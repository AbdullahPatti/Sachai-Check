"""Stage 4: transformer fine-tuning, calibration, per-slice evaluation and OOD scoring for every experiment."""
import gc
import json
import logging
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from . import config as C
from . import ood
from .data import length_match, protocol_frames, view
from .evaluate import dump, metrics, sliced

log = logging.getLogger("sachai")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16 if DEV.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _batches(tok, texts, bs):
    """Length-sorted batches for inference (much faster on short headlines); returns order to undo sorting."""
    order = np.argsort([len(t) for t in texts])
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        yield idx, tok([texts[j] for j in idx], truncation=True, max_length=C.MAX_LEN, padding=True, return_tensors="pt")


@torch.no_grad()
def predict(model, tok, texts, bs=None):
    """-> logits [n,2], CLS embeddings [n,h] (float32 numpy)."""
    model.eval()
    bs = bs or C.BATCH * 4
    logits, embs = np.zeros((len(texts), 2), np.float32), None
    for idx, enc in _batches(tok, list(texts), bs):
        with torch.autocast(DEV.type, dtype=AMP, enabled=DEV.type == "cuda"):
            out = model(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0].float().cpu().numpy()
        if embs is None:
            embs = np.zeros((len(texts), cls.shape[1]), np.float32)
        logits[idx], embs[idx] = out.logits.float().cpu().numpy(), cls
    return logits, embs


def fit_temperature(logits, y) -> float:
    """Guo et al. (2017) temperature scaling on validation logits."""
    lg, yt = torch.tensor(logits), torch.tensor(y)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def p_fake(logits, T=1.0):
    return torch.softmax(torch.tensor(logits) / T, 1)[:, 1].numpy()


def _atomic_save(obj, path):
    """Write to .tmp then rename: a power cut mid-write leaves the previous checkpoint intact."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_model(exp, train, val, texts_train, texts_val, out):
    """Resumable: every C.SAVE_EVERY_MIN minutes and at each epoch end the full training state (weights, optimizer,
    scheduler, epoch, step, best-so-far) goes to out/resume.pt. After a power cut, F5 continues from that exact step."""
    name = C.MODELS[exp["model"]]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2, ignore_mismatched_sizes=True).to(DEV)

    enc = tok(list(texts_train), truncation=True, max_length=C.MAX_LEN)
    items = [{"input_ids": i, "attention_mask": a, "labels": int(y)} for i, a, y in zip(enc["input_ids"], enc["attention_mask"], train.label)]
    # Source x label balanced sampling (temperature-smoothed) so the 67k-row HnB can't drown the 1.5k-row urdufake.
    grp = train["source"] + "_" + train["label"].astype(str)
    w = torch.tensor(((1.0 / grp.map(grp.value_counts())) ** C.SAMPLER_TEMPERATURE).values, dtype=torch.double)
    collate = lambda b: tok.pad(b, return_tensors="pt")

    def loader(epoch):  # same seed per epoch -> identical batch order after a restart
        g = torch.Generator().manual_seed(exp["seed"] * 1000 + epoch)
        return DataLoader(items, batch_size=C.BATCH, sampler=WeightedRandomSampler(w, len(items), replacement=True, generator=g),
                          collate_fn=collate, num_workers=0)

    lr = C.LR.get(exp["model"], C.DEFAULT_LR)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": C.WEIGHT_DECAY},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=lr)
    n_steps = math.ceil(len(items) / C.BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(C.WARMUP * n_steps * C.EPOCHS), n_steps * C.EPOCHS)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda" and AMP == torch.float16)

    best, bad, history, start_ep, start_step = -1, 0, [], 0, 0
    resume, best_path = out / "resume.pt", out / "best.pt"
    if resume.exists():
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(st["rng"])
        best, bad, history, start_ep, start_step = st["best"], st["bad"], st["history"], st["epoch"], st["step"]
        log.info("  RESUMING from epoch %d step %d (best val macroF1 so far %.4f)", start_ep, start_step, best)

    def save_state(epoch, step):
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                      "rng": torch.get_rng_state(), "best": best, "bad": bad, "history": history, "epoch": epoch, "step": step}, resume)

    last_save = time.time()
    for ep in range(start_ep, C.EPOCHS):
        if bad >= C.PATIENCE:
            break
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for i, b in enumerate(loader(ep)):
            if ep == start_ep and i < start_step:
                continue  # already trained before the interruption
            b = {k: v.to(DEV) for k, v in b.items()}
            with torch.autocast(DEV.type, dtype=AMP, enabled=DEV.type == "cuda"):
                loss = model(**b).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tot, seen = tot + loss.item(), seen + 1
            if math.isnan(loss.item()):
                raise RuntimeError("loss is NaN - lower LR or switch AMP off")
            if i % 200 == 0:
                log.info("  ep%d step %d/%d loss %.4f", ep, i, n_steps, loss.item())
            if time.time() - last_save > C.SAVE_EVERY_MIN * 60:
                save_state(ep, i + 1)
                last_save = time.time()
        vl, _ = predict(model, tok, texts_val)
        vm = metrics(val.label, p_fake(vl))
        history.append({"epoch": ep, "train_loss": tot / max(seen, 1), "val_macro_f1": vm["macro_f1"], "val_auroc": vm["auroc"], "secs": round(time.time() - t0)})
        log.info("  epoch %d  loss %.4f  val macroF1 %.4f  auroc %.4f  (%ds)", ep, tot / max(seen, 1), vm["macro_f1"], vm["auroc"], time.time() - t0)
        if vm["macro_f1"] > best:
            best, bad = vm["macro_f1"], 0
            _atomic_save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= C.PATIENCE:
                log.info("  early stop")
        save_state(ep + 1, 0)
        last_save = time.time()
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    return model, tok, history


def run_experiment(exp: dict, df: pd.DataFrame, out_root, ood_sets: dict):
    out = out_root / exp["name"]
    if (out / "metrics.json").exists():
        log.info("skip %s (done)", exp["name"])
        return
    out.mkdir(parents=True, exist_ok=True)
    seed_all(exp["seed"])
    debunk, tv = exp.get("debunk", C.DEBUNK), exp.get("view", C.TEXT_VIEW)
    mask = debunk == "drop"
    train, val, tests = protocol_frames(df, exp["protocol"], debunk)
    if exp.get("length_match", C.LENGTH_MATCH_TRAIN):
        train = length_match(train, exp["seed"])
    if C.SMOKE:
        train, val = train.sample(min(400, len(train)), random_state=0), val.sample(min(200, len(val)), random_state=0)
        tests = {k: v.sample(min(200, len(v)), random_state=0) for k, v in tests.items()}
        ood_sets = {k: v[:100] for k, v in ood_sets.items()}
    log.info("== %s  model=%s protocol=%s debunk=%s view=%s  train=%d val=%d tests=%s", exp["name"], exp["model"], exp["protocol"],
             debunk, tv, len(train), len(val), {k: len(v) for k, v in tests.items()})

    t0 = time.time()
    model, tok, history = train_model(exp, train, val, view(train, mask, tv), view(val, mask, tv).tolist(), out)

    val_logits, val_emb = predict(model, tok, view(val, mask, tv).tolist())
    T = fit_temperature(val_logits, val.label.values)
    fit_rows = train.sample(min(len(train), 20000), random_state=0)
    tr_logits, tr_emb = predict(model, tok, view(fit_rows, mask, tv).tolist())
    stats = ood.fit(tr_emb, fit_rows.label.values)
    val_sc = ood.scores(val_emb, val_logits, stats)
    thr = ood.threshold(val_sc["mahalanobis"])

    res = {"experiment": exp, "n_train": len(train), "n_val": len(val), "history": history, "temperature": T,
           "val": metrics(val.label, p_fake(val_logits, T)), "tests": {}, "ood": {}}
    ood_scores = {}
    for tname, te in tests.items():
        lg, em = predict(model, tok, view(te, mask, tv).tolist())
        p = p_fake(lg, T)
        res["tests"][tname] = sliced(te, p) | {"uncalibrated_ece": metrics(te.label, p_fake(lg))["ece"]}
        sc = ood.scores(em, lg, stats)
        te.assign(p_fake=p, ood_mahalanobis=sc["mahalanobis"], ood_flag=sc["mahalanobis"] > thr)[
            ["source", "domain", "label", "p_fake", "ood_mahalanobis", "ood_flag", "text"]].to_csv(out / f"preds_{tname}.csv", index_label="row")
        if tname.startswith("heldout_"):
            ood_scores["near_" + tname] = sc  # a whole unseen dataset = near-OOD
        else:  # selective prediction: accuracy on what the gate lets through
            keep = sc["mahalanobis"] <= thr
            res["tests"][tname]["in_gate_only"] = metrics(te.label[keep], p[keep]) | {"coverage": round(float(keep.mean()), 4)}
    for oname, texts in ood_sets.items():
        lg, em = predict(model, tok, texts)
        ood_scores[oname] = ood.scores(em, lg, stats)
    res["ood"] = ood.evaluate(val_sc, ood_scores, thr)
    res["minutes"] = round((time.time() - t0) / 60, 1)

    if exp["name"] in C.KEEP_CHECKPOINT:
        md = out / "model"
        model.save_pretrained(md)
        tok.save_pretrained(md)
        np.savez(md / "ood_stats.npz", means=stats["means"], precision=stats["precision"], threshold=thr, temperature=T)
        (md / "sachai.json").write_text(json.dumps({"debunk": debunk, "view": tv, "lead_words": C.LEAD_WORDS, "max_len": C.MAX_LEN,
                                                    "labels": {"0": "real", "1": "fake"}}, indent=1))
    dump(res, out / "metrics.json")  # written last: its presence marks the experiment as finished
    for f in ("resume.pt", "best.pt"):  # training state no longer needed (xlmr-large: ~9 GB)
        (out / f).unlink(missing_ok=True)
    log.info("== %s done in %.1f min: %s", exp["name"], res["minutes"], {k: v["all"]["macro_f1"] for k, v in res["tests"].items()})
    del model
    gc.collect()
    torch.cuda.empty_cache()
