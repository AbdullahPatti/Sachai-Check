"""Deployable inference (what the FastAPI backend imports) + stage 6: end-to-end evaluation of the full system.

    from sachai.infer import Predictor, verify
    pred = Predictor("outputs/runs/<run>/experiments/main_s13/model")   # load once at server start
    verify("<Urdu or Roman-Urdu claim>", pred)                          # per request
"""
import json
import logging

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config as C
from . import judge, ood
from .data import lead, mask_debunk
from .evaluate import dump, metrics
from .train import DEV, p_fake, predict

log = logging.getLogger("sachai")


class Predictor:
    def __init__(self, model_dir):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEV).eval()
        s = np.load(f"{model_dir}/ood_stats.npz")
        self.stats = {"means": s["means"], "precision": s["precision"]}
        self.thr, self.T = float(s["threshold"]), float(s["temperature"])
        self.cfg = json.loads(open(f"{model_dir}/sachai.json").read())

    def _view(self, t):
        t = mask_debunk(t) if self.cfg["debunk"] == "drop" else t
        return lead(t, self.cfg["lead_words"]) if self.cfg["view"] == "lead" else t

    def __call__(self, texts: list) -> list:
        lg, em = predict(self.model, self.tok, [self._view(t) for t in texts])
        maha = ood.mahalanobis(em, self.stats)
        return [{"p_fake": round(float(p), 4), "ood_score": round(float(m), 2), "ood_flag": bool(m > self.thr)}
                for p, m in zip(p_fake(lg, self.T), maha)]


def verify(text: str, pred: Predictor, use_judge: bool = True) -> dict:
    gate = ood.input_gate(text)
    if gate["status"] == "roman_urdu":
        if judge.available():
            gate["text"] = judge.transliterate(gate["text"])
        else:
            gate["status"] = "roman_urdu_no_transliterator"
    if gate["status"] not in ("ok", "roman_urdu"):
        return {"input": text, "gate": gate, "final": judge.fuse(gate, {}, None)}
    clf = pred([gate["text"]])[0]
    j = judge.judge_claim(gate["text"]) if use_judge and judge.available() else None
    return {"input": text, "analysed_text": gate["text"], "gate": gate, "classifier": clf, "judge": j, "final": judge.fuse(gate, clf, j)}


# ------------------------------------------------------------------ stage 6
def _binary(v):
    return {"likely_fake": 1, "likely_real": 0}.get(v, np.nan)


def evaluate_system(run_dir, df: pd.DataFrame):
    out = run_dir / "judge"
    model_dir = run_dir / "experiments" / sorted(C.KEEP_CHECKPOINT)[0] / "model"
    if not judge.available():
        log.warning("GEMINI_API_KEY not set -> judge stage skipped (set it and re-run with SACHAI_RUN=%s)", run_dir.name)
        return
    if not model_dir.exists():
        log.warning("no deployment checkpoint at %s -> judge stage skipped", model_dir)
        return
    pred = Predictor(model_dir)

    # evaluation items: every manual claim + a stratified sample of the pooled test split
    items = []
    manual = C.MANUAL / "claims.csv"
    if manual.exists():
        m = pd.read_csv(manual).dropna(subset=["text", "label"])
        items += [dict(origin="manual", text=r.text, label=int(str(r.label).strip().lower() in ("fake", "1")),
                       scope_gold=getattr(r, "scope", None), source="manual", domain=getattr(r, "domain", "unknown")) for r in m.itertuples()]
    test = df[(df.split == "test") & ~(df.debunk if C.DEBUNK == "drop" else False)]
    per = max(1, C.JUDGE_TEST_SAMPLE // (2 * test.source.nunique()))
    samp = test.groupby(["source", "label"], group_keys=False).apply(lambda g: g.sample(min(per, len(g)), random_state=0))
    items += [dict(origin="test", text=r.text, label=r.label, scope_gold=None, source=r.source, domain=r.domain) for r in samp.itertuples()]
    log.info("judge stage: %d items (%d manual)", len(items), sum(i["origin"] == "manual" for i in items))

    rows = []
    for i, it in enumerate(items):
        r = verify(it["text"], pred)
        j = r.get("judge") or {}
        meta = judge.rate_explanation(r.get("analysed_text", it["text"]), j) if j else {}
        rows.append(it | {"clf_p_fake": r.get("classifier", {}).get("p_fake", np.nan), "ood_flag": r.get("classifier", {}).get("ood_flag"),
                          "gate": r["gate"]["status"], "judge_verdict": j.get("verdict"), "judge_conf": j.get("confidence"),
                          "judge_scope": j.get("scope"), "n_citations": len(j.get("grounding_citations", [])),
                          "final": r["final"]["verdict"], "final_basis": r["final"]["basis"], "caveats": " | ".join(r["final"]["caveats"]),
                          **{f"meta_{k}": v for k, v in meta.items() if k != "comment"}, "raw": json.dumps(r, ensure_ascii=False)})
        if i % 10 == 0:
            log.info("  judged %d/%d", i + 1, len(items))
    res = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "system_predictions.csv", index=False)

    rep = {}
    for origin, g in [("all", res)] + list(res.groupby("origin")):
        y = g.label.values
        jb = g.judge_verdict.map({"REFUTED": 1, "SUPPORTED": 0})
        dec = jb.notna().values
        fb = g.final.map(_binary)
        rep[origin] = {
            "n": len(g),
            "classifier_only": metrics(y[g.clf_p_fake.notna()], g.clf_p_fake.dropna()),
            "judge_decisive": metrics(y[dec], jb[dec]) | {"coverage": round(float(dec.mean()), 3)} if dec.any() else {},
            "judge_verdict_dist": g.judge_verdict.value_counts().to_dict(),
            "system_final": metrics(y[fb.notna()], fb.dropna()) | {"coverage": round(float(fb.notna().mean()), 3)} if fb.notna().any() else {},
            "final_verdict_dist": g.final.value_counts().to_dict(),
            "classifier_vs_judge_agreement": round(float(((g.clf_p_fake >= 0.5).astype(int)[dec] == jb[dec]).mean()), 3) if dec.any() else None,
            "judge_scope_dist": g.judge_scope.value_counts().to_dict(),
            "meta_judge_mean": g.filter(like="meta_").apply(pd.to_numeric, errors="coerce").mean().round(2).to_dict(),
        }
    if res.scope_gold.notna().any():
        g = res[res.scope_gold.notna()]
        rep["scope_accuracy_on_manual"] = round(float((g.scope_gold == g.judge_scope).mean()), 3)

    # Roman-Urdu round trip: Urdu test item -> Gemini romanises -> gate must detect -> transliterate back -> classifier
    rt = samp.sample(min(100, len(samp)), random_state=1)
    roman = [judge.romanize(t) for t in rt.text]
    gates = [ood.input_gate(r)["status"] for r in roman]
    back = [judge.transliterate(r) for r in roman]
    p_orig, p_back = [x["p_fake"] for x in pred(rt.text.tolist())], [x["p_fake"] for x in pred(back)]
    rep["roman_roundtrip"] = {"n": len(rt), "gate_detected_roman": round(float(np.mean([g == "roman_urdu" for g in gates])), 3),
                              "classifier_on_original": metrics(rt.label, p_orig), "classifier_after_roundtrip": metrics(rt.label, p_back),
                              "prediction_flip_rate": round(float(np.mean((np.array(p_orig) >= .5) != (np.array(p_back) >= .5))), 3),
                              "examples": [{"urdu": u[:200], "roman": r[:200], "back": b[:200]} for u, r, b in zip(rt.text[:5], roman[:5], back[:5])]}
    dump(rep, out / "judge_metrics.json")
    log.info("judge stage done: %s", {k: v.get("system_final", {}).get("macro_f1") for k, v in rep.items() if isinstance(v, dict) and "n" in v})
