"""Grounded explanation: facts -> LLM justifier -> rule checks -> LLM judge -> fallback template.

The detector decides. The LLM only turns validated measurements (numbered facts E1..En) into plain
language. Every bullet must cite the facts it uses; a rule check rejects numbers that do not appear in
the facts; a second model (the judge) rejects bullets that are unfaithful, claim to know *why* the
network decided, or contradict the verdict. If the LLM is unavailable or fewer than two bullets survive,
a deterministic template built from the same facts is used. The explanation can never change the verdict.

Environment variables (any OpenAI-compatible endpoint works, e.g. Groq, OpenRouter, Ollama, Gemini):
    LLM_API_KEY    key for the provider
    LLM_BASE_URL   default https://api.groq.com/openai/v1
    LLM_MODEL      justifier model, default llama-3.3-70b-versatile
    JUDGE_MODEL    judge model, default openai/gpt-oss-120b (a different model limits self-preference bias)
"""
from __future__ import annotations

import json
import os
import re

SUMMARY_UR = {
    "synthetic": "ڈیٹیکٹر کے مطابق یہ آواز غالباً مصنوعی (کمپیوٹر سے بنائی گئی) ہے۔ یہ اشارہ ہے، حتمی ثبوت نہیں۔",
    "bona_fide": "ڈیٹیکٹر کے مطابق یہ آواز غالباً اصلی انسانی آواز ہے۔ یہ اشارہ ہے، حتمی ثبوت نہیں۔",
    "inconclusive": "ڈیٹیکٹر واضح فیصلہ نہیں کر سکا۔ بہتر معیار کی ریکارڈنگ کے ساتھ دوبارہ کوشش کریں۔",
}
VERDICT_WORD = {"synthetic": "synthetic", "bona_fide": "genuine", "inconclusive": "inconclusive"}


def fmt_t(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------- facts
def build_facts(res: dict, n_speakers_train: int | None = None) -> list[dict]:
    facts = []
    pol, v = res["policy"], res["verdict"]
    if res.get("spoof_score") is not None:
        facts.append({"id": "E1", "kind": "detector", "text":
                      f"The detector's calibrated spoof score is {res['spoof_score']:.2f} on a 0 to 1 scale. Scores of "
                      f"{pol['high']:.2f} or more are reported as synthetic, {pol['low']:.2f} or less as genuine, and "
                      f"anything between as inconclusive. Verdict: {VERDICT_WORD[v]}."})
    else:
        facts.append({"id": "E1", "kind": "detector", "text": "The clip had too little speech to score. Verdict: inconclusive."})
    segs = res.get("segments") or []
    if segs:
        flagged = [s for s in segs if s["flagged"]]
        top = max(segs, key=lambda s: s["score"])
        low = min(segs, key=lambda s: s["score"])
        txt = (f"{len(flagged)} of {len(segs)} two-second segments score at or above the synthetic cut-off. "
               f"Highest segment score {top['score']:.2f} at {fmt_t(top['start'])} to {fmt_t(top['end'])}; "
               f"lowest {low['score']:.2f} at {fmt_t(low['start'])} to {fmt_t(low['end'])}.")
        facts.append({"id": "E2", "kind": "segments", "text": txt})
    i = 3
    for d in sorted(res.get("descriptors", []), key=lambda r: (not r["unusual_for_genuine"], -r["separation"]))[:4]:
        side = "lower" if d["genuine_pct"] < 50 else "higher"
        share = d["genuine_pct"] if side == "higher" else 100 - d["genuine_pct"]
        name = d["label"][0].lower() + d["label"][1:]
        facts.append({"id": f"E{i}", "kind": "measurement", "text":
                      f"Measured {name}: {d['value']:.3g}. This is {side} than {share:.0f}% of genuine training "
                      f"clips (genuine median {d['genuine_median']:.3g}, synthetic median {d['synthetic_median']:.3g}); "
                      f"the value is closer to the {d['closer_to']} median."})
        i += 1
    q = res.get("quality", {})
    qtxt = f"Speech after trimming silence: {q.get('trimmed_sec', 0):.1f} s"
    if q.get("snr_db") is not None:
        qtxt += f"; rough signal-to-noise estimate {q['snr_db']:.0f} dB"
    qtxt += f"; clipping {q.get('clipping_pct', 0):.2f}% of samples"
    if q.get("codec"):
        qtxt += f"; uploaded as {q['codec']} at {q.get('source_sample_rate')} Hz"
    if q.get("flags"):
        qtxt += f"; quality flags: {', '.join(q['flags'])}"
    facts.append({"id": f"E{i}", "kind": "quality", "text": qtxt + "."})
    i += 1
    ood = res.get("ood") or {}
    if ood.get("flag"):
        facts.append({"id": f"E{i}", "kind": "coverage", "text":
                      f"The recording's internal representation is far from the training data (distance {ood['ratio']:.1f} "
                      f"times the usual limit), so it lies outside the conditions the detector was tested on."})
        i += 1
    spk = f"{n_speakers_train} " if n_speakers_train else ""
    facts.append({"id": f"E{i}", "kind": "scope", "text":
                  f"The detector was trained on genuine Urdu speech and Tacotron and VITS voice clones of {spk}speakers. "
                  f"Voices from other generators, such as commercial voice-cloning services, were not in training."})
    return facts


# ---------------------------------------------------------------- template
def template_bullets(res: dict, facts: list[dict]) -> list[str]:
    v = res["verdict"]
    out = []
    if res.get("spoof_score") is not None:
        word = {"synthetic": "above the synthetic cut-off", "bona_fide": "below the genuine cut-off",
                "inconclusive": "inside the uncertainty band, so no decision is made"}[v]
        out.append(f"The detector scored this clip {res['spoof_score']:.2f}, which is {word}.")
    else:
        out.append("There was not enough speech in the clip to score it reliably.")
    seg = next((f for f in facts if f["kind"] == "segments"), None)
    if seg:
        out.append(seg["text"])
    meas = [f for f in facts if f["kind"] == "measurement"]
    if meas:
        out.append(meas[0]["text"] + " This measurement is context, not the reason for the verdict.")
    cov = next((f for f in facts if f["kind"] in ("coverage",)), None)
    if cov:
        out.append(cov["text"])
    return out[:4]


# ---------------------------------------------------------------- llm plumbing
def _client():
    key = os.environ.get("LLM_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"), timeout=20, max_retries=0)


def _chat_json(client, model: str, system: str, user: str) -> dict:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        r = client.chat.completions.create(model=model, messages=msgs, temperature=0.1, response_format={"type": "json_object"})
    except Exception:
        r = client.chat.completions.create(model=model, messages=msgs, temperature=0.1)
    text = r.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in LLM reply")
    return json.loads(m.group(0))


JUSTIFIER_SYSTEM = """You explain the output of an Urdu audio deepfake detector to a general reader.
Rules:
1. Use ONLY the numbered facts provided. Do not add any measurement, number, cause or claim that is not in the facts.
2. The detector already decided. Never argue for a different verdict and never call the verdict certain.
3. You cannot see inside the neural network. Do not say what the network 'noticed', 'heard' or 'used'. Present
   measurements as context ("a measured property of the clip"), not as the cause of the decision.
4. Write 3 short bullets in plain English, each under 35 words. Each bullet lists the fact ids it relies on.
5. If a coverage or scope fact applies, one bullet should state that limit.
Return JSON: {"bullets":[{"text":"...","facts":["E1"]}, ...]}"""

JUDGE_SYSTEM = """You audit explanations of an audio deepfake detector. For each bullet decide if it is faithful:
faithful means every claim and number is directly supported by the cited facts, it does not contradict the verdict,
and it does not claim to know why the neural network decided. Be strict.
Return JSON: {"results":[{"index":0,"faithful":true,"reason":"..."}, ...],"overall":1-5}"""


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text)]


def _rule_check(bullet: dict, facts_by_id: dict, verdict: str) -> str | None:
    text, cited = bullet.get("text", ""), [f for f in bullet.get("facts", []) if f in facts_by_id]
    if not text or not cited:
        return "no valid fact citation"
    allowed = set()
    for f in facts_by_id.values():
        allowed.update(_numbers(f["text"]))
    for n in _numbers(text):
        if n <= 10 and float(n).is_integer():
            continue
        if not any(abs(n - a) <= max(0.011, 0.02 * abs(a)) for a in allowed):
            return f"number {n} not in facts"
    low = text.lower()
    if verdict == "synthetic" and re.search(r"\b(is|sounds) (genuine|real|authentic)\b", low):
        return "contradicts verdict"
    if verdict == "bona_fide" and re.search(r"\b(is|sounds) (synthetic|fake|cloned)\b", low):
        return "contradicts verdict"
    if re.search(r"\b(definitely|certainly|proves|100%)\b", low):
        return "overclaims certainty"
    return None


def explain(res: dict, n_speakers_train: int | None = None, use_llm: bool = True) -> dict:
    facts = build_facts(res, n_speakers_train)
    verdict = res["verdict"]
    out = {"facts": facts, "summary_ur": SUMMARY_UR[verdict], "source": "template",
           "note": "Generated from the detector's measurements. It explains the verdict and cannot change it."}
    client = _client() if use_llm else None
    if client is None:
        out["bullets"] = template_bullets(res, facts)
        return out
    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    judge_model = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
    fact_block = "\n".join(f"{f['id']}: {f['text']}" for f in facts)
    try:
        draft = _chat_json(client, model, JUSTIFIER_SYSTEM, f"Verdict: {VERDICT_WORD[verdict]}\nFacts:\n{fact_block}")
        bullets = [b for b in draft.get("bullets", []) if isinstance(b, dict)][:4]
        by_id = {f["id"]: f for f in facts}
        kept, dropped = [], []
        for b in bullets:
            why = _rule_check(b, by_id, verdict)
            (dropped if why else kept).append({**b, "rule": why})
        judge = {"model": judge_model, "overall": None}
        if kept:
            listing = "\n".join(f"{i}. {b['text']} [cites {', '.join(b['facts'])}]" for i, b in enumerate(kept))
            verdict_j = _chat_json(client, judge_model, JUDGE_SYSTEM,
                                   f"Verdict: {VERDICT_WORD[verdict]}\nFacts:\n{fact_block}\n\nBullets:\n{listing}")
            ok = {int(r.get("index", -1)): bool(r.get("faithful")) for r in verdict_j.get("results", [])}
            judge["overall"] = verdict_j.get("overall")
            final = [b for i, b in enumerate(kept) if ok.get(i, False)]
            dropped += [b for i, b in enumerate(kept) if not ok.get(i, False)]
        else:
            final = []
        judge["dropped"] = len(dropped)
        if len(final) >= 2:
            out.update(bullets=[b["text"] for b in final], cited=[b["facts"] for b in final], source="llm_judged",
                       judge=judge, llm_model=model,
                       note="Written by an LLM from the detector's measured output and checked by a second LLM judge. "
                            "It explains the verdict and cannot change it.")
        else:
            out.update(bullets=template_bullets(res, facts), judge=judge, llm_model=model)
    except Exception as e:  # provider down, quota, bad JSON: never block the verdict
        out.update(bullets=template_bullets(res, facts), llm_error=type(e).__name__)
    return out
