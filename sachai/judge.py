"""LLM layer (Gemini, Interactions API, google-genai >= 2.3): retrieval-grounded judge, scope check,
Roman-Urdu transliteration, domain tagging, and a second "meta-judge" that grades the judge's explanations.

Design choices worth defending in the report:
  * The judge is BLIND to the classifier's output (no anchoring); the two signals are fused afterwards (fuse()).
  * Today's date goes into every prompt and the judge must rule on the CURRENT state of the world, citing dated
    evidence -> handles recency ("Spain won the 2026 World Cup") and war->peace style contradictions.
  * Every response is cached on disk (outputs/llm_cache.jsonl) so re-runs cost nothing and are reproducible.
"""
import datetime as dt
import hashlib
import json
import logging
import re
import time

from . import config as C

log = logging.getLogger("sachai")
CACHE = C.OUTPUTS / "llm_cache.jsonl"
_cache = None
_client = None

VERDICTS = ["SUPPORTED", "REFUTED", "MIXED", "NOT_ENOUGH_EVIDENCE"]
SCOPES = ["in_scope", "out_of_scope_region", "not_a_claim"]
DOMAINS = ["politics", "business", "health", "sports", "technology", "showbiz", "education", "religion",
           "international", "crime", "weather", "social", "other"]

JUDGE_PROMPT = """You are the evidence judge of "Sachai Check", an Urdu fake-news verification system.
Today's date is {today}. Use Google Search before answering. Search in Urdu AND English.

CLAIM (Urdu):
<<<{claim}>>>

Do all of the following:
1. scope: "in_scope" if it is a factual news claim about Pakistan or South Asia; "out_of_scope_region" if it is a
   factual news claim about somewhere else; "not_a_claim" if it is opinion, a question, satire or not news.
2. Find evidence. Prefer reputable news outlets and fact-checkers. Record each source's publication date.
3. If the facts changed over time (e.g. a war that later ended in a peace deal), judge the claim against the
   CURRENT state of the world as of today, and explain the change in "temporal_note".
4. verdict: SUPPORTED (evidence confirms it), REFUTED (evidence contradicts it or a fact-check debunks it),
   MIXED (partly true), NOT_ENOUGH_EVIDENCE (nothing reliable found - do NOT guess).
5. confidence: 0.0-1.0, how sure you are of the verdict given the evidence you actually found.

Reply with ONLY this JSON, no markdown:
{{"scope": "...", "scope_reason": "...", "claim_en": "<English translation>", "verdict": "...", "confidence": 0.0,
 "explanation_ur": "<2-4 sentences in Urdu>", "explanation_en": "<2-4 sentences in English>",
 "evidence": [{{"title": "...", "url": "...", "date": "YYYY-MM-DD or unknown", "stance": "supports|refutes|context", "summary": "..."}}],
 "temporal_note": "..."}}"""

META_PROMPT = """You grade explanations written by an automated fact-checking judge (LLM-as-a-judge, rubric scoring).
CLAIM: {claim}
JUDGE VERDICT: {verdict} (confidence {confidence})
JUDGE EXPLANATION: {explanation}
EVIDENCE THE JUDGE CITED: {evidence}

Score 1-5 each: faithfulness (explanation only states things the cited evidence supports), relevance (evidence is
about this exact claim), recency (evidence is dated and recent enough for the claim), clarity (a non-expert Urdu
reader would understand it), calibration (confidence matches evidence strength).
Reply with ONLY JSON: {{"faithfulness": 0, "relevance": 0, "recency": 0, "clarity": 0, "calibration": 0, "comment": "..."}}"""

TRANSLIT_PROMPT = """Transliterate this Roman Urdu text into standard Urdu (Nastaliq/Perso-Arabic) script.
Keep the meaning and names exactly; do not translate English loanwords, write them in Urdu script.
Reply with ONLY the Urdu text.
TEXT: {text}"""

ROMANIZE_PROMPT = """Write this Urdu text in Roman Urdu the way Pakistanis type on WhatsApp (Latin letters,
informal spelling). Reply with ONLY the Roman Urdu text.
TEXT: {text}"""

DOMAIN_PROMPT = """For each numbered Urdu news item, pick ONE domain from {domains}.
Reply with ONLY a JSON list of strings in the same order, length {n}.
{items}"""


# ------------------------------------------------------------------ plumbing
def available() -> bool:
    return bool(C.GEMINI_API_KEY)


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    _cache[r["k"]] = r["v"]
                except json.JSONDecodeError:  # torn last line after a power cut: that call is simply redone
                    pass
    return _cache


def _call(kind: str, prompt: str, search: bool = False) -> dict:
    """-> {"text": str, "citations": [{url,title}]}. Cached; retried with backoff."""
    global _client
    key = hashlib.md5(f"{kind}|{C.JUDGE_MODEL}|{prompt}".encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=C.GEMINI_API_KEY)
    kw = {"tools": [{"type": "google_search"}]} if search else {}
    for attempt in range(6):
        try:
            it = _client.interactions.create(model=C.JUDGE_MODEL, input=prompt, **kw)
            break
        except Exception as e:  # rate limits / transient 5xx
            if attempt == 5:
                raise
            wait = 2 ** attempt * 5
            log.warning("gemini %s failed (%s); retry in %ds", kind, str(e)[:120], wait)
            time.sleep(wait)
    cites = []
    for step in getattr(it, "steps", None) or []:
        for block in getattr(step, "content", None) or []:
            for a in getattr(block, "annotations", None) or []:
                if getattr(a, "type", "") == "url_citation":
                    cites.append({"url": getattr(a, "url", ""), "title": getattr(a, "title", "")})
    v = {"text": it.output_text or "", "citations": list({c["url"]: c for c in cites}.values())}
    cache[key] = v
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"k": key, "v": v}, ensure_ascii=False) + "\n")
    return v


def _json(text: str, default):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    try:
        return json.loads(m.group(0)) if m else default
    except json.JSONDecodeError:
        return default


# ------------------------------------------------------------------ public calls
def judge_claim(claim: str) -> dict:
    r = _call("judge", JUDGE_PROMPT.format(today=dt.date.today().isoformat(), claim=claim), search=True)
    j = _json(r["text"], {})
    j["verdict"] = j.get("verdict") if j.get("verdict") in VERDICTS else "NOT_ENOUGH_EVIDENCE"
    j["scope"] = j.get("scope") if j.get("scope") in SCOPES else "in_scope"
    try:
        j["confidence"] = min(max(float(j.get("confidence", 0)), 0.0), 1.0)
    except (TypeError, ValueError):
        j["confidence"] = 0.0
    j["grounding_citations"] = r["citations"]  # URLs the search tool actually returned (not model-written)
    j["parse_ok"] = bool(r["text"]) and "explanation_en" in j
    return j


def rate_explanation(claim: str, j: dict) -> dict:
    r = _call("meta", META_PROMPT.format(claim=claim, verdict=j["verdict"], confidence=j["confidence"],
                                         explanation=j.get("explanation_en", ""),
                                         evidence=json.dumps(j.get("evidence", [])[:6], ensure_ascii=False)))
    return _json(r["text"], {})


def transliterate(roman: str) -> str:
    return _call("translit", TRANSLIT_PROMPT.format(text=roman))["text"].strip()


def romanize(urdu: str) -> str:
    return _call("romanize", ROMANIZE_PROMPT.format(text=urdu))["text"].strip()


def annotate_domains(texts: list) -> list:
    out = []
    for i in range(0, len(texts), 25):
        chunk = texts[i:i + 25]
        items = "\n".join(f"{k + 1}. {t[:300]}" for k, t in enumerate(chunk))
        lab = _json(_call("domain", DOMAIN_PROMPT.format(domains=DOMAINS, n=len(chunk), items=items))["text"], [])
        lab = lab if isinstance(lab, list) and len(lab) == len(chunk) else ["unknown"] * len(chunk)
        out += [d if d in DOMAINS else "unknown" for d in lab]
    return out


# ------------------------------------------------------------------ fusion
def fuse(gate: dict, clf: dict, j: dict | None) -> dict:
    """Final verdict from classifier (style) + judge (evidence). Returns the response the backend will store."""
    caveats = []
    if gate["status"] not in ("ok", "roman_urdu"):
        return {"verdict": "cannot_assess", "confidence": 0.0, "basis": "input_gate", "caveats": [f"input rejected: {gate['status']}"]}
    if gate["status"] == "roman_urdu":
        caveats.append("Roman Urdu input was machine-transliterated before analysis")
    if clf.get("ood_flag"):
        caveats.append("text is unlike the training data (distribution shift) - style signal is less reliable")
    if j and j["scope"] != "in_scope":
        caveats.append({"out_of_scope_region": "claim is about a region outside the system's Pakistan/South-Asia focus",
                        "not_a_claim": "input does not look like a checkable news claim"}[j["scope"]])

    style_fake = clf["p_fake"] >= 0.5
    style_conf = max(clf["p_fake"], 1 - clf["p_fake"]) * (0.5 if caveats else 1.0)
    decisive = j and j["verdict"] in ("SUPPORTED", "REFUTED") and j["confidence"] >= C.JUDGE_MIN_CONF
    if decisive:
        ev_fake = j["verdict"] == "REFUTED"
        if ev_fake != style_fake:
            caveats.append("live evidence disagrees with the writing-style signal")
        if C.JUDGE_CAN_OVERRIDE:
            return {"verdict": "likely_fake" if ev_fake else "likely_real", "confidence": round(j["confidence"], 3),
                    "basis": "evidence", "caveats": caveats}
    elif j:
        caveats.append({"MIXED": "evidence is mixed - parts of the claim are true",
                        "NOT_ENOUGH_EVIDENCE": "no reliable live evidence found - verdict rests on writing style only"}
                       .get(j["verdict"], "judge was not confident"))
    else:
        caveats.append("evidence judge unavailable - verdict rests on writing style only")
    return {"verdict": ("likely_fake" if style_fake else "likely_real") if style_conf >= 0.6 else "unverified",
            "confidence": round(style_conf, 3), "basis": "style", "caveats": caveats}
