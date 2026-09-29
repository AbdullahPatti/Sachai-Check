"""Fast self-check of the logic that silently corrupts results if it breaks. Run: python test_logic.py"""
import pandas as pd

from sachai import config as C
from sachai.data import debunk_framed, length_match, normalize, split
from sachai.judge import fuse
from sachai.ood import input_gate

# normalisation: Arabic letters -> Urdu, digits, diacritics, spelling variants, boilerplate
assert normalize("يك ۱۲") == "یک 12"
assert normalize("کے لئے") == "کے لیے" and normalize("کیلئے") == "کے لیے" and normalize("کردیا") == "کر دیا"
assert normalize("خبر تصویر کے کاپی رائٹ Getty Images") == "خبر"
assert normalize("مُحَمَّد") == "محمد"

# debunk framing is judged on the headline zone only
assert debunk_framed("وائرل تصویر پرانی ہے اور یہ واقعہ لاہور کا نہیں")
assert not debunk_framed("پاکستان اسٹاک ایکسچینج میں کاروبار کا مثبت دن")

# input gate
assert input_gate("پاکستان اسٹاک ایکسچینج میں کاروبار کا مثبت دن")["status"] == "ok"
assert input_gate("Pakistan ki team ne match jeet liya hai aur sab khush hain")["status"] == "roman_urdu"
assert input_gate("The weather in London is nice today again")["status"] == "not_urdu"
assert input_gate("ہیلو")["status"] == "too_short"

# fusion: decisive evidence overrides style (when allowed), NEI falls back to style with a caveat
ok = {"status": "ok"}
j = {"verdict": "REFUTED", "confidence": 0.9, "scope": "in_scope"}
f = fuse(ok, {"p_fake": 0.1, "ood_flag": False}, j)
assert f["verdict"] == ("likely_fake" if C.JUDGE_CAN_OVERRIDE else "likely_real") and any("disagrees" in c for c in f["caveats"])
f = fuse(ok, {"p_fake": 0.9, "ood_flag": True}, {"verdict": "NOT_ENOUGH_EVIDENCE", "confidence": 0.2, "scope": "out_of_scope_region"})
assert f["basis"] == "style" and f["verdict"] == "unverified" and len(f["caveats"]) == 3  # OOD + scope halve confidence
assert fuse({"status": "not_urdu"}, {}, None)["verdict"] == "cannot_assess"

# splits: a story cluster never straddles train/val/test; length matching equalises classes per bin
df = pd.DataFrame({"text": [" ".join(["لفظ"] * (5 + i % 40)) for i in range(2000)], "label": [i % 2 for i in range(2000)],
                   "source": ["a", "b"] * 1000, "cluster": [i // 2 for i in range(2000)]})
s = split(df)
assert s.groupby("cluster").split.nunique().max() == 1 and set(s.split) == {"train", "val", "test"}
lm = length_match(s[s.split == "train"], 0)
assert (lm.groupby("source").label.value_counts().unstack().nunique(axis=1) == 1).all()
print("all checks passed")
