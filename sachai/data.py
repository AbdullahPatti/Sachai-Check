"""Load -> normalise -> clean -> de-duplicate -> split. Label convention everywhere: 1 = fake, 0 = real."""
import hashlib
import logging
import re
import tarfile
import unicodedata
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C

log = logging.getLogger("sachai")

# ---------------------------------------------------------------- download
_GH = "https://raw.githubusercontent.com/"
DOWNLOADS = {
    "axg/combined.csv": "Sheetal83/Ax-to-Grind-Urdu-Dataset/main/Combined .csv",
    "hnb/Fake News 12166.xlsx": "Sheetal83/Hook-and-Bait-Urdu/main/Fake News 12166.xlsx",
    "hnb/Fake News After 12166.xlsx": "Sheetal83/Hook-and-Bait-Urdu/main/Fake News After 12166.xlsx",
    "hnb/Fake_News_After_31190.xlsx": "Sheetal83/Hook-and-Bait-Urdu/main/Fake_News_After_31190.xlsx",
    "hnb/Final True News-11012.xlsx": "Sheetal83/Hook-and-Bait-Urdu/main/Final True News-11012.xlsx",
    "hnb/True_News_After_28478.xlsx": "Sheetal83/Hook-and-Bait-Urdu/main/True_News_After_28478.xlsx",
    "hnb/True_News_after_11012.xlsx": "Sheetal83/Hook-and-Bait-Urdu/main/True_News_after_11012.xlsx",
    "urdufake/train.zip": "MaazAmjad/Urdu-Fake-news-detection-FIRE2021/main/Training Dataset@FIRE2021.zip",
    "urdufake/test.zip": "MaazAmjad/Urdu-Fake-news-detection-FIRE2021/main/Test Dataset @ FIRE 2021.zip",
    "urdufake/test_labels.xlsx": "MaazAmjad/Urdu-Fake-news-detection-FIRE2021/main/Orignal_labels.xlsx",
    "ood/qa_gk.csv": "mirfan899/Urdu/master/qa/qa_gk.csv",
    "ood/products.tar.gz": "mirfan899/Urdu/master/sentiment/products_sentiment_urdu.csv.tar.gz",
    "ood/roman.tar.gz": "mirfan899/Urdu/master/sentiment/roman.csv.tar.gz",
}
_TAR_MEMBER = {"ood/products.tar.gz": ("products_sentiment_urdu.csv", "urdu_products_sentiment.csv"),
               "ood/roman.tar.gz": ("sentiment.csv", "roman_urdu_sentiment.csv")}


def download_all():
    for rel, repo_path in DOWNLOADS.items():
        dst = C.RAW / rel
        final = dst.parent / _TAR_MEMBER[rel][1] if rel in _TAR_MEMBER else dst.with_suffix("") if rel.endswith(".zip") else dst
        if final.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", rel)
        urllib.request.urlretrieve(_GH + urllib.parse.quote(repo_path), dst)
        if rel.endswith(".zip"):
            zipfile.ZipFile(dst).extractall(dst.with_suffix(""))
            dst.unlink()
        if rel in _TAR_MEMBER:
            with tarfile.open(dst) as t:
                m = next(m for m in t.getmembers() if m.name.endswith(_TAR_MEMBER[rel][0]))
                final.write_bytes(t.extractfile(m).read())
            dst.unlink()


# ---------------------------------------------------------------- normalisation
_CHARMAP = str.maketrans({
    0x064A: 0x06CC, 0x0649: 0x06CC,   # Arabic yeh / alef maksura -> Farsi yeh
    0x0643: 0x06A9,                   # Arabic kaf -> keheh
    0x0647: 0x06C1,                   # Arabic heh -> heh goal
    0x06C0: 0x06C2,                   # heh+yeh -> heh goal+hamza
    **{0x06F0 + i: str(i) for i in range(10)},   # Urdu digits
    **{0x0660 + i: str(i) for i in range(10)},   # Arabic digits
    0x0640: None,                     # tatweel
})
_DIACRITICS = re.compile("[%s-%s%s%s-%s]" % tuple(map(chr, (0x064B, 0x065F, 0x0670, 0x06D6, 0x06ED))))
_INVISIBLE = re.compile("[%s-%s%s-%s%s-%s%s%s]" % tuple(map(chr, (0x200B, 0x200F, 0x202A, 0x202E, 0x2066, 0x2069, 0xFEFF, 0x00AD))))
_QUOTES = " '\"" + "".join(map(chr, (0x2018, 0x2019, 0x201C, 0x201D)))
_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"[@#]\w+")
_WS = re.compile(r"\s+")
# Outlet boilerplate (BBC Urdu "image copyright/caption", agency tags, "read more"): tells the model the source, not the truth.
_BOILER = re.compile(r"(تصویر کے )?کاپی رائٹ(\s+[A-Za-z]+)*|(?i:image caption|getty images|afp|reuters|epa)|مزید پڑھیے|مزید پڑھیں|اس سے متعلق مزید")
# House-style spelling variants and writer typos. The audit showed لئے/لیے alone separates classes in AXG
# (and flips direction in HnB), and the Bend-the-Truth journalists' typos (مذید, بلکل) mark fakes.
_SPELLING = {
    "لئے": "لیے", "لیئے": "لیے", "کیلئے": "کے لیے", "کیلیے": "کے لیے", "دئیے": "دیے", "دئے": "دیے",
    "امریکا": "امریکہ", "کرونا": "کورونا", "بلکل": "بالکل", "مذید": "مزید",
    "انکا": "ان کا", "انکی": "ان کی", "انکے": "ان کے", "اسکا": "اس کا", "اسکی": "اس کی", "اسکے": "اس کے",
    **{a + b: a + " " + b for a in ("کر", "ہو") for b in ("دیا", "دی", "دیے", "لیا", "لی", "کے", "گیا", "گئی", "گئے", "رہا", "رہی", "رہے", "سکتا", "سکتی", "سکتے")},
}
_SPELL_RE = re.compile(r"(?<!\w)(" + "|".join(sorted(_SPELLING, key=len, reverse=True)) + r")(?!\w)")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))  # also folds Arabic presentation forms (U+FB50-FEFF) to base letters
    t = _INVISIBLE.sub("", t)
    t = _DIACRITICS.sub("", t.translate(_CHARMAP))
    t = _URL.sub(" ", t)
    t = _HANDLE.sub(" ", t)
    t = _BOILER.sub(" ", t)
    t = _SPELL_RE.sub(lambda m: _SPELLING[m.group(0)], t)
    return _WS.sub(" ", t).strip(_QUOTES)


def urdu_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(0x0600 <= ord(c) <= 0x06FF for c in letters) / max(len(letters), 1)


def _words_re(words):
    return re.compile(r"(?<!\w)(" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) + r")(?!\w)")


# Fact-checker meta-vocabulary. Many Hook-and-Bait / AXG "fakes" are debunk headlines ("this old photo is not
# of X"), i.e. a TRUE statement about a fake. audit.py prints the top class tokens per source; extend from it.
DEBUNK_TERMS = [normalize(w) for w in [
    "فیکٹ چیک", "گمراہ کن", "من گھڑت", "بے بنیاد", "ڈیپ فیک", "وائرل", "دعویٰ", "دعوی", "دعوے", "فرضی",
    "جعلی", "فیک", "جھوٹا", "جھوٹی", "جھوٹ", "غلط", "پڑتال", "ایڈٹ", "ایڈیٹڈ", "مسخ", "افواہ", "افواہیں",
    "حقیقت", "تصدیق", "سچ",
]]
MEDIA_TERMS = [normalize(w) for w in ["تصویر", "تصاویر", "ویڈیو", "شیئر", "پوسٹ", "سوشل میڈیا", "دکھایا", "پرانی", "پرانا"]]
_DEBUNK = _words_re(DEBUNK_TERMS)
_MEDIA = _words_re(MEDIA_TERMS)


def debunk_framed(text: str) -> bool:
    """Headline zone (first 25 words) talks about a claim/photo/video instead of stating news."""
    head = lead(text, 25)
    return bool(_DEBUNK.search(head) or _MEDIA.search(head))


def mask_debunk(text: str) -> str:
    return _WS.sub(" ", _DEBUNK.sub(" ", text)).strip()


def lead(text: str, n: int = None) -> str:
    return " ".join(text.split()[: n or C.LEAD_WORDS])


def view(df: pd.DataFrame, mask: bool, text_view: str) -> pd.Series:
    """The exact string the models see. Used identically for train, test and inference."""
    t = df["text"]
    if mask:
        t = t.map(mask_debunk)
    return t.map(lead) if text_view == "lead" else t


# ---------------------------------------------------------------- loaders
_FAKE = {"fake", "unreal", "false", "f", "1"}
_REAL = {"true", "real", "r", "t", "0"}
_FIRE_DOMAIN = {"bs": "business", "bus": "business", "hlth": "health", "sbz": "showbiz",
                "sp": "sports", "tc": "technology", "tch": "technology"}


def _label(v) -> float:
    v = str(v).strip().lower()
    return 1 if v in _FAKE else 0 if v in _REAL else np.nan


def load_urdufake() -> pd.DataFrame:
    root = C.RAW / "urdufake"
    rows = []
    for f in (root / "train").rglob("*.txt"):
        prefix = re.sub(r"\d+$", "", f.stem).lower()
        rows.append((f.read_text("utf-8-sig", errors="replace"), f.parent.name, _FIRE_DOMAIN.get(prefix, "unknown"), f"train/{f.parent.parent.name}/{f.parent.name}/{f.name}"))
    labels = pd.read_excel(root / "test_labels.xlsx").set_index("File No")["Real/Fake"]
    for f in (root / "test").rglob("*.txt"):
        rows.append((f.read_text("utf-8-sig", errors="replace"), labels.get(int(f.stem)), "unknown", f"test/{f.name}"))
    return pd.DataFrame(rows, columns=["text", "label", "domain", "orig_id"])


def load_axg() -> pd.DataFrame:
    d = pd.read_csv(C.RAW / "axg" / "combined.csv")
    return pd.DataFrame({"text": d["News Items"], "label": d["Label"], "domain": "unknown", "orig_id": d["Sr. No."].astype(str)})


def load_hnb() -> pd.DataFrame:
    parts = []
    for f in sorted((C.RAW / "hnb").glob("*.xlsx")):
        d = pd.read_excel(f).rename(columns=lambda c: str(c).strip())
        parts.append(pd.DataFrame({"text": d["News Items"], "label": d["Label"], "domain": "unknown",
                                   "orig_id": f.stem + ":" + d["Sr. No."].astype(str)}))
    return pd.concat(parts)


def load_notrifact() -> pd.DataFrame:
    files = [f for f in (C.RAW / "notrifact").glob("*") if f.suffix in (".csv", ".xlsx", ".tsv")]
    if not files:
        return pd.DataFrame()
    d = pd.concat(pd.read_excel(f) if f.suffix == ".xlsx" else pd.read_csv(f, sep=None, engine="python") for f in files)
    d.columns = [str(c).strip().lower() for c in d.columns]
    obj = [c for c in d.columns if d[c].dtype == object]
    text_col = max(obj, key=lambda c: d[c].astype(str).str.len().mean())
    label_col = next(c for c in d.columns if c != text_col and d[c].astype(str).str.strip().str.lower().isin(_FAKE | _REAL).mean() > 0.95)
    dom = d["domain"] if "domain" in d.columns else d["category"] if "category" in d.columns else "unknown"
    log.info("notrifact: text=%r label=%r", text_col, label_col)
    return pd.DataFrame({"text": d[text_col], "label": d[label_col], "domain": dom, "orig_id": d.index.astype(str)})


LOADERS = {"urdufake": load_urdufake, "axg": load_axg, "hnb": load_hnb, "notrifact": load_notrifact}
PRIORITY = {s: i for i, s in enumerate(["urdufake", "axg", "notrifact", "hnb"])}  # who keeps a cross-source duplicate


# ---------------------------------------------------------------- dedup
def near_dup_clusters(texts: pd.Series, thr: float) -> np.ndarray:
    """Cluster id per row: rows linked by char-4gram TF-IDF cosine >= thr end up in one cluster."""
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=2, max_df=0.02, dtype=np.float32).fit_transform(texts)
    rows, cols = [], []
    for i in range(0, X.shape[0], 2000):  # ponytail: chunked sparse cosine, O(n^2); swap in MinHash-LSH past ~500k rows
        S = (X[i:i + 2000] @ X.T).tocoo()
        keep = (S.data >= thr) & (S.row + i != S.col)
        rows.append(S.row[keep] + i)
        cols.append(S.col[keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(X.shape[0],) * 2)
    return connected_components(A, directed=False)[1]


def build_corpus(report: dict) -> pd.DataFrame:
    frames = []
    for src in C.SOURCES:
        d = LOADERS[src]()
        if d.empty:
            log.warning("source %s: no files found, skipped", src)
            continue
        d["source"] = src
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    report["raw_rows"] = df.groupby("source").size().to_dict()

    df["label"] = df["label"].map(_label)
    df["text"] = df["text"].astype(str).map(normalize)
    steps = {}

    def drop(mask, why):
        steps[why] = df[mask].groupby("source").size().to_dict()
        return df[~mask]

    df = drop(df["label"].isna(), "bad_label")
    df = drop(df["text"].str.split().str.len() < C.MIN_WORDS, "too_short")
    df = drop(df["text"].map(urdu_ratio) < C.MIN_URDU_RATIO, "not_urdu_script")
    df["label"] = df["label"].astype(int)
    df["debunk"] = df["text"].map(debunk_framed)
    report["debunk_framed_rate"] = df.groupby(["source", "label"])["debunk"].mean().round(3).unstack().to_dict("index")

    # exact duplicates (on punctuation/space-stripped text), then near duplicates
    df["key"] = df["text"].str.replace(r"[\W_]+", "", regex=True).map(lambda s: hashlib.md5(s.encode()).hexdigest())
    df = df.reset_index(drop=True)
    df["cluster"] = near_dup_clusters(df["text"], C.NEAR_DUP_COSINE)
    df["cluster"] = df.groupby("key")["cluster"].transform("min")  # exact dups always share a cluster
    df["cluster"] = pd.factorize(df["cluster"])[0]

    # A cluster holding both labels is a minimal pair (Bend-the-Truth fakes are edits of real articles:
    # "flights start" -> "flights stop"). Keep one real + one fake per cluster; the group split keeps them together.
    df["pair"] = df.groupby("cluster")["label"].transform("nunique") > 1
    report["minimal_pair_clusters"] = int(df.loc[df.pair, "cluster"].nunique())
    report["minimal_pair_examples"] = df[df.pair].sort_values("cluster").head(6)[["cluster", "source", "label", "text"]].to_dict("records")
    multi = df.groupby("cluster")["source"].transform("nunique") > 1
    report["rows_in_cross_source_clusters"] = df[multi].groupby("source").size().to_dict()
    report["cross_source_overlap"] = {f"{a}&{b}": int(len(set(df.cluster[df.source == a]) & set(df.cluster[df.source == b])))
                                      for i, a in enumerate(C.SOURCES) for b in C.SOURCES[i + 1:]}

    df = df.assign(_p=df["source"].map(PRIORITY)).sort_values(["cluster", "label", "_p"])
    before = df.groupby("source").size()
    df = df.drop_duplicates(["cluster", "label"]).drop(columns=["_p", "key"])
    steps["duplicates"] = (before - df.groupby("source").size()).fillna(0).astype(int).to_dict()

    report["dropped"] = steps
    report["clean_rows"] = df.groupby(["source", "label"]).size().unstack().to_dict("index")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- splits
def _group_holdout(d: pd.DataFrame, frac: float, seed: int) -> pd.Index:
    """Index of ~frac of rows, whole clusters only, stratified by source x label."""
    k = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    _, hold = next(sgkf.split(d, d["source"] + "_" + d["label"].astype(str), d["cluster"]))
    return d.index[hold]


def split(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified group split: a story cluster (incl. its real/fake minimal pair) never straddles splits."""
    df = df.copy()
    df["split"] = "train"
    test = _group_holdout(df, C.TEST_FRAC, C.SPLIT_SEED)
    df.loc[test, "split"] = "test"
    rest = df[df.split == "train"]
    df.loc[_group_holdout(rest, C.VAL_FRAC / (1 - C.TEST_FRAC), C.SPLIT_SEED + 1), "split"] = "val"
    assert df.groupby("cluster")["split"].nunique().max() == 1
    return df


def length_match(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Per source: bin by word count, keep equal real/fake per bin -> length carries no label signal."""
    out = []
    for _, d in train.groupby("source"):
        n = d["text"].str.split().str.len()
        bins = pd.qcut(n.rank(method="first"), C.LENGTH_BINS, labels=False)
        for _, b in d.groupby(bins):
            k = b["label"].value_counts().min() if b["label"].nunique() == 2 else 0
            out += [b[b.label == y].sample(k, random_state=seed) for y in (0, 1)]
    return pd.concat(out).sample(frac=1, random_state=seed)  # shuffled


def protocol_frames(df: pd.DataFrame, protocol: str, debunk: str = None):
    """-> (train, val, {test_name: test_df}). lodo:<src> tests on ALL of <src>, trains on the rest.
    debunk="drop" removes debunk-framed rows (both classes, every split); "keep" leaves them in."""
    if (debunk or C.DEBUNK) == "drop":
        df = df[~df.debunk]
    if protocol == "pooled":
        return df[df.split == "train"], df[df.split == "val"], {"pooled": df[df.split == "test"]}
    src = protocol.split(":")[1]
    held = df[df.source == src]
    rest = df[(df.source != src) & ~df.cluster.isin(held.cluster)]  # no story of the held-out set leaks into training
    return rest[rest.split == "train"], rest[rest.split == "val"], {f"heldout_{src}": held}


def load_ood_sets() -> dict:
    """Never-seen text for the OOD gate: Urdu non-news (far), Roman-Urdu (script shift)."""
    o = C.RAW / "ood"
    gk = pd.read_csv(o / "qa_gk.csv")
    prod = pd.read_csv(o / "urdu_products_sentiment.csv", sep=None, engine="python", on_bad_lines="skip")
    roman = pd.read_csv(o / "roman_urdu_sentiment.csv", on_bad_lines="skip")
    sets = {
        "far_urdu_qa": (gk["Question"].astype(str) + " " + gk["Answer"].astype(str)),
        "far_urdu_reviews": prod["sentence"].astype(str),
        "script_roman_urdu": roman["sentence"].astype(str),
    }
    return {k: v.map(normalize).loc[lambda s: s.str.split().str.len() >= C.MIN_WORDS].drop_duplicates().sample(frac=1, random_state=0).head(2000).tolist()
            for k, v in sets.items()}
