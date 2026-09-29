"""Stage 2 (audit) and stage 3 (classical baselines). Everything here runs on CPU in minutes."""
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

from . import config as C
from .data import length_match, protocol_frames, view
from .evaluate import dump, sliced

log = logging.getLogger("sachai")


def _len_feats(texts):
    t = pd.Series(list(texts))
    return np.c_[np.log1p(t.str.split().str.len()), np.log1p(t.str.len()), t.str.count("۔") + t.str.count(r"\.")]


def log_odds(texts, y, top=30):
    """Monroe et al. (2008) log-odds with informative Dirichlet prior -> tokens most indicative of each class."""
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=5)
    X = cv.fit_transform(texts)
    y = np.asarray(y)
    f, r = np.asarray(X[y == 1].sum(0)).ravel(), np.asarray(X[y == 0].sum(0)).ravel()
    a = f + r + 0.01
    a0 = a.sum()
    d = np.log((f + a) / (f.sum() + a0 - f - a)) - np.log((r + a) / (r.sum() + a0 - r - a))
    z = d / np.sqrt(1 / (f + a) + 1 / (r + a))
    vocab = cv.get_feature_names_out()
    o = np.argsort(z)
    return {"fake": [(vocab[i], round(float(z[i]), 1)) for i in o[::-1][:top]], "real": [(vocab[i], round(float(z[i]), 1)) for i in o[:top]]}


def audit(df: pd.DataFrame, out):
    rep = {}
    nw = df["text"].str.split().str.len()
    rep["length_words"] = df.assign(nw=nw).groupby(["source", "label"])["nw"].describe().round(1).reset_index().to_dict("records")

    # Length-only classifier: in-source AUROC (diagonal) and cross-source transfer (off-diagonal).
    # AUROC far from 0.5 = length leaks the label; a flip below 0.5 across sources = the 75% -> chance collapse.
    srcs = sorted(df.source.unique())
    mat = {}
    for a in srcs:
        tr = df[(df.source == a) & (df.split == "train")]
        clf = LogisticRegression(max_iter=1000).fit(_len_feats(tr.text), tr.label)
        mat[a] = {b: round(roc_auc_score(te.label, clf.predict_proba(_len_feats(te.text))[:, 1]), 3)
                  for b in srcs if (te := df[(df.source == b) & (df.split == "test")]).label.nunique() == 2}
    rep["length_only_auroc_train_row_test_col"] = mat

    rep["shortcut_tokens_raw"] = {s: log_odds(g.text, g.label) for s, g in df.groupby("source")}
    rep["shortcut_tokens_after_mask"] = {s: log_odds(view(g, True, "full"), g.label) for s, g in df.groupby("source")}
    rep["domains"] = df.groupby(["source", "domain"]).size().unstack(fill_value=0).to_dict("index")
    dump(rep, out / "audit.json")
    for s, m in mat.items():
        log.info("length-only AUROC trained on %s: %s", s, m)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(srcs), figsize=(5 * len(srcs), 3.5))
        for ax, s in zip(np.atleast_1d(axes), srcs):
            for y, name in ((0, "real"), (1, "fake")):
                ax.hist(np.log10(nw[(df.source == s) & (df.label == y)]), bins=40, alpha=0.55, label=name, density=True)
            ax.set(title=s, xlabel="log10(words)")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "length_by_class.png", dpi=150)
        plt.close(fig)
    except ImportError:
        log.warning("matplotlib missing, skipped length figure")


BASELINES = {
    "majority": lambda: make_pipeline(FunctionTransformer(lambda t: np.zeros((len(t), 1))), LogisticRegression()),
    "length_only": lambda: make_pipeline(FunctionTransformer(_len_feats), LogisticRegression(max_iter=1000)),
    "tfidf_char_lr": lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000),
                                           LogisticRegression(C=4, max_iter=2000, class_weight="balanced")),
    "tfidf_word_char_lr": lambda: make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                                           TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000)),
                                                LogisticRegression(C=4, max_iter=2000, class_weight="balanced")),
}


def baselines(df: pd.DataFrame, out):
    protocols = ["pooled"] + [f"lodo:{s}" for s in sorted(df.source.unique())]
    runs = [(p, b, C.DEBUNK, C.TEXT_VIEW) for p in protocols for b in BASELINES]
    runs += [("pooled", b, d, v) for b in ("length_only", "tfidf_char_lr")                       # ablations
             for d, v in (("keep" if C.DEBUNK == "drop" else "drop", C.TEXT_VIEW), (C.DEBUNK, "full"))]
    for proto, name, debunk, tv in runs:
        mask = debunk == "drop"
        f = out / f"{name}__{proto.replace(':', '_')}__debunk-{debunk}_{tv}.json"
        if f.exists():
            continue
        tr, _, tests = protocol_frames(df, proto, debunk)
        if C.LENGTH_MATCH_TRAIN:
            tr = length_match(tr, C.SPLIT_SEED)
        model = BASELINES[name]().fit(view(tr, mask, tv).tolist(), tr.label)
        res = {t: sliced(te, model.predict_proba(view(te, mask, tv).tolist())[:, 1]) for t, te in tests.items()}
        dump({"protocol": proto, "model": name, "debunk": debunk, "view": tv, "n_train": len(tr), "results": res}, f)
        log.info("baseline %-20s %-14s debunk=%s %-4s macroF1=%s", name, proto, debunk, tv, {t: r["all"]["macro_f1"] for t, r in res.items()})
