"""Self-checks for external/run_external.py. No GPU, model or network needed.

    .venv/Scripts/python external/test_external.py
"""
import io
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_external as X  # noqa: E402


def test_select_fakes():
    cols = ["file", "rg", "row", "generator", "target", "source", "gender", "source_clip"]
    # XTTS-like: one target per row group, every clip a different sentence (no source recording)
    tts = [("f0", 10 * t + k, r, "xtts_v2", t, None, "Female", None) for t in (1, 2, 3) for k in range(3) for r in range(4)]
    # FreeVC-like: each row group holds 3 targets, one source recording each, converted 4 times
    vc = [("f1", k, 4 * j + c, "freevc24", t, 9.0, "Male", f"src{k}_{t}.wav")
          for k in range(4) for j, t in enumerate((1, 2, 3)) for c in range(4)]
    idx = pd.DataFrame(tts + vc, columns=cols)
    n = {"xtts_v2": 6, "freevc24": 6}                                        # 2 clips per target
    a, b = X.select_fakes(idx, n, max_per_source=1), X.select_fakes(idx, n, max_per_source=1)
    assert a.equals(b), "selection must be deterministic"
    counts = a.groupby(["generator", "target"]).size()
    assert (counts == 2).all() and len(counts) == 6, counts
    assert not a.duplicated(["file", "rg", "row"]).any()
    t = a[a.generator == "xtts_v2"]
    assert t.groupby("target").rg.nunique().max() == 1, "TTS: one row group per target is enough"
    v = a[a.generator == "freevc24"]
    assert (v.groupby("target").source_clip.nunique() == 2).all(), "FreeVC: copies of one recording are capped"
    assert v.rg.nunique() == 2, "a row group that is read serves all of its targets"


def test_split_speakers():
    man = pd.DataFrame({"speaker": ["K1", "K1", "K2", "K3", "K4", "K5", "K5", "K6"],
                        "label": [0, 1, 0, 0, 1, 0, 1, 1]})
    a, b = X.split_speakers(man), X.split_speakers(man)
    assert a == b
    cal, ev = set(a["calibration"]), set(a["evaluation"])
    assert not cal & ev and cal | ev == set(man.speaker)
    both = {"K1", "K5"}
    assert len(both & cal) == 1 and len(both & ev) == 1, "speakers with real and fake clips go to both sides"


def test_extract_urdu_is_safe():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tar_path = td / "a.tar"
        with tarfile.open(tar_path, "w") as tf:
            for name in ["kb/urdu/test/audio/111-29-m.m4a", "kb/punjabi/test/audio/222-5-f.m4a",
                         "../../evil/urdu/333-7-f.m4a", "kb/urdu/test/transcription.txt"]:
                data = name.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        out = td / "out"
        out.mkdir()
        n = X.extract_urdu(tar_path, out)
        assert n == 2 and sorted(p.name for p in out.iterdir()) == ["111-29-m.m4a", "333-7-f.m4a"]
        assert not (td.parent / "evil").exists(), "archive paths must never be used"
        assert X.extract_urdu(tar_path, out) == 2, "re-run skips existing files and still counts them"


def test_name_pattern():
    assert X.NAME_RE.search("844424932912223-1049-f.m4a").groups() == ("844424932912223", "1049", "f")
    assert X.NAME_RE.search("x.wav") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
