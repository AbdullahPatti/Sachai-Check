"""Every knob for the Sachai Check text pipeline lives here. Edit this file, then press F5 on run_pipeline.py."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANUAL = ROOT / "data" / "manual"
OUTPUTS = ROOT / "outputs"

# ---- run control -------------------------------------------------------------
SMOKE = os.environ.get("SACHAI_SMOKE") == "1"   # tiny model + 400 rows, for checking the plumbing on a laptop
RUN_NAME = os.environ.get("SACHAI_RUN", "")     # empty -> continue the latest UNFINISHED run, else start a new one
FRESH = os.environ.get("SACHAI_NEW") == "1"     # force a brand-new run folder even if an unfinished one exists
STAGES = ["data", "audit", "baselines", "transformers", "judge", "report"]
SEEDS = [13, 42, 2026]                          # main experiment repeated over these; ablations use SEEDS[0]

# ---- sources -----------------------------------------------------------------
# urdufake = FIRE-2021 train+test (it already contains Bend-the-Truth and FIRE-2020, see README).
# notrifact is picked up automatically if you drop a csv into data/raw/notrifact/.
# mirfan899 is off: its real_fake_news file is paragraph fragments of the same articles (leakage).
SOURCES = ["urdufake", "axg", "hnb", "notrifact"]

# ---- preprocessing -----------------------------------------------------------
MIN_WORDS = 4
NEAR_DUP_COSINE = 0.90          # char-ngram TF-IDF cosine; items above this are one "story cluster"
DEBUNK = "drop"                 # "drop": remove rows whose headline is a debunk ("this old photo is not...") from
                                # both classes; debunk words are also masked at inference. "keep": leave everything in.
TEXT_VIEW = "lead"              # "lead" = headline/first ~64 words for every source (removes headline-vs-article
                                # granularity shortcut); "full" = whole article
LEAD_WORDS = 64
LENGTH_MATCH_TRAIN = True       # within source x label, subsample so length distributions of real/fake match
LENGTH_BINS = 10

# ---- splits ------------------------------------------------------------------
TEST_FRAC = 0.15
VAL_FRAC = 0.15
SPLIT_SEED = 1234

# ---- transformer training ----------------------------------------------------
MODELS = {
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "muril": "google/muril-base-cased",
    "mbert": "google-bert/bert-base-multilingual-cased",
}
MAIN_MODEL = "xlmr-large"
MAX_LEN = 128
BATCH = 32
EPOCHS = 6
PATIENCE = 2
SAVE_EVERY_MIN = 15               # mid-epoch resume checkpoint interval (power-cut protection)
LR = {"xlmr-large": 1e-5}       # default 2e-5 for the rest
DEFAULT_LR = 2e-5
WARMUP = 0.1
WEIGHT_DECAY = 0.01
SAMPLER_TEMPERATURE = 0.5       # source-balanced sampling: weight ~ (1 / n_source_label) ** T, so HnB can't drown the rest

# Each experiment = model x protocol x ablation. "pooled" = stratified group split over all sources,
# "lodo:<src>" = train on every other source, test on <src> (the cross-dataset protocol).
EXPERIMENTS = (
    [dict(name=f"main_s{s}", model=MAIN_MODEL, protocol="pooled", seed=s) for s in SEEDS]
    + [dict(name=f"lodo_{src}", model=MAIN_MODEL, protocol=f"lodo:{src}", seed=SEEDS[0]) for src in SOURCES]
    + [
        dict(name="abl_keep_debunk", model=MAIN_MODEL, protocol="pooled", seed=SEEDS[0], debunk="keep"),
        dict(name="abl_no_lenmatch", model=MAIN_MODEL, protocol="pooled", seed=SEEDS[0], length_match=False),
        dict(name="abl_full_text", model=MAIN_MODEL, protocol="pooled", seed=SEEDS[0], view="full"),
        dict(name="bb_xlmr_base", model="xlmr-base", protocol="pooled", seed=SEEDS[0]),
        dict(name="bb_muril", model="muril", protocol="pooled", seed=SEEDS[0]),
        dict(name="bb_mbert", model="mbert", protocol="pooled", seed=SEEDS[0]),
    ]
)
KEEP_CHECKPOINT = {"main_s13"}  # only the deployment model's weights are kept (xlmr-large = 2 GB each)

# ---- OOD / scope -------------------------------------------------------------
OOD_TPR = 0.95                  # threshold keeps 95% of in-distribution val inside the gate
MIN_URDU_RATIO = 0.5            # share of letters that must be Arabic-script to count as Urdu

# ---- evaluation --------------------------------------------------------------
BOOTSTRAP = 1000
DOMAIN_F1_GATE = 0.70           # per-domain macro-F1 below this is flagged in the report

# ---- LLM judge (Gemini + Google Search grounding) ---------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
JUDGE_MODEL = os.environ.get("SACHAI_JUDGE_MODEL", "gemini-2.5-flash")
JUDGE_TEST_SAMPLE = 150         # test items sent to the judge (cost control); all manual claims are always sent
JUDGE_CAN_OVERRIDE = True       # False = judge only explains (proposal wording); True = evidence can overturn classifier
JUDGE_MIN_CONF = 0.7            # judge must be at least this sure to override

if SMOKE:
    MODELS = {k: "optimum-intel-internal-testing/tiny-random-xlm-roberta" for k in MODELS}
    SEEDS = [13]
    EPOCHS, BATCH, BOOTSTRAP, JUDGE_TEST_SAMPLE = 1, 8, 50, 5
    EXPERIMENTS = [e for e in EXPERIMENTS if e["name"] in ("main_s13", "lodo_axg", "abl_keep_debunk")]
