# Sachai Check — Urdu fake-news text pipeline

One file to run: **`run_pipeline.py`** (open it, press F5). All data is already in `data/raw/`
(it re-downloads anything missing). Results land in `outputs/runs/<timestamp>/`.

## Run it on the GPU PC

1. Copy this whole folder to the GPU PC.
2. Install **Python 3.11 or 3.12**. (3.14 works but CUDA wheels lag behind.)
3. Install the CUDA build of PyTorch. Pick the index that matches `nvidia-smi` (cu121 / cu124 / cu128):
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
   (`run_pipeline.py` installs anything else that is missing, but not torch.)
4. Optional but recommended: set `GEMINI_API_KEY` as an environment variable. Without it the judge stage and
   the domain tagging are skipped. Everything else still runs.
5. Open `run_pipeline.py` → F5.

- **Power cut / crash / PC restart:** just press F5 again. It continues the newest unfinished run: finished stages
  and experiments are skipped, and an interrupted training run continues from its last checkpoint (saved every 15 min
  and at every epoch end), not from the beginning. Force a brand-new run with `SACHAI_NEW=1`. Re-open an older
  finished run (e.g. to add the judge after setting the API key) with `SACHAI_RUN=<folder name>`.
- **Check the plumbing on a laptop in ~3 min:** set `SACHAI_SMOKE=1` (tiny model, small data slice).
- **Time:** roughly 5–8 h on one 24 GB GPU (RTX 4090 / A5000 class) for all 13 transformer runs. This is an estimate, not
  measured, since only the CPU smoke test was run here. If a card has less than 24 GB, set `BATCH = 16` in `sachai/config.py`.
- **Self-test:** `python test_logic.py`

## What F5 does

| Stage | What | Output |
|---|---|---|
| data | load 3 datasets (+ Notri-Fact if present) → normalise → filter → exact + near-duplicate clustering → group-stratified 70/15/15 split | `data/corpus.csv`, `data/data_report.json` |
| audit | length confound per class, length-only cross-dataset transfer, class-indicative tokens (log-odds) per source | `audit/audit.json`, `audit/length_by_class.png` |
| baselines | majority, length-only, TF-IDF char, TF-IDF word+char × {pooled, leave-one-dataset-out} + ablations | `baselines/*.json` |
| transformers | XLM-R-large ×3 seeds, LODO per dataset, 3 ablations, 3 other backbones; temperature calibration; OOD detector fit + evaluation | `experiments/<name>/metrics.json`, `preds_*.csv`, deployment model in `experiments/main_s13/model/` |
| judge | full system (gate → classifier → OOD → Gemini + Google Search judge → fusion) on manual claims + 150 test items; meta-judge grades explanations; Roman-Urdu round-trip | `judge/judge_metrics.json`, `judge/system_predictions.csv` |
| report | every table above in one markdown file | `report/summary.md` ← paste into the report |

The log is in `logs/pipeline.log`. The config snapshot and exact environment (GPU, `pip freeze`) are in `logs/`.

## Files you add yourselves

- `data/raw/notrifact/` — drop the Notri-Fact csv/xlsx here (it is not publicly downloadable). The text and label
  columns are auto-detected, and it joins the pooled and leave-one-out protocols automatically.
- `data/manual/claims.csv` — the 45 SC-7 hand-verified claims. Columns: `id,text,label(real|fake),script(urdu|roman),
  domain,scope(in_scope|out_of_scope_region|not_a_claim),event_date,source_url,added_by,notes`. Deliberately include
  recent (2025–26) claims, non-Pakistan claims and Roman-Urdu claims. These are the only test of recency and scope.

## Deployment (for the FastAPI backend later)

```python
from sachai.infer import Predictor, verify
pred = Predictor("outputs/runs/<run>/experiments/main_s13/model")   # once at startup
verify("<Urdu or Roman-Urdu text>", pred)                          # per request -> JSON
```

---

## Methodology notes (for the mid-evaluation report)

### Pipeline
```
input ─► input gate (empty / too short / non-Urdu / Roman-Urdu → Gemini transliteration)
      ─► stylistic classifier (XLM-R-large, calibrated)  +  Mahalanobis OOD score on its [CLS] embedding
      ─► Gemini judge with live Google Search (blind to the classifier): scope, verdict, dated evidence
      ─► fusion: decisive evidence (≥0.7 conf) decides; otherwise style verdict with caveats; OOD /
         out-of-scope halve the style confidence → "likely_fake" | "likely_real" | "unverified" | "cannot_assess"
```

### Data findings (measured locally on the downloaded data — all reproduced by the `data` + `audit` stages)

| Finding | Evidence | What the pipeline does |
|---|---|---|
| FIRE-2021 train ⊃ Bend-the-Truth ⊃ FIRE-2020 | file counts 1300 = 900 + 400 | one source `urdufake` (1,600 items incl. labelled FIRE-21 test) |
| **Hook-and-Bait contains almost all of Ax-to-Grind** | 8,740 shared story clusters | cross-source dedup (keep AXG copy); LODO excludes shared clusters. *Published AXG↔HnB transfer numbers are contaminated.* |
| Bend-the-Truth fakes are minimal edits of real articles ("flights start" → "flights stop") | 115 real/fake pairs at cosine ≥ 0.9 | keep both, force them into the same split (group split), report accuracy on pairs separately |
| Length cue flips direction between datasets | length-only AUROC 0.65–0.75 in-dataset, **0.28–0.30** AXG/HnB → urdufake | this is the 75% → chance collapse. Fixes: length-matched training, lead-64-words view, length-quintile slices |
| House-style spelling (لئے vs لیے, کردیا vs کر دیا) was the #1 AXG feature, reversed in HnB | log-odds z = 14 / −22 | spelling normalisation |
| Writer typos / outlet boilerplate (مذید, بلکل, BBC "image copyright") | top urdufake tokens | normalised / stripped |
| 40% of HnB fakes and 26% of AXG fakes are **debunk headlines** ("this old photo is not…"), vs about 6% of reals | debunk-framed rate | drop debunk-framed rows from both classes (ablation keeps them); mask the words at inference |
| HnB fakes are mostly translated US fact-checks; reals are Pakistani outlets | "Trump/Clinton" = fake, "Pakistan/Kashmir" = real tokens | cannot be fully removed. It is measured by the `by_geo` slice and is exactly why the OOD/scope layer exists |
| mirfan899 `real_fake_news` = paragraph fragments of the same articles; `urdu-news-1M` is real-only | inspection | **excluded** (fragments leak across splits; real-only data turns "source" into the label) |

Classical baselines on this cleaned data (CPU, local run): TF-IDF pooled macro-F1 **0.89**, but leave-one-dataset-out
only **0.55–0.62**. This gap is the central result: in-distribution scores massively overstate real-world skill.

### Evaluation protocol
- **Pooled:** group-stratified split, 3 seeds, mean ± std, bootstrap 95% CI.
- **Leave-one-dataset-out (LODO):** the cross-dataset standing protocol.
- Slices on every test: per source, per domain (gate at macro-F1 ≥ 0.70), per length quintile, per geography,
  minimal pairs, claim-style only.
- Metrics: macro-F1, AUROC, MCC, ECE (before and after temperature scaling), Brier, AURC (selective prediction).
- **Ablations:** keep debunk rows / no length matching / full text instead of lead / backbones (XLM-R base, MuRIL, mBERT).
- **OOD:** AUROC and FPR@95TPR for Mahalanobis vs energy vs MSP. The OOD sets are far-OOD (Urdu Q&A, product reviews),
  script shift (Roman Urdu) and near-OOD (a whole unseen dataset).
- **Judge:** classifier-only vs judge-only (with coverage) vs fused system. Also scope accuracy on the manual claims, a
  meta-judge rubric (faithfulness / relevance / recency / clarity / calibration) and a Roman-Urdu round trip.

### Known limitations (put these in the report)
- Only `urdufake` has native domain labels. The other test domains are tagged by Gemini, so spot-check 50.
- The judge may retrieve the very fact-check article a test item came from. Judge accuracy on the 2017–23 test items is
  therefore optimistic. The manual recent claims are the honest measure.
- The topic/geography confound in HnB cannot be removed without new data.
- Roman Urdu is evaluated by machine round-trip plus the manual claims. There is no Roman-Urdu training corpus.
