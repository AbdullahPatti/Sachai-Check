# CLAUDE.md · Sachai Check audio deepfake module

Read this whole file before running anything. It tells you what the project is, the decisions already made,
the exact order to run things, how to check each step, and what never to change.

## 1. What this project is

Sachai Check is a final year project at FAST-NUCES Lahore (team: Muhammad Abdullah Haroon, Ikhlas Ahmad,
Zayan Amjad; supervisor Ms. Sukhan Amir). It has two independent pipelines: Urdu fake-news verification (not in
this repo) and **Urdu audio deepfake detection (this repo)**.

It is applied research. The end goal is the **product** (a web app that tells a user whether an Urdu voice clip is
genuine or synthetic, and why). A paper (ACL/ACM style) comes out of the same work, so every number must be honest
and reproducible.

The audio module:
- Detector: `facebook/wav2vec2-xls-r-300m` fine-tuned end to end (CNN encoder frozen), softmax weights over all
  25 hidden layers (SLS idea), attentive statistics pooling, 2-layer MLP.
- Data: Hugging Face `CSALT/deepfake_detection_dataset_urdu` (6,794 clips, 17 speakers: 3,398 genuine in
  `Bonafide/Speaker_XX/Part 1|2`, 1,698 VITS in `Spoofed_TTS`, 1,698 Tacotron in `Spoofed_Tacotron`).
  Part 2 genuine files share sentence IDs with the spoofs of the same speaker.
- Split: by speaker, 12 train / 2 dev / 3 test, frozen in `data/splits.json`.
- Decision: calibrated three-way output `bona_fide | synthetic | inconclusive`.
- Explanation: numbered measured facts, LLM justifier, rule checks, second-model LLM judge, template fallback.
  The LLM never changes the verdict.
- Product: FastAPI (`api/main.py`) serving the Pigeon, Almond & Lime frontend (`frontend/index.html`).

## 2. Decisions already made by the user (do not change without asking)

1. **One pipeline only.** No baseline comparisons, no AASIST/RawNet2 runs, no multiple seeds, no hyperparameter
   sweeps. Do not add them.
2. Research and product both matter. Never trade honesty of metrics for a nicer number.
3. Training must survive power cuts and resume where it stopped (already implemented, see section 7).
4. The model must stay alive: train once, export, push to the Hugging Face Hub, and load it afterwards.
   Never retrain just to use the model.
5. Notebook stays compact: small `###` headings, no long explanations, no literature review in it.
6. The FYP report may be updated to match the pipeline. LaTeX-ready edits live in `REPORT_CHANGES.md`.
7. HF token lives in `.env` as `HF_TOKEN`. Never print it, log it, or commit `.env`.

## 3. Machine

Windows PC: Intel Core i7 13th gen, 32 GB RAM, NVIDIA RTX A4000 (16 GB, Ampere, supports bf16).
Config defaults already match it: `batch_size=16`, `grad_accum=2` (effective batch 32), `num_workers=4`, bf16,
gradient checkpointing on. Expect peak GPU memory around 9 to 12 GB.

Claude Code on Windows runs commands in Git Bash. Always call the venv interpreter explicitly:
`.venv/Scripts/python` (not a bare `python`, which may be the system one).

## 4. One-time setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env        # the user fills HF_TOKEN (and optionally LLM_API_KEY)
.venv/Scripts/python run.py check
```

`run.py check` must show `CUDA available: True`, the A4000 with about 16 GB and `bf16: True`, an ffmpeg path with
`libmp3lame` and `libopus` in the encoder list, and at least 15 GB free disk. If CUDA is False: reinstall torch from
the cu132 index above and update the NVIDIA driver. Torch 2.15 publishes only CUDA 13.x wheels, which need a recent
driver.

`.env` keys (see `.env.example`):
| key | needed for | notes |
|---|---|---|
| `HF_TOKEN` | push to Hub | Write token. Dataset is public, so download works without it |
| `HF_REPO_ID` | push | optional, default `<username>/sachai-check-urdu-audio-deepfake` |
| `HF_PRIVATE` | push | `true` by default |
| `LLM_API_KEY` | LLM explanation | optional; without it the template explanation is used |
| `LLM_BASE_URL`, `LLM_MODEL`, `JUDGE_MODEL` | LLM explanation | defaults: Groq, `llama-3.3-70b-versatile`, judge `openai/gpt-oss-120b` |
| `SACHAI_MODEL` | API | optional; export folder or HF repo id |

## 5. Run order

Run stages in this order. Every stage is idempotent: re-running skips finished work and resumes interrupted work.
The notebook `sachai_audio_pipeline.ipynb` runs the same steps for the human; prefer `run.py` in the terminal.

| # | Command | Time on A4000 | Done when | Main outputs |
|---|---|---|---|---|
| 0 | `run.py check` | seconds | GPU, ffmpeg, disk look right | none |
| 1 | `run.py data` | download 2 GB + 3 to 6 min cache | split counts printed | `data/raw/`, `data/cache_16k/`, `data/manifest.csv`, `data/splits.json` |
| 2 | `run.py audit` | seconds | table + leakage AUCs printed | `report/data_audit.csv`, `report/data_audit.json` |
| 3 | `run.py train` | about 40 to 70 min | `runs/<run>/train_done.json` exists | `checkpoints/`, `best/`, `history.csv` |
| 4 | `run.py fit` | seconds | diagnosis line printed | `report/fig_learning_curves.png/pdf` |
| 5 | `run.py finalize` | 5 to 10 min | `export/policy.json` exists | `export/` (model + policy + OOD + reference stats) |
| 6 | `run.py test` | 1 to 2 min | tables printed | `table_main.*`, `test_per_attack.csv`, `test_per_speaker.csv`, figures |
| 7 | `run.py robustness` | 10 to 20 min | 10 conditions printed | `table_robustness.*`, `robustness/*.csv`, figure |
| 8 | `run.py analysis` | 3 to 5 min | descriptor table printed | `fig_layer_weights.*`, `table_descriptors.*` |
| 9 | `run.py demo` | under 1 min | two clips + latency printed | none |
| 10 | `run.py push` | upload 1.3 GB | prints the Hub URL | model repo on the Hub |
| 11 | serve | always on | page loads, audio tab works | see section 9 |

`run.py all` runs 1 to 9 in one go. Ask the user before the first `push` (it publishes to an external service).

Paths: `<run>` is `runs/xlsr300m-sls-v1` unless `--run-name` is given. `report/` means `<run>/report/`.

Options: `--batch-size`, `--grad-accum`, `--workers`, `--max-epochs`, `--run-name`, `--force` (redo a finished
data or finalize stage), `--force-retrain` (train from scratch, overwrites the run; ask the user first).

## 6. Running long jobs from Claude Code

Training and `all` take longer than a single command timeout. Run them in the background with unbuffered output
to a log, then poll:

```bash
mkdir -p logs
.venv/Scripts/python -u run.py train > logs/train.log 2>&1        # run with run_in_background
tail -n 5 logs/train.log                                           # progress bar + epoch summaries
cat runs/xlsr300m-sls-v1/history.csv                               # one row per finished epoch
ls runs/xlsr300m-sls-v1/train_done.json                            # exists = training finished
nvidia-smi                                                         # GPU memory and utilisation
```

Each epoch prints: augmented train loss, clean train-probe EER and loss, dev EER, AUC and loss, and `*best*` when
the dev result improves. Do not start a second training process on the same run while one is alive.

## 7. Fault tolerance and resume (already implemented, keep it working)

- Full checkpoint `checkpoints/last.pt` (weights, optimizer, scheduler, scaler, sampler position, RNG state,
  history, early-stopping state) every 8 minutes and at every epoch end.
- Writes are atomic: temp file, fsync, rename. The previous one is kept as `last_prev.pt`. A truncated file from a
  power cut is detected and the older checkpoint is used automatically.
- Batches are a pure function of (seed, epoch, batch index), so resume continues at the exact batch.
- On resume, the run keeps its original training settings (batch size, accumulation, LRs, epochs, seed and others
  in `LOCKED_ON_RESUME`), even if the command line differs. To change them, use a new `--run-name`.
- Ctrl+C saves before exiting. After a power cut or crash, simply run `run.py train` (or `all`) again.
- Data cache, test scores and every robustness condition are cached separately, so they resume too.
- Each checkpoint is about 3.8 GB (two are kept). Do not delete `checkpoints/` while training is unfinished.

## 8. What good results look like, and red flags

Do not invent or predict numbers. Read them from `report/`. Use these checks:

| Check | Where | Red flag | What to do |
|---|---|---|---|
| Leakage | `data_audit.json` `metadata_auc_after_trim` | above about 0.75 | report it; the model may use duration; tell the user |
| Underfitting | `run.py fit` | train-probe EER above 5% at the best epoch | more epochs or higher `lr_backbone`, as a new run, after asking |
| Overfitting | `run.py fit` | dev EER much higher than train-probe EER (gap above 5 points) | more augmentation or dropout, as a new run, after asking |
| Calibration | `export/policy.json` | temperature stuck at 0.2 or 10 (the bounds) | mention it; dev may be too easy or too noisy |
| Coverage | `table_main.csv` | test coverage far below dev coverage | speaker shift; report honestly |
| Robustness | `table_robustness.csv` | large EER rise for MP3/Opus | expected limitation (codecs unseen in training); report, do not tune on test |

Dev EER near 0 is plausible for in-domain Tacotron/VITS. It is not proof of real-world accuracy: modern voice
cloning was never in training. That limit must stay visible in the product and the paper.

## 9. The product (API + frontend)

```bash
.venv/Scripts/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000      # run in background
curl http://127.0.0.1:8000/health/ready
curl -F "file=@some_clip.wav" -F "confirm_permission=true" http://127.0.0.1:8000/api/audio
```
Open http://127.0.0.1:8000 and use the Audio Deepfake tab.

- Model source: `SACHAI_MODEL`, else newest `runs/*/export`, else `HF_REPO_ID`. Loaded once at start-up.
- `POST /api/audio`: multipart `file` (wav, mp3, m4a, ogg/opus, webm, flac), 3 to 60 s, max 20 MB.
  Errors: 413 too large, 415 unsupported or undecodable, 422 empty/short/long, 429 busy, 503 no model, 504 timeout.
- Response: `verdict`, `label_en`, `label_ur`, `spoof_score`, `policy` (low/high/temperature), `segments`
  (2 s windows with `flagged`), `quality`, `ood`, `descriptors`, `explanation` (`bullets`, `summary_ur`, `source`,
  `judge`, `note`), `caveats`, `versions`, `trim_offset_sec`, `total_ms`.
- Uploads go to a private temp folder and are deleted in a `finally` block. Nothing is stored or logged.
- The news tab in the frontend is still demo data; only the audio tab is live.

## 10. Code map

| File | Responsibility |
|---|---|
| `run.py` | CLI for every stage (section 5) |
| `sachai_audio/config.py` | all settings, derived paths, `PREPROCESS_VERSION`, `POLICY_VERSION` |
| `sachai_audio/audio_io.py` | decode (soundfile, ffmpeg via imageio-ffmpeg), resample to 16 kHz, trim, normalise, windows, codec round trip, quality |
| `sachai_audio/augment.py` | label-blind training augmentation (noise, reverb, band-limit, mu-law, EQ, impulsive) |
| `sachai_audio/data.py` | download, manifest, frozen speaker split, int16 cache, resumable sampler, datasets, shortcut audit |
| `sachai_audio/model.py` | `SachaiDetector` (XLS-R + layer weights + ASP + MLP), save/load safetensors |
| `sachai_audio/train.py` | resumable training loop, checkpoints, window scoring (`score_clips`) |
| `sachai_audio/metrics.py` | EER, AUC, temperature scaling, ECE, decision policy, grouped bootstrap |
| `sachai_audio/evaluate.py` | finalize/export, test, robustness, descriptor analysis, LaTeX tables |
| `sachai_audio/evidence.py` | 7 acoustic descriptors, reference distributions, comparison |
| `sachai_audio/infer.py` | `Detector`: product inference on any file |
| `sachai_audio/explain.py` | facts, LLM justifier, rule checks, LLM judge, template fallback |
| `sachai_audio/hub.py` | model card + push to Hub |
| `sachai_audio/plots.py` | paper figures (genuine blue `#2a78d6`, synthetic orange `#eb6834`) |
| `api/main.py` | FastAPI app, limits, temp-file cleanup, serves `frontend/` |
| `frontend/index.html` | single-file UI; audio page calls `/api/audio` |
| `REPORT_CHANGES.md` | LaTeX edits for the FYP report |

## 11. Invariants (never break these)

1. `data/splits.json` is written once and never regenerated. Do not delete it. Train, dev and test speakers never mix.
2. Everything fitted (temperature, thresholds, OOD threshold, reference stats) uses train/dev speakers only.
3. The test set is scored once with the frozen policy. Never tune anything on test or robustness results, and never
   retrain and keep whichever run tests best.
4. Training and inference use the same preprocessing (`audio_io.py`). If preprocessing changes, bump
   `PREPROCESS_VERSION`, use a new run name, and redo data cache, training and export.
5. The explanation layer may only use numbered facts and can never change `verdict`.
6. Synthetic is the positive class (label 1) everywhere. Scores are spoof probabilities.
7. Uploaded audio is never kept, logged or used for training.
8. Keep Windows compatibility: `if __name__ == "__main__":` in scripts, datasets defined in modules (not notebooks),
   `pathlib` paths, no shell-specific code inside the package.

## 12. Changing things safely

- Any change to model or training settings: new `--run-name`, then `train`, `fit`, `finalize`, `test`,
  `robustness`, `analysis`. Old runs stay for comparison only if the user asks for it.
- `finalize` fingerprints `best/model.safetensors`; if the weights changed it re-exports automatically, and every
  cached score file (`dev_scores.csv`, `test_scores.csv`, `robustness/*.csv`) carries a `model_id`, so stale scores
  are recomputed instead of reused. An old `export/` can never pair with new weights.
- Data stage with `--force` rebuilds the manifest and cache but keeps `splits.json`.
- Clear a cached evaluation by deleting its CSV in `report/` (for example `report/robustness/mp3_32kbps.csv`).
- After code edits, run a quick syntax check:
  `.venv/Scripts/python -c "import ast,glob;[ast.parse(open(f).read()) for f in glob.glob('sachai_audio/*.py')+['run.py','api/main.py']]"`

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA available: False` | reinstall torch from the cu132 index; update NVIDIA driver |
| CUDA out of memory | `--batch-size 8 --grad-accum 4` on a **new** run name (resume locks the old settings) |
| DataLoader hangs or worker errors on Windows | `--workers 0` (new run name if training already started) |
| Download stalls or 429 from Hub | re-run `run.py data`; set `HF_TOKEN` for higher rate limits |
| `data/manifest.csv missing` | run `run.py data` first |
| Robustness has no MP3/Opus rows | ffmpeg build lacks the encoder; `run.py check` shows which; reinstall `imageio-ffmpeg` |
| Explanation `source` is `template` | no `LLM_API_KEY`, provider error, or judge rejected bullets; check `llm_error` / `judge` in the response |
| API 503 on `/health/ready` | no export found; run `finalize`, or set `SACHAI_MODEL` / `HF_REPO_ID` |
| `push` 401/403 | `HF_TOKEN` missing or read-only; needs a Write token |
| Long-path errors on Windows | move the project to a short path such as `C:\sachai` |

## 14. Report and paper

- `REPORT_CHANGES.md` holds section-by-section LaTeX replacements. Numbers marked `\todo{}` must be filled only
  from `report/table_main.csv`, `test_per_attack.csv`, `table_robustness.csv`, `table_descriptors.csv` and
  `metrics_test.json`.
- Paper-ready assets: `report/table_*.tex` and `report/fig_*.pdf`.
- Cite the dataset size as 6,794 clips (HF release), noting the paper reports 37,281.

## 15. Status

Done and tested on synthetic data with a tiny model: every stage, kill -9 resume, corrupted-checkpoint fallback,
API with WAV/MP3/WebM uploads and error cases, LLM judge rejection path, frontend rendering.
Not yet done: the real run on the A4000 (stages 1 to 10). The text pipeline is a separate codebase.

## 16. Writing style for anything the user will read

Plain, direct sentences of varied length. No em dashes, no filler, no corporate buzzwords. Notebook markdown stays
one line per section. End summaries with a clear verdict.
