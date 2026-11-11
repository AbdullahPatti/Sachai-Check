# Sachai Check · Audio deepfake module

Urdu speech in, verdict out: genuine, synthetic or inconclusive, with segment scores and an explanation
built only from the detector's measurements.

## Setup (Windows, Python 3.11 or 3.12)
Claude Code: read `CLAUDE.md` first.

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
copy .env.example .env      # then put your HF_TOKEN (and LLM_API_KEY) in .env
```

## Run
1. Open `sachai_audio_pipeline.ipynb` and run all cells, or run `python run.py all` then `python run.py push`.
   First run on the RTX A4000 PC: download (~2 GB), cache, train (roughly 40 to 70 minutes), export, test,
   robustness, push to Hugging Face. Defaults are set for the A4000; on an 8 GB GPU use
   `--batch-size 8 --grad-accum 4`.
2. Serve the product: `uvicorn api.main:app --host 127.0.0.1 --port 8000`, open http://127.0.0.1:8000.

Power cut, crash or Ctrl+C: run the training cell again. It resumes from the exact batch of the last
checkpoint (written every 8 minutes and every epoch, atomically, previous one kept as a fallback).
Once training is done, every later run loads the saved model; nothing is retrained.

## Pipeline
| Step | What happens |
|---|---|
| Data | 16 kHz mono, leading/trailing silence trimmed for both classes, frozen speaker-disjoint split (12 train / 2 dev / 3 test speakers) |
| Model | XLS-R 300M fine-tuned, softmax weights over all 25 hidden layers, attentive statistics pooling, MLP |
| Training | random 2 to 4 s crops, label-blind channel augmentation (noise, reverb, band-limit, mu-law, EQ, impulses), bf16, early stopping on dev EER |
| Decision | 4 s windows, mean log-odds, temperature scaling, three-way band fitted on dev at 99% precision |
| Product extras | 2 s segment timeline, quality gate, out-of-distribution flag, acoustic descriptors vs training reference |
| Explanation | numbered facts -> LLM justifier -> number/claim rule checks -> second-model judge -> template fallback |

## Layout
```
sachai_audio/   config, audio_io, augment, data, model, train, metrics, evaluate, evidence, infer, explain, hub, plots
api/main.py     FastAPI: POST /api/audio, GET /health/live, GET /health/ready, serves frontend/
frontend/       Pigeon, Almond & Lime UI, audio page wired to the API
runs/<run>/     checkpoints/, best/, export/ (what the API and the Hub use), report/ (tables .csv/.tex, figures .png/.pdf)
```

## Environment variables
`HF_TOKEN`, `HF_REPO_ID`, `HF_PRIVATE`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `JUDGE_MODEL`, `SACHAI_MODEL` (see `.env.example`).

Data: CSALT Urdu deepfake dataset, CC BY-NC 4.0. The model inherits the non-commercial condition.
