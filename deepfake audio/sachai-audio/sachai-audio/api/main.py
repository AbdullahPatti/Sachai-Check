"""Sachai Check audio API + frontend.

Run from the project folder:
    uvicorn api.main:app --host 127.0.0.1 --port 8000
then open http://127.0.0.1:8000

The model is loaded once at startup from SACHAI_MODEL (a local export folder or a Hugging Face repo id).
Nothing is retrained. Uploaded audio lives in a private temp folder and is deleted on every exit path.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from sachai_audio.explain import explain  # noqa: E402
from sachai_audio.infer import Detector  # noqa: E402

ALLOWED = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".flac", ".aac", ".mp4", ".oga", ".weba"}
STATE: dict = {"detector": None, "error": None}
LIMIT = asyncio.Semaphore(int(os.environ.get("SACHAI_CONCURRENCY", "1")))


def model_source() -> str:
    if os.environ.get("SACHAI_MODEL"):
        return os.environ["SACHAI_MODEL"]
    local = sorted((ROOT / "runs").glob("*/export/policy.json"))
    if local:
        return str(local[-1].parent)
    if os.environ.get("HF_REPO_ID"):
        return os.environ["HF_REPO_ID"]
    raise RuntimeError("No model found. Train and export first, or set SACHAI_MODEL / HF_REPO_ID in .env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["detector"] = await run_in_threadpool(Detector, model_source())
        d = STATE["detector"]
        metrics = json.loads((d.dir / "metrics.json").read_text()) if (d.dir / "metrics.json").exists() else {}
        STATE["n_train_speakers"] = len(metrics.get("splits", {}).get("train", [])) or None
        print(f"[sachai] model loaded from {d.dir} on {d.device}")
    except Exception as e:  # keep serving the page and report the problem on /health/ready
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[sachai] model failed to load: {STATE['error']}")
    yield


app = FastAPI(title="Sachai Check audio API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8000", "http://localhost:8000",
                                                  "http://127.0.0.1:5173", "http://localhost:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    d = STATE["detector"]
    if d is None:
        raise HTTPException(503, STATE["error"] or "model not loaded")
    return {"status": "ready", "device": str(d.device), "versions": d.version}


def caveats(res: dict) -> list[str]:
    out = ["This checks whether the voice is genuine, not whether what it says is true."]
    if res.get("reason") == "quality_gate":
        out.append("The recording quality was too low for a confident decision. Try a clearer clip of at least 3 seconds.")
    if (res.get("ood") or {}).get("flag"):
        out.append("This recording is unlike anything in the training data, so treat the result with extra caution.")
    out.append("Tested on Tacotron and VITS Urdu voices. Newer voice-cloning tools may not be detected.")
    return out


@app.post("/api/audio")
async def assess_audio(file: UploadFile = File(...), confirm_urdu: bool = Form(True), confirm_permission: bool = Form(True)):
    d = STATE["detector"]
    if d is None:
        raise HTTPException(503, STATE["error"] or "model not loaded")
    if not confirm_permission:
        raise HTTPException(422, "Please confirm you have permission to analyse this recording.")
    suffix = Path(file.filename or "").suffix.lower() or ".bin"
    if suffix not in ALLOWED and not (file.content_type or "").startswith("audio/"):
        raise HTTPException(415, "Unsupported file type. Use WAV, MP3, M4A, OGG/Opus or WebM.")
    if LIMIT.locked():
        raise HTTPException(429, "The detector is busy. Please retry in a few seconds.")

    request_id = uuid.uuid4().hex
    tmpdir = Path(tempfile.mkdtemp(prefix="sachai_"))
    dst = tmpdir / f"{request_id}{suffix if suffix in ALLOWED else '.bin'}"
    max_bytes = int(d.cfg.api_max_mb * 1024 * 1024)
    t0 = time.time()
    try:
        size = 0
        with open(dst, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"File is larger than {d.cfg.api_max_mb:.0f} MB.")
                f.write(chunk)
        if size == 0:
            raise HTTPException(422, "The file is empty.")
        async with LIMIT:
            try:
                y, meta = await asyncio.wait_for(run_in_threadpool(d.decode_file, dst), timeout=30)
            except asyncio.TimeoutError:
                raise HTTPException(504, "Decoding timed out.")
            except Exception:
                raise HTTPException(415, "This audio could not be decoded. Try WAV or MP3.")
            dur = meta["decoded_sec"]
            if dur < d.cfg.api_min_sec:
                raise HTTPException(422, f"Clip is {dur:.1f} s long. Please upload at least {d.cfg.api_min_sec:.0f} s of speech.")
            if dur > d.cfg.api_max_sec:
                raise HTTPException(422, f"Clip is {dur:.0f} s long. The limit is {d.cfg.api_max_sec:.0f} s.")
            try:
                res = await asyncio.wait_for(run_in_threadpool(d.analyze_array, y, meta), timeout=90)
            except asyncio.TimeoutError:
                raise HTTPException(504, "Analysis timed out.")
        try:
            expl = await asyncio.wait_for(run_in_threadpool(explain, res, STATE.get("n_train_speakers")), timeout=45)
        except asyncio.TimeoutError:
            expl = explain(res, STATE.get("n_train_speakers"), use_llm=False)
        res.update(request_id=request_id, modality="audio", explanation=expl, caveats=caveats(res),
                   assessed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), confirm_urdu=confirm_urdu,
                   total_ms=int((time.time() - t0) * 1000))
        return res
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


FRONTEND = ROOT / "frontend"
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
