"""FastAPI server for the webcam demo.

Run:
    uvicorn app.server:app --port 8000
    # or: python -m app.server --checkpoint checkpoints/fer_resnet18.pt

The browser grabs webcam frames, POSTs them as JPEG to ``/api/analyze`` and
draws the returned boxes. The server keeps no per-user state; session
statistics live in the page.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fer import EMOTIONS
from fer.pipeline import EmotionRecognizer

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "fer_resnet18.pt"
DEFAULT_METRICS = ROOT / "results" / "resnet18" / "metrics.json"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_recognizer()  # load the model before the first request
    yield


app = FastAPI(title="AI Emotion Recognition", version="2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache(maxsize=1)
def get_recognizer() -> EmotionRecognizer:
    checkpoint = Path(os.environ.get("FER_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint.exists():
        raise RuntimeError(f"checkpoint not found: {checkpoint} (see README: Training)")
    return EmotionRecognizer(checkpoint, device=os.environ.get("FER_DEVICE", "cpu"))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info() -> dict:
    rec = get_recognizer()
    out = {
        "emotions": EMOTIONS,
        "model": {"arch": rec.config["arch"], "input_size": rec.config["input_size"],
                  "params": rec.config.get("params"), "tta": rec.tta},
        "detector": rec.detector.backend,
    }
    metrics_path = Path(os.environ.get("FER_METRICS", DEFAULT_METRICS))
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())["tta_hflip"]
        out["test_metrics"] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    return out


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> dict:
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large (max 8 MB)")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    return get_recognizer().analyze(bgr)


if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser(description="Serve the emotion recognition web demo")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    os.environ["FER_CHECKPOINT"] = str(args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)
