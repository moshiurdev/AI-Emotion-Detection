"""End-to-end inference: image -> faces -> emotion probabilities."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import EMOTIONS, IMG_SIZE
from .face_detector import Face, FaceDetector
from .utils import load_checkpoint

# Crop side as a multiple of the detector box side. FER-2013 faces are tightly
# cropped, so the best framing sits slightly inside the YuNet box. Chosen on
# the val split with scripts/pipeline_eval.py (0.85: 67.4%, 0.90: 67.3%,
# 1.00: 66.4%, 1.25: 59.6%); 0.90 keeps a little more context for real frames.
CROP_MARGIN = 0.9


def align_and_crop(gray: np.ndarray, face: Face, margin: float = CROP_MARGIN) -> np.ndarray:
    """Cut a 48x48 face from ``gray``, rotated so the eyes are level.

    A single affine warp handles rotation (from the eye landmarks, if any),
    scaling and translation, so the crop is resampled only once.
    """
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    angle = 0.0
    if face.eyes is not None:
        (rx, ry), (lx, ly) = face.eyes
        angle = math.degrees(math.atan2(ly - ry, lx - rx))
    side = max(face.w, face.h) * margin
    m = cv2.getRotationMatrix2D((cx, cy), angle, IMG_SIZE / side)
    m[0, 2] += IMG_SIZE / 2 - cx
    m[1, 2] += IMG_SIZE / 2 - cy
    return cv2.warpAffine(gray, m, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


class EmotionRecognizer:
    def __init__(self, checkpoint: Path, detector: FaceDetector | None = None,
                 device: str = "cpu", tta: bool = True):
        self.device = torch.device(device)
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.config = ckpt["config"]
        self.detector = detector or FaceDetector()
        self.tta = tta

    @torch.no_grad()
    def classify(self, crops: list[np.ndarray]) -> np.ndarray:
        """``crops``: list of 48x48 uint8 faces -> ``(N, 7)`` probabilities."""
        if not crops:
            return np.zeros((0, len(EMOTIONS)), dtype=np.float32)
        x = torch.from_numpy(np.stack(crops)).float().div_(255).unsqueeze(1).to(self.device)
        p = self.model(x).softmax(1)
        if self.tta:
            p = (p + self.model(x.flip(-1)).softmax(1)) / 2
        return p.cpu().numpy()

    def analyze(self, bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        faces = self.detector.detect(bgr)
        t1 = time.perf_counter()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        probs = self.classify([align_and_crop(gray, f) for f in faces])
        t2 = time.perf_counter()

        results = []
        for i, (face, p) in enumerate(zip(faces, probs)):
            k = int(p.argmax())
            results.append({
                "id": i + 1,
                "box": {"x": face.x, "y": face.y, "w": face.w, "h": face.h},
                "detector_score": round(face.score, 4),
                "emotion": EMOTIONS[k],
                "confidence": round(float(p[k]), 4),
                "probabilities": {e: round(float(v), 4) for e, v in zip(EMOTIONS, p)},
            })
        return {
            "faces": results,
            "image": {"width": int(bgr.shape[1]), "height": int(bgr.shape[0])},
            "timing_ms": {
                "detect": round((t1 - t0) * 1000, 1),
                "classify": round((t2 - t1) * 1000, 1),
                "total": round((t2 - t0) * 1000, 1),
            },
        }
