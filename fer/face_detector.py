"""Multi-face detection with OpenCV.

The default backend is YuNet (OpenCV Zoo, ~230 KB ONNX, also returns five
facial landmarks that we use to level the eyes). If the ONNX file is missing
the detector falls back to the Haar cascade bundled with ``opencv-python``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_YUNET = Path(__file__).resolve().parents[1] / "models" / "face_detection_yunet_2023mar.onnx"


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float
    # (right_eye, left_eye) in image coordinates, if the backend provides them
    eyes: tuple[tuple[float, float], tuple[float, float]] | None = field(default=None, repr=False)


class FaceDetector:
    def __init__(self, backend: str = "auto", yunet_path: Path = DEFAULT_YUNET,
                 score_threshold: float = 0.7, min_size: int = 40):
        if backend == "auto":
            backend = "yunet" if Path(yunet_path).exists() else "haar"
        self.backend = backend
        self.min_size = min_size
        if backend == "yunet":
            self._yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320),
                                                    score_threshold=score_threshold,
                                                    nms_threshold=0.3, top_k=50)
        elif backend == "haar":
            cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(str(cascade))
        else:
            raise ValueError(f"unknown detector backend: {backend}")

    def detect(self, bgr: np.ndarray) -> list[Face]:
        faces = self._detect_yunet(bgr) if self.backend == "yunet" else self._detect_haar(bgr)
        faces = [f for f in faces if min(f.w, f.h) >= self.min_size]
        return sorted(faces, key=lambda f: f.x)  # left-to-right so "Face 1" is stable

    def _detect_yunet(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        self._yunet.setInputSize((w, h))
        _, out = self._yunet.detect(bgr)
        faces = []
        for row in [] if out is None else out:
            x, y, fw, fh = row[:4]
            faces.append(Face(int(round(x)), int(round(y)), int(round(fw)), int(round(fh)),
                              float(row[14]), eyes=((row[4], row[5]), (row[6], row[7]))))
        return faces

    def _detect_haar(self, bgr: np.ndarray) -> list[Face]:
        gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        boxes = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                            minSize=(self.min_size, self.min_size))
        return [Face(int(x), int(y), int(w), int(h), 1.0) for x, y, w, h in boxes]
