import cv2
import numpy as np
import pytest

from fer import EMOTIONS, IMG_SIZE
from fer.face_detector import Face, FaceDetector
from fer.pipeline import EmotionRecognizer, align_and_crop


def test_crop_without_landmarks_is_a_plain_resize():
    gray = np.zeros((200, 200), np.uint8)
    gray[50:150, 50:150] = 255  # the "face" fills the box exactly
    crop = align_and_crop(gray, Face(50, 50, 100, 100, 1.0), margin=1.0)
    assert crop.shape == (IMG_SIZE, IMG_SIZE)
    assert crop[2:-2, 2:-2].min() > 200  # box interior only


def test_crop_levels_tilted_eyes():
    # Two bright "eyes" on a line tilted by 20 degrees around the box centre.
    gray = np.zeros((300, 300), np.uint8)
    angle = np.radians(20)
    cx, cy, r = 150, 150, 40
    right = (cx - r * np.cos(angle), cy - r * np.sin(angle))
    left = (cx + r * np.cos(angle), cy + r * np.sin(angle))
    for x, y in (right, left):
        cv2.circle(gray, (int(round(x)), int(round(y))), 6, 255, -1)
    face = Face(100, 100, 100, 100, 1.0, eyes=(right, left))
    crop = align_and_crop(gray, face, margin=1.0)
    ys, xs = np.nonzero(crop > 128)
    left_half, right_half = xs < IMG_SIZE / 2, xs >= IMG_SIZE / 2
    # after alignment both eyes sit on the same row
    assert abs(ys[left_half].mean() - ys[right_half].mean()) < 1.0


@pytest.mark.parametrize("backend", ["yunet", "haar"])
def test_detector_finds_nothing_in_blank_image(backend):
    detector = FaceDetector(backend=backend)
    assert detector.detect(np.full((240, 320, 3), 127, np.uint8)) == []


def test_recognizer_output_schema(tiny_checkpoint):
    rec = EmotionRecognizer(tiny_checkpoint)
    probs = rec.classify([np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)] * 3)
    assert probs.shape == (3, len(EMOTIONS))
    np.testing.assert_allclose(probs.sum(1), 1.0, rtol=1e-5)

    out = rec.analyze(np.zeros((120, 160, 3), np.uint8))
    assert out["faces"] == []
    assert out["image"] == {"width": 160, "height": 120}
    assert set(out["timing_ms"]) == {"detect", "classify", "total"}
