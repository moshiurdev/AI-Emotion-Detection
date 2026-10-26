"""Build a multi-face test image from FER-2013 PrivateTest faces.

Useful for trying the web demo without a webcam ("Upload Image"), or for
checking multi-face handling. One face per emotion is upscaled and placed on
a grey canvas. The output lives under ``data/`` (git-ignored) because the
FER-2013 images should not be redistributed in the repository.

    python scripts/make_demo_image.py --out data/demo/faces.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fer import EMOTIONS  # noqa: E402
from fer.data import load_split  # noqa: E402
from fer.face_detector import FaceDetector  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/demo/faces.jpg"))
    p.add_argument("--size", type=int, default=176, help="face size in pixels")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    images, labels = load_split("test")
    rng = np.random.default_rng(args.seed)
    detector = FaceDetector("yunet")
    gap, s = args.size // 3, args.size
    cols, rows = 4, 2
    canvas = np.full((rows * (s + gap) + gap, cols * (s + gap) + gap), 128, np.uint8)

    chosen = []
    for k in range(len(EMOTIONS)):
        # pick a random face of this class that the detector can find on its own
        for i in rng.permutation(np.flatnonzero(labels == k)):
            face = cv2.resize(images[i], (s, s), interpolation=cv2.INTER_CUBIC)
            probe = cv2.copyMakeBorder(face, gap, gap, gap, gap, cv2.BORDER_REPLICATE)
            if len(detector.detect(cv2.cvtColor(probe, cv2.COLOR_GRAY2BGR))) == 1:
                chosen.append((i, face))
                break

    for n, (i, face) in enumerate(chosen):
        r, c = divmod(n, cols)
        y, x = gap + r * (s + gap), gap + c * (s + gap)
        canvas[y:y + s, x:x + s] = face

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), canvas)
    order = ", ".join(f"{EMOTIONS[labels[i]]} (#{i})" for i, _ in chosen)
    print(f"wrote {args.out}; ground truth left-to-right, top-to-bottom: {order}")


if __name__ == "__main__":
    main()
