"""Measure the full detect -> align -> classify pipeline on FER-2013 faces.

``evaluate.py`` scores the classifier on the dataset's own 48x48 crops. In
the web app, faces are instead cropped from YuNet boxes, so the framing
differs. This script re-creates that situation: each FER image is upscaled,
padded, run through the detector and re-cropped exactly as the app does.

It also sweeps the crop margin (box side multiplier). Tune it on ``val``,
then report ``test`` with the chosen value:

    python scripts/pipeline_eval.py --checkpoint checkpoints/fer_resnet18.pt --split val
    python scripts/pipeline_eval.py --checkpoint checkpoints/fer_resnet18.pt --split test --margins 1.25 \
        --out results/resnet18/pipeline_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fer.data import load_split  # noqa: E402
from fer.face_detector import FaceDetector  # noqa: E402
from fer.pipeline import EmotionRecognizer, align_and_crop  # noqa: E402


def classify_all(rec: EmotionRecognizer, crops: list[np.ndarray], batch: int = 256) -> np.ndarray:
    return np.concatenate([rec.classify(crops[i:i + batch]) for i in range(0, len(crops), batch)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--margins", type=float, nargs="+", default=[1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5])
    p.add_argument("--upscale", type=int, default=4, help="48 px -> 192 px, a typical webcam face size")
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    images, labels = load_split(args.split)
    if args.limit:
        images, labels = images[:args.limit], labels[:args.limit]
    rec = EmotionRecognizer(args.checkpoint, detector=FaceDetector("yunet"), device=args.device)

    scenes, faces, keep = [], [], []
    for i, img in enumerate(images):
        big = cv2.resize(img, None, fx=args.upscale, fy=args.upscale, interpolation=cv2.INTER_CUBIC)
        scene = cv2.copyMakeBorder(big, args.pad, args.pad, args.pad, args.pad, cv2.BORDER_REPLICATE)
        found = rec.detector.detect(cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR))
        if found:
            scenes.append(scene)
            faces.append(max(found, key=lambda f: f.score))
            keep.append(i)
    keep = np.array(keep)
    y = labels[keep]

    direct = classify_all(rec, list(images[keep])).argmax(1)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "images": int(len(labels)),
        "detected": int(len(keep)),
        "detection_rate": float(len(keep) / len(labels)),
        "acc_dataset_crops_on_detected": float((direct == y).mean()),
        "acc_pipeline_by_margin": {},
    }
    print(f"{args.split}: detected {len(keep)}/{len(labels)} ({result['detection_rate']:.1%}); "
          f"dataset-crop acc on detected subset {result['acc_dataset_crops_on_detected']:.4f}")
    for m in args.margins:
        crops = [align_and_crop(s, f, margin=m) for s, f in zip(scenes, faces)]
        acc = float((classify_all(rec, crops).argmax(1) == y).mean())
        result["acc_pipeline_by_margin"][str(m)] = acc
        print(f"  margin {m:.2f}: pipeline acc {acc:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
