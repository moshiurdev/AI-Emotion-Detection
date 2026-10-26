"""Download FER-2013 and the YuNet face detector.

FER-2013 is fetched from a Hugging Face mirror that keeps the three original
splits (Training / PublicTest / PrivateTest) as parquet files. The images are
decoded once and cached as a compressed ``.npz`` per split so that training
does not pay the PNG-decoding cost every epoch.

Usage:
    python scripts/download_data.py            # dataset + face detector
    python scripts/download_data.py --skip-data  # face detector only
"""

from __future__ import annotations

import argparse
import io
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "fer2013"
MODELS_DIR = ROOT / "models"

HF_REPO = "Aaryan333/fer2013_train_publicTest_privateTest"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data"
# Mirror split name -> (parquet file, our split name)
SPLITS = {
    "train": ("train-00000-of-00001-5eab84e1c6a2fc27.parquet", "train"),
    "publicTest": ("publicTest-00000-of-00001-f41bb7384b8aad6e.parquet", "val"),
    "privateTest": ("privateTest-00000-of-00001-4b8a0715cf1b7560.parquet", "test"),
}

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  exists: {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    # python.org builds on macOS ship without a CA bundle; prefer certifi's.
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(url, context=context) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.rename(dest)


def parquet_to_npz(parquet: Path, out: Path) -> None:
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet)
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    images = np.empty((len(labels), 48, 48), dtype=np.uint8)
    for i, cell in enumerate(table.column("image").to_pylist()):
        img = Image.open(io.BytesIO(cell["bytes"])).convert("L")
        if img.size != (48, 48):
            raise ValueError(f"unexpected image size {img.size} at row {i}")
        images[i] = np.asarray(img)
    np.savez_compressed(out, images=images, labels=labels)
    print(f"  wrote {out.relative_to(ROOT)}: {len(labels)} images, "
          f"class counts {np.bincount(labels, minlength=7).tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-data", action="store_true", help="only fetch the face detector")
    args = parser.parse_args()

    print("Face detector (YuNet, OpenCV Zoo):")
    fetch(YUNET_URL, MODELS_DIR / "face_detection_yunet_2023mar.onnx")

    if args.skip_data:
        return 0

    print(f"FER-2013 (hf://{HF_REPO}):")
    for mirror_name, (fname, split) in SPLITS.items():
        parquet = DATA_DIR / "raw" / fname
        fetch(f"{HF_BASE}/{fname}", parquet)
        out = DATA_DIR / f"{split}.npz"
        if out.exists():
            print(f"  exists: {out.relative_to(ROOT)}")
        else:
            parquet_to_npz(parquet, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
