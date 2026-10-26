"""FER-2013 dataset, augmentation and class weighting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from . import IMG_SIZE, NUM_CLASSES

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fer2013"


def load_split(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images uint8 [N,48,48], labels int64 [N])`` for train/val/test."""
    path = Path(data_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python scripts/download_data.py` first")
    with np.load(path) as f:
        return f["images"], f["labels"]


def train_transform() -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToDtype(torch.float32, scale=True),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),
    ])


def eval_transform() -> T.Compose:
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


class FER2013(Dataset):
    def __init__(self, split: str, transform=None, data_dir: Path = DEFAULT_DATA_DIR):
        self.images, self.labels = load_split(split, data_dir)
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # (48, 48) -> (48, 48, 1) so ToImage yields a 1x48x48 tensor
        img = self.transform(self.images[idx][:, :, None])
        return img, int(self.labels[idx])


def class_weights(labels: np.ndarray, mode: str = "sqrt") -> torch.Tensor:
    """Loss weights that counter FER-2013's class imbalance.

    ``inverse`` is the classic ``N / (K * n_c)``; ``sqrt`` takes its square
    root, which still boosts rare classes (Disgust has 16x fewer samples than
    Happy) without letting them dominate the loss. Weights are normalised to
    mean 1 so the learning rate keeps the same meaning.
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    if mode == "none":
        w = np.ones(NUM_CLASSES)
    elif mode == "inverse":
        w = counts.sum() / (NUM_CLASSES * counts)
    elif mode == "sqrt":
        w = np.sqrt(counts.sum() / (NUM_CLASSES * counts))
    else:
        raise ValueError(f"unknown class weighting mode: {mode}")
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)
