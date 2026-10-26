import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fer.models import build_model  # noqa: E402
from fer.utils import save_checkpoint  # noqa: E402


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A randomly initialised small_cnn checkpoint, so tests never need trained weights."""
    torch.manual_seed(0)
    model = build_model("small_cnn", pretrained=False)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    save_checkpoint(path, model, {"arch": "small_cnn", "input_size": 48, "dropout": 0.4, "params": 0})
    return path
