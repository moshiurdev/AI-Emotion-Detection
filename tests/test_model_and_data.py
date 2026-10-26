import numpy as np
import pytest
import torch

from fer import EMOTIONS, IMG_SIZE
from fer.data import class_weights, eval_transform, train_transform
from fer.models import build_model
from fer.utils import load_checkpoint


@pytest.mark.parametrize("arch,size", [("small_cnn", 48), ("resnet18", 96), ("mobilenet_v3_small", 96)])
def test_models_map_48px_gray_to_7_logits(arch, size):
    model = build_model(arch, input_size=size, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 1, IMG_SIZE, IMG_SIZE))
    assert out.shape == (2, len(EMOTIONS))


def test_class_weights_boost_rare_classes():
    labels = np.array([0] * 100 + [1] * 10 + [2] * 100 + [3] * 400 + [4] * 100 + [5] * 100 + [6] * 100)
    for mode in ("sqrt", "inverse"):
        w = class_weights(labels, mode)
        assert w.mean().item() == pytest.approx(1.0)
        assert w.argmax().item() == 1 and w.argmin().item() == 3
    # sqrt weighting is a softened version of inverse frequency
    assert class_weights(labels, "sqrt")[1] < class_weights(labels, "inverse")[1]
    assert torch.allclose(class_weights(labels, "none"), torch.ones(7))


@pytest.mark.parametrize("transform", [train_transform(), eval_transform()])
def test_transforms_keep_shape_and_range(transform):
    img = np.random.default_rng(0).integers(0, 256, (IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    out = transform(img)
    assert out.shape == (1, IMG_SIZE, IMG_SIZE)
    assert out.dtype == torch.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_checkpoint_round_trip(tiny_checkpoint):
    model, ckpt = load_checkpoint(tiny_checkpoint)
    assert ckpt["config"]["arch"] == "small_cnn"
    assert not model.training
