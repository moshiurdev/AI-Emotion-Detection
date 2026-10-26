"""Model definitions.

Every model takes a batch of grayscale faces shaped ``(B, 1, 48, 48)`` with
values in ``[0, 1]`` and returns ``(B, 7)`` logits. Resizing and
normalisation live inside the model, so the data pipeline and the inference
code never need to know which backbone is in use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tvm

from . import NUM_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PretrainedBackbone(nn.Module):
    """ImageNet backbone adapted to 48x48 grayscale input.

    The face is upsampled to ``input_size`` (ImageNet filters are tuned for
    larger objects than 48 px), replicated to three channels and normalised
    with ImageNet statistics before entering the backbone.
    """

    def __init__(self, arch: str, input_size: int, pretrained: bool, dropout: float):
        super().__init__()
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        if arch == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, NUM_CLASSES))
        elif arch == "mobilenet_v3_small":
            weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.mobilenet_v3_small(weights=weights, dropout=dropout)
            in_features = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
        else:
            raise ValueError(f"unknown backbone: {arch}")
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        return self.net(x)


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """VGG-style baseline trained from scratch at native 48x48 resolution."""

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 64),     # 24x24
            _conv_block(64, 128),   # 12x12
            _conv_block(128, 256),  # 6x6
            _conv_block(256, 512),  # 3x3
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features((x - 0.5) / 0.25))


def build_model(arch: str, input_size: int = 96, pretrained: bool = True,
                dropout: float = 0.3) -> nn.Module:
    if arch == "small_cnn":
        return SmallCNN(dropout=dropout)
    return PretrainedBackbone(arch, input_size=input_size, pretrained=pretrained, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
