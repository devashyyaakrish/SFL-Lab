"""Modular U-shaped models; the eight units are residual blocks, not layers."""
from __future__ import annotations

import torch
from torch import nn

SUPPORTED_SPLITS = ((1, 6, 1), (2, 5, 1), (2, 4, 2), (3, 4, 1), (3, 3, 2))


class UShapedModel(nn.Module):
    def __init__(self, stem: nn.Module, blocks: list[nn.Module], head: nn.Module, split: list[int]):
        super().__init__()
        if len(split) != 3 or min(split) < 1 or sum(split) != len(blocks):
            raise ValueError("Three positive split sizes must sum to the eight model blocks.")
        a, b, _ = split
        self.front = nn.Sequential(stem, *blocks[:a])
        self.middle = nn.Sequential(*blocks[a:a + b])
        self.back = nn.Sequential(*blocks[a + b:], head)
        self.split = list(split)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.back(self.middle(self.front(x)))


def build_model(model_name: str, num_classes: int, split: list[int]) -> UShapedModel:
    if num_classes < 2:
        raise ValueError("Classification requires at least two classes.")
    if model_name == "resnet18":
        from torchvision.models import resnet18
        base = resnet18(weights=None, num_classes=num_classes)
        stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        blocks = list(base.layer1) + list(base.layer2) + list(base.layer3) + list(base.layer4)
        head = nn.Sequential(base.avgpool, nn.Flatten(), base.fc)
    elif model_name == "tiny_cnn":
        stem = nn.Identity()
        channels = [3, 8, 8, 12, 12, 16, 16, 20, 20]
        blocks = [nn.Sequential(
            nn.Conv2d(channels[i], channels[i + 1], 3, padding=1, stride=2 if i in (2, 4, 6) else 1),
            nn.GroupNorm(2, channels[i + 1]), nn.ReLU(),
        ) for i in range(8)]
        head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(20, num_classes))
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return UShapedModel(stem, blocks, head, split)
