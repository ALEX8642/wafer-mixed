"""
model.py — ResNet+CBAM multi-label classifier, ported from wafer-defect-classifier.

The architecture is identical to the main repo (ResNet-18 default, CBAM after
each stage); only the head differs: 8 independent logits for the 8 basic
defect labels, consumed by BCE-with-logits. A wafer map can activate any
subset of the 8 (all-zero = normal wafer), so this is multi-label, not
38-way multi-class over the combinations.

Input compatibility:
    Both ResNet variants expect 3-channel input. One-hot encoding of {0,1,2}
    pixel values produces exactly 3 channels — no first-conv adaptation needed.

CBAM (Convolutional Block Attention Module — Woo et al., ECCV 2018):
    Channel attention (what features matter) followed by spatial attention
    (where they matter), appended after each ResNet stage. On WM-811K this
    gave substantial tail-class recall gains; for mixed patterns the spatial
    branch is the interesting part — Phase 3 checks whether attention
    separates superposed signatures.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from wafer_mixed.config import MixedConfig
from wafer_mixed.data import NUM_LABELS


# ---------------------------------------------------------------------------
# CBAM components (verbatim from wafer-defect-classifier)
# ---------------------------------------------------------------------------

class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.shared_mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.shared_mlp(x.amax(dim=(2, 3), keepdim=True))
        return self.sigmoid(avg + mx)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Channel + spatial attention appended after a ResNet stage."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction)
        self.sa = _SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg: MixedConfig, num_labels: int = NUM_LABELS) -> nn.Module:
    """
    Build ResNet-18 (default) or ResNet-50 with a num_labels-logit head.
    When cfg.cbam=True, CBAM attention is appended after each ResNet stage.

    The head is a plain Linear producing independent logits — no sigmoid here;
    train.py uses BCEWithLogitsLoss and inference applies torch.sigmoid.
    """
    arch = cfg.arch.lower()
    pretrained = cfg.pretrained

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        stage_channels = [64, 128, 256, 512]
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        stage_channels = [256, 512, 1024, 2048]
    else:
        raise ValueError(f"Unknown arch {arch!r}. Supported: resnet18, resnet50.")

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_labels)

    if cfg.cbam:
        r = cfg.cbam_reduction
        backbone.layer1 = nn.Sequential(backbone.layer1, CBAM(stage_channels[0], r))
        backbone.layer2 = nn.Sequential(backbone.layer2, CBAM(stage_channels[1], r))
        backbone.layer3 = nn.Sequential(backbone.layer3, CBAM(stage_channels[2], r))
        backbone.layer4 = nn.Sequential(backbone.layer4, CBAM(stage_channels[3], r))

    return backbone
