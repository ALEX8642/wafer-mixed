"""
Tests for the ported ResNet+CBAM multi-label model.

All synthetic — no data or GPU required. Small input (64 px) keeps the
forward passes fast on CPU.
"""
import pytest
import torch

from wafer_mixed.config import MixedConfig
from wafer_mixed.data import NUM_LABELS
from wafer_mixed.model import CBAM, build_model


def _cfg(**overrides) -> MixedConfig:
    base = dict(device="cpu", cbam=True)
    base.update(overrides)
    return MixedConfig(**base)


def test_forward_shape_is_8_logits():
    model = build_model(_cfg())
    x = torch.rand(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, NUM_LABELS)


def test_logits_are_unbounded():
    """Head must emit raw logits (no sigmoid) — BCEWithLogitsLoss applies it."""
    model = build_model(_cfg())
    model.eval()
    with torch.no_grad():
        logits = model(torch.rand(4, 3, 64, 64))
    # Raw logits from a randomly initialised net are not confined to [0, 1];
    # a sigmoid-squashed head would be. Check the value range is plausible logits.
    assert logits.min() < 0 or logits.max() > 1


def test_cbam_flag_appends_attention():
    with_cbam = build_model(_cfg(cbam=True))
    without   = build_model(_cfg(cbam=False))
    n_with    = sum(1 for m in with_cbam.modules() if isinstance(m, CBAM))
    n_without = sum(1 for m in without.modules() if isinstance(m, CBAM))
    assert n_with == 4          # one per ResNet stage
    assert n_without == 0


def test_cbam_preserves_shape():
    block = CBAM(channels=32)
    x = torch.rand(2, 32, 16, 16)
    assert block(x).shape == x.shape


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match="Unknown arch"):
        build_model(_cfg(arch="vgg16"))
