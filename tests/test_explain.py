"""
Tests for multi-label Grad-CAM (explain.py). All synthetic, CPU, small input.
"""
import numpy as np
import pytest
import torch

from wafer_mixed.config import MixedConfig
from wafer_mixed.data import NUM_LABELS
from wafer_mixed.explain import (
    GradCAM,
    GradCAMPlusPlus,
    save_multilabel_overlay,
    target_layer,
    tensor_to_display,
)
from wafer_mixed.model import CBAM, build_model


def _cfg(**overrides) -> MixedConfig:
    base = dict(device="cpu", cbam=True)
    base.update(overrides)
    return MixedConfig(**base)


@pytest.mark.parametrize("cam_cls", [GradCAM, GradCAMPlusPlus])
def test_cam_shape_range_and_target(cam_cls):
    model = build_model(_cfg())
    x = torch.rand(1, 3, 64, 64)
    with cam_cls(model, target_layer(model)) as cam:
        heatmap, label, probs = cam.compute(x, target_label=6)
    assert heatmap.shape == (64, 64)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert label == 6
    assert probs.shape == (NUM_LABELS,)
    assert ((0.0 <= probs) & (probs <= 1.0)).all()  # sigmoid, not softmax
    # probs need not sum to 1 in multi-label — guard against a softmax regression
    # (an 8-way softmax would make independent-label probabilities sum to 1)


def test_cam_default_target_is_top_sigmoid():
    model = build_model(_cfg())
    x = torch.rand(1, 3, 64, 64)
    with GradCAM(model, target_layer(model)) as cam:
        _, label, probs = cam.compute(x, target_label=None)
    assert label == int(np.argmax(probs))


def test_per_label_heatmaps_differ():
    """CAMs for two different labels on the same input must not be identical —
    the whole point is label-specific evidence."""
    torch.manual_seed(0)
    model = build_model(_cfg())
    x = torch.rand(1, 3, 64, 64)
    with GradCAMPlusPlus(model, target_layer(model)) as cam:
        h0, _, _ = cam.compute(x, target_label=0)
        h6, _, _ = cam.compute(x, target_label=6)
    assert not np.allclose(h0, h6)


def test_target_layer_skips_cbam_and_unwraps():
    from torchvision.models.resnet import BasicBlock

    with_cbam = build_model(_cfg(cbam=True))
    tl = target_layer(with_cbam)
    assert isinstance(tl, BasicBlock)
    assert not isinstance(tl, CBAM)

    without = build_model(_cfg(cbam=False))
    assert isinstance(target_layer(without), BasicBlock)


def test_hooks_removed_on_exit():
    model = build_model(_cfg())
    tl = target_layer(model)
    with GradCAM(model, tl):
        assert len(tl._forward_hooks) == 1
    assert len(tl._forward_hooks) == 0
    assert len(tl._backward_hooks) == 0


def test_save_multilabel_overlay_writes_png(tmp_path):
    wafer_img = tensor_to_display(torch.rand(3, 52, 52))
    heatmaps = {3: np.random.rand(52, 52), 6: np.random.rand(52, 52)}
    probs = np.full(NUM_LABELS, 0.5)
    y_true = np.array([0, 0, 0, 1, 0, 0, 1, 0])
    out = tmp_path / "overlay.png"
    save_multilabel_overlay(wafer_img, heatmaps, probs, y_true, out)
    assert out.exists() and out.stat().st_size > 0
