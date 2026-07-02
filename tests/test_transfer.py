"""
Tests for the Phase 2 donor-backbone loading (train.load_donor_backbone).

All synthetic — tiny checkpoints written to tmp_path, CPU only. Covers the
two donor formats that exist on disk:
    - wafer-ssl SimCLR export: {"backbone_state_dict": headless state}
    - WM-811K supervised best.pt: {"model_state_dict": full model incl. 9-class fc}
"""
import pytest
import torch

from wafer_mixed.config import MixedConfig
from wafer_mixed.model import build_model
from wafer_mixed.train import load_donor_backbone


def _cfg(**overrides) -> MixedConfig:
    base = dict(device="cpu", cbam=True)
    base.update(overrides)
    return MixedConfig(**base)


def _fresh_pair():
    """Two independently initialised copies of the model."""
    return build_model(_cfg()), build_model(_cfg())


def test_loads_backbone_state_dict_format(tmp_path):
    donor, target = _fresh_pair()
    state = {k: v for k, v in donor.state_dict().items() if not k.startswith("fc.")}
    p = tmp_path / "ssl.pt"
    torch.save({"backbone_state_dict": state, "epoch": 1}, p)

    n = load_donor_backbone(target, str(p), "cpu")
    assert n == len(state)
    for k in state:
        torch.testing.assert_close(target.state_dict()[k], donor.state_dict()[k])


def test_loads_model_state_dict_format_dropping_foreign_head(tmp_path):
    """Supervised donor: full model with a 9-class head (WM-811K). The fc
    shape clash must not raise, the backbone must land, and the target's own
    8-logit head must keep its fresh initialisation."""
    donor, target = _fresh_pair()
    fc_before = target.state_dict()["fc.weight"].clone()

    state = donor.state_dict()
    state["fc.weight"] = torch.rand(9, state["fc.weight"].shape[1])  # 9-class head
    state["fc.bias"] = torch.rand(9)
    p = tmp_path / "supervised.pt"
    torch.save({"model_state_dict": state, "epoch": 30}, p)

    n = load_donor_backbone(target, str(p), "cpu")
    assert n == len(state) - 2  # everything but fc.*
    torch.testing.assert_close(
        target.state_dict()["conv1.weight"], donor.state_dict()["conv1.weight"]
    )
    torch.testing.assert_close(target.state_dict()["fc.weight"], fc_before)


def test_unknown_checkpoint_format_raises(tmp_path):
    p = tmp_path / "junk.pt"
    torch.save({"weights": {}}, p)
    _, target = _fresh_pair()
    with pytest.raises(KeyError, match="Not a known donor format"):
        load_donor_backbone(target, str(p), "cpu")


def test_architecture_mismatch_fails_loud(tmp_path):
    """A no-CBAM donor into a CBAM model loads plain ResNet tensors but the
    guard must reject silent partial loads only when <50 % match; a resnet50
    donor into a resnet18 model matches almost nothing and must raise."""
    donor = build_model(_cfg(arch="resnet50", cbam=False))
    state = {k: v for k, v in donor.state_dict().items() if not k.startswith("fc.")}
    p = tmp_path / "wrong_arch.pt"
    torch.save({"backbone_state_dict": state}, p)

    target = build_model(_cfg(arch="resnet18", cbam=True))
    with pytest.raises(RuntimeError, match="Architecture mismatch"):
        load_donor_backbone(target, str(p), "cpu")
