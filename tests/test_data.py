"""
Tests for MixedWM38 encoding, multi-hot targets, and split integrity.

Synthetic tests always run; tests marked `needs_data` are skipped unless
data/raw/MixedWM38.npz (and data/splits.npz) exist locally.
"""
import numpy as np
import pytest
import torch

from wafer_mixed.config import MixedConfig, REPO_ROOT
from wafer_mixed.data import (
    LABEL_NAMES,
    NUM_LABELS,
    MixedWaferDataset,
    combo_ids,
    combo_name,
    encode_map,
    load_raw,
    load_splits,
)

_NPZ = REPO_ROOT / "data" / "raw" / "MixedWM38.npz"
_SPLITS = REPO_ROOT / "data" / "splits.npz"

needs_data = pytest.mark.skipif(
    not (_NPZ.exists() and _SPLITS.exists()),
    reason="MixedWM38.npz / splits.npz not present (run scripts/download_data.py)",
)


def _synthetic_wafer(size: int = 52) -> np.ndarray:
    """Ring of 0 (outside wafer), interior 1 (pass), centre pixel 2 (fail)."""
    m = np.ones((size, size), dtype=np.int32)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0
    m[size // 2, size // 2] = 2
    return m


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def test_encode_roundtrip():
    """encode_map → argmax along channel dim must recover the original map."""
    wmap = _synthetic_wafer()
    tensor = encode_map(wmap)
    assert tensor.shape == (3, 52, 52)
    assert tensor.dtype == torch.float32
    decoded = tensor.argmax(dim=0).numpy().astype(np.int32)
    np.testing.assert_array_equal(decoded, wmap)


def test_encode_clips_stray_value_3():
    """The raw data contains 214 stray value-3 pixels; they must clip to 2."""
    wmap = np.array([[0, 1, 2, 3, -1]], dtype=np.int32)
    decoded = encode_map(wmap).argmax(dim=0).numpy()
    np.testing.assert_array_equal(decoded, np.array([[0, 1, 2, 2, 0]]))


# ---------------------------------------------------------------------------
# Dataset / multi-hot targets
# ---------------------------------------------------------------------------

def test_dataset_yields_multihot_float():
    maps = np.stack([_synthetic_wafer(), _synthetic_wafer()])
    labels = np.array([[1, 0, 1, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    ds = MixedWaferDataset(maps, labels, input_size=64)
    tensor, target = ds[0]
    assert tensor.shape == (3, 64, 64)
    assert target.shape == (NUM_LABELS,)
    assert target.dtype == torch.float32
    np.testing.assert_array_equal(target.numpy(), labels[0].astype(np.float32))
    # normal wafer → all-zero target
    _, target_normal = ds[1]
    assert target_normal.sum() == 0


def test_combo_helpers():
    row = np.array([1, 0, 0, 0, 0, 0, 1, 0])
    assert combo_name(row) == "Center+Scratch"
    assert combo_name(np.zeros(8, dtype=int)) == "normal"
    # combo_ids must be injective over distinct multi-hot rows
    rows = np.array([[0]*8, [1]+[0]*7, [0]*7+[1], [1]*8])
    assert len(np.unique(combo_ids(rows))) == 4


def test_label_names_count():
    assert len(LABEL_NAMES) == NUM_LABELS == 8


# ---------------------------------------------------------------------------
# Real-data split integrity
# ---------------------------------------------------------------------------

@needs_data
def test_split_no_leakage_and_full_coverage():
    """Train/val/test must be pairwise disjoint and jointly cover all rows."""
    cfg = MixedConfig.from_yaml(REPO_ROOT / "configs" / "baseline.yaml")
    _, labels = load_raw(cfg.data_root)
    splits = load_splits(cfg)
    train, val, test = splits["train"], splits["val"], splits["test"]

    assert len(np.intersect1d(train, val)) == 0
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(val, test)) == 0

    union = np.concatenate([train, val, test])
    assert len(union) == len(labels)
    np.testing.assert_array_equal(np.sort(union), np.arange(len(labels)))


@needs_data
def test_split_stratification_covers_all_combos():
    """Every one of the 38 combinations must appear in each split."""
    cfg = MixedConfig.from_yaml(REPO_ROOT / "configs" / "baseline.yaml")
    _, labels = load_raw(cfg.data_root)
    splits = load_splits(cfg)
    ids = combo_ids(labels)
    all_combos = np.unique(ids)
    assert len(all_combos) == 38
    for name, idx in splits.items():
        present = np.unique(ids[idx])
        assert len(present) == 38, f"{name} split is missing combinations"


@needs_data
def test_dataset_targets_match_raw_labels():
    """Dataset targets must equal the raw multi-hot rows for the same indices."""
    cfg = MixedConfig.from_yaml(REPO_ROOT / "configs" / "baseline.yaml")
    maps, labels = load_raw(cfg.data_root)
    idx = load_splits(cfg)["val"][:16]
    ds = MixedWaferDataset(maps[idx], labels[idx], input_size=64)
    for i in range(len(idx)):
        tensor, target = ds[i]
        assert tensor.shape == (3, 64, 64)
        np.testing.assert_array_equal(
            target.numpy(), labels[idx[i]].astype(np.float32)
        )
