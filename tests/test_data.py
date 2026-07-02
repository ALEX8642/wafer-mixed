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
    subsample_indices,
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


def test_augmentation_preserves_onehot_and_target():
    """D4 augmentation must keep the map one-hot and never touch the target."""
    maps = np.stack([_synthetic_wafer()] * 4)
    labels = np.tile(np.array([[0, 1, 0, 1, 0, 0, 0, 0]], dtype=np.int32), (4, 1))
    ds = MixedWaferDataset(maps, labels, input_size=64, augment=True)
    for i in range(len(ds)):
        tensor, target = ds[i]
        assert tensor.shape == (3, 64, 64)
        # rot90/flip permute pixels but each pixel stays a valid one-hot triple
        np.testing.assert_allclose(tensor.sum(dim=0).numpy(), 1.0)
        np.testing.assert_array_equal(target.numpy(), labels[i].astype(np.float32))


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
# Train-fraction subsampling (Phase 2 low-data arms)
# ---------------------------------------------------------------------------

def _synthetic_split(n_per_combo: int = 100, n_combos: int = 5):
    """Labels with n_combos distinct combinations, n_per_combo rows each,
    interleaved so strata are not contiguous. Returns (idx, labels)."""
    combos = np.eye(NUM_LABELS, dtype=np.int32)[:n_combos]
    labels = np.tile(combos, (n_per_combo, 1))
    # a non-trivial "train split": every other row
    idx = np.arange(0, len(labels), 2)
    return idx, labels


def test_subsample_fraction_one_is_identity():
    idx, labels = _synthetic_split()
    out = subsample_indices(idx, labels, 1.0, seed=42)
    np.testing.assert_array_equal(out, idx)


def test_subsample_is_deterministic_and_seed_sensitive():
    idx, labels = _synthetic_split()
    a = subsample_indices(idx, labels, 0.1, seed=42)
    b = subsample_indices(idx, labels, 0.1, seed=42)
    c = subsample_indices(idx, labels, 0.1, seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c), "different seeds must select different maps"


def test_subsample_size_subset_and_stratification():
    idx, labels = _synthetic_split(n_per_combo=100, n_combos=5)
    out = subsample_indices(idx, labels, 0.1, seed=42)
    # subset of the input split — never invents indices (no val/test leakage)
    assert np.isin(out, idx).all()
    assert len(np.unique(out)) == len(out)
    # 5 combos × 50 train rows × 0.1 = 5 per combo, 25 total
    assert len(out) == 25
    per_combo = np.unique(combo_ids(labels[out]), return_counts=True)[1]
    np.testing.assert_array_equal(per_combo, [5] * 5)


def test_subsample_keeps_every_combo_at_tiny_fraction():
    """max(1, ·) floor: rare combos survive even below one expected row."""
    idx, labels = _synthetic_split(n_per_combo=100, n_combos=5)
    out = subsample_indices(idx, labels, 0.001, seed=42)
    assert len(np.unique(combo_ids(labels[out]))) == 5
    assert len(out) == 5  # exactly the floor


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
