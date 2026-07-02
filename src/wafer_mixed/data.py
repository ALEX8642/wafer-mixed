"""
data.py — MixedWM38 dataset loading, splitting, and DataLoader construction.

Dataset: MixedWM38 (Wang et al. 2020), single .npz file with two arrays:
    arr_0: (38015, 52, 52) int32 wafer maps, documented values {0, 1, 2}
           (0 = outside wafer boundary, 1 = passing die, 2 = failing die).
           Verified deviation: 214 pixels across 105 maps carry the value 3
           (undocumented artifact, mostly in Random-labeled maps); encode_map
           clips to [0, 2] so they read as failing die. See docs/DATA.md.
    arr_1: (38015, 8) int32 multi-hot labels over the 8 basic defect types.
           All-zero row = normal wafer. Label ordering is undocumented
           upstream; it was verified visually (docs/DATA.md, assets/) as:
           [Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Scratch, Random].

Encoding choice (one-hot, 3 channels): identical to wafer-defect-classifier —
pixel values {0,1,2} are categorical, not ordinal, so they become 3 binary
channels compatible with a 3-channel ResNet first conv.

Split strategy: MixedWM38 provides no train/test split. We stratify by the
full 38-type combination (each unique multi-hot row is one stratum) so every
mix — including the 149-sample single Near-full — appears in train, val, and
test. Indices are generated once with a fixed seed and persisted to
data/splits.npz; loaders always read the persisted file so every machine and
every later phase sees the identical split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from wafer_mixed.config import MixedConfig

NPZ_NAME = "MixedWM38.npz"

# arr_1 column ordering, verified visually in Phase 0 (see docs/DATA.md).
LABEL_NAMES: list[str] = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Scratch", "Random",
]
NUM_LABELS = len(LABEL_NAMES)


# ---------------------------------------------------------------------------
# Loading and label helpers
# ---------------------------------------------------------------------------

def load_raw(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load (maps, labels) from data/raw/MixedWM38.npz."""
    npz_path = data_root / NPZ_NAME
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run scripts/download_data.py first."
        )
    d = np.load(npz_path)
    maps, labels = d["arr_0"], d["arr_1"]
    if maps.shape[1:] != (52, 52) or labels.shape != (maps.shape[0], NUM_LABELS):
        raise ValueError(
            f"Unexpected array shapes: maps {maps.shape}, labels {labels.shape}. "
            "The download may be corrupt — re-run scripts/download_data.py."
        )
    return maps, labels


def combo_name(multi_hot: np.ndarray) -> str:
    """Human-readable combination name, e.g. 'Center+Scratch' or 'normal'."""
    active = [LABEL_NAMES[i] for i in range(NUM_LABELS) if multi_hot[i]]
    return "+".join(active) if active else "normal"


def combo_ids(labels: np.ndarray) -> np.ndarray:
    """
    Map each multi-hot row to an integer id of its full combination
    (bit-packed columns). Used as the stratification key: two rows share an
    id iff they have the identical defect combination.
    """
    weights = 1 << np.arange(NUM_LABELS)
    return (labels.astype(np.int64) * weights).sum(axis=1)


# ---------------------------------------------------------------------------
# Map encoding and resizing (same semantics as wafer-defect-classifier)
# ---------------------------------------------------------------------------

def encode_map(wmap: np.ndarray) -> torch.Tensor:
    """
    Convert a 2D wafer map (values 0/1/2) to a (3, H, W) one-hot float tensor.
    Values outside [0, 2] are clipped before encoding — this absorbs the 214
    stray value-3 pixels in the raw data (treated as failing die).
    """
    arr = np.clip(np.asarray(wmap, dtype=np.int64), 0, 2)
    t = torch.tensor(arr, dtype=torch.long)
    return F.one_hot(t, num_classes=3).permute(2, 0, 1).float()  # (3, H, W)


def resize_map(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """
    Resize a (3, H, W) one-hot tensor to (3, size, size).
    Nearest-neighbour keeps pixel values binary (no interpolation artefacts).
    """
    return F.interpolate(
        tensor.unsqueeze(0), size=(size, size), mode="nearest"
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MixedWaferDataset(Dataset):
    """
    PyTorch Dataset over MixedWM38 wafer maps.

    Yields (tensor, target) where tensor is (3, input_size, input_size) and
    target is an 8-dim float32 multi-hot vector (BCE-with-logits ready).
    """

    def __init__(
        self,
        maps: np.ndarray,
        labels: np.ndarray,
        input_size: int = 224,
    ) -> None:
        assert len(maps) == len(labels), "maps/labels length mismatch"
        self.maps = maps
        self.targets = torch.tensor(labels, dtype=torch.float32)
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = encode_map(self.maps[idx])
        tensor = resize_map(tensor, self.input_size)
        return tensor, self.targets[idx]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def make_splits(cfg: MixedConfig, labels: np.ndarray) -> None:
    """
    Generate stratified train/val/test index arrays and persist them to
    cfg.split_path. Stratification key = full 38-type combination, so rare
    mixes land in all three splits. Overwrites any existing file.
    """
    n = len(labels)
    strata = combo_ids(labels)
    idx_all = np.arange(n)

    idx_pool, idx_test = train_test_split(
        idx_all, test_size=cfg.test_frac, stratify=strata,
        random_state=cfg.seed,
    )
    idx_train, idx_val = train_test_split(
        idx_pool, test_size=cfg.val_frac, stratify=strata[idx_pool],
        random_state=cfg.seed,
    )

    cfg.split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.split_path,
        train=np.sort(idx_train),
        val=np.sort(idx_val),
        test=np.sort(idx_test),
        seed=np.array([cfg.seed]),
    )
    print(
        f"Splits written to {cfg.split_path} — "
        f"train: {len(idx_train):,}  val: {len(idx_val):,}  test: {len(idx_test):,}"
    )


def load_splits(cfg: MixedConfig) -> Dict[str, np.ndarray]:
    """Load persisted split indices. Raises if make_splits has not been run."""
    if not cfg.split_path.exists():
        raise FileNotFoundError(
            f"{cfg.split_path} not found. Run scripts/download_data.py "
            "(which generates the persisted split) first."
        )
    d = np.load(cfg.split_path)
    return {"train": d["train"], "val": d["val"], "test": d["test"]}


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: MixedConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) over the persisted split."""
    maps, labels = load_raw(cfg.data_root)
    splits = load_splits(cfg)

    def _loader(name: str, shuffle: bool) -> DataLoader:
        idx = splits[name]
        ds = MixedWaferDataset(maps[idx], labels[idx], cfg.input_size)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.device.startswith("cuda"),
        )

    return _loader("train", True), _loader("val", False), _loader("test", False)
