"""
download_data.py — Download MixedWM38, verify it, and generate the persisted split.

Unlike WM-811K (manual Kaggle download), MixedWM38 is directly downloadable
from the authors' Google Drive link (github.com/Junliangwangdhu/WaferMap):

    python scripts/download_data.py               # download + verify + write splits
    python scripts/download_data.py --check-gpu   # additionally run CUDA smoke test

If the Drive download fails (rate limit, link rot), fetch the file manually
from any source listed in docs/DATA.md and place it at data/raw/MixedWM38.npz,
then re-run this script to verify and generate splits.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

# Works without pip install -e . (same convention as scripts/eda.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Authors' Google Drive file (see github.com/Junliangwangdhu/WaferMap README).
_DRIVE_ID = "1M59pX-lPqL9APBIbp2AKQRTvngeUK8Va"
_URL = (
    "https://drive.usercontent.google.com/download"
    f"?id={_DRIVE_ID}&export=download&confirm=t"
)
# SHA256 of the file as downloaded 2026-07-01 (412,387,226 bytes).
_SHA256 = "a19e791c85ccd5051c080169a3d2b17902a42545cf749e9869f6c96312bcdc69"

_EXPECTED_N = 38015
_EXPECTED_COMBOS = 38


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(npz_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MixedWM38 (~412 MB) from Google Drive...")
    tmp = npz_path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(_URL, tmp)
    except Exception as e:
        print(
            f"\nERROR: download failed ({e}).\n"
            "Fetch the file manually (sources in docs/DATA.md) and place it at\n"
            f"{npz_path}, then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    tmp.rename(npz_path)


def verify(npz_path: Path) -> None:
    import numpy as np

    digest = _sha256(npz_path)
    if digest != _SHA256:
        print(
            f"WARNING: SHA256 mismatch.\n  expected {_SHA256}\n  got      {digest}\n"
            "The upstream file may have changed — inspect before trusting results.",
            file=sys.stderr,
        )
    else:
        print(f"SHA256 OK: {digest}")

    try:
        d = np.load(npz_path)
        maps, labels = d["arr_0"], d["arr_1"]
    except Exception as e:
        print(
            f"\nERROR: {npz_path} is not a readable npz ({e}).\n"
            "Google Drive may have served an HTML page (rate limit / quota).\n"
            "Delete the file, fetch it manually from a source in docs/DATA.md,\n"
            "and re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"maps:   {maps.shape} {maps.dtype}")
    print(f"labels: {labels.shape} {labels.dtype}")

    assert maps.shape == (_EXPECTED_N, 52, 52), f"unexpected maps shape {maps.shape}"
    assert labels.shape == (_EXPECTED_N, 8), f"unexpected labels shape {labels.shape}"

    n_combos = len(np.unique(labels, axis=0))
    assert n_combos == _EXPECTED_COMBOS, f"expected 38 combos, found {n_combos}"

    n_stray = int((maps > 2).sum())
    print(f"combos: {n_combos}  |  stray pixels >2: {n_stray} (clipped by encode_map)")
    print("Dataset verified.")


def write_splits(data_root: Path) -> None:
    from wafer_mixed.config import MixedConfig, REPO_ROOT
    from wafer_mixed.data import load_raw, make_splits

    cfg = MixedConfig.from_yaml(REPO_ROOT / "configs" / "baseline.yaml")
    cfg.data_root = data_root
    if cfg.split_path.exists():
        print(f"Splits already exist at {cfg.split_path} — leaving untouched.")
        return
    _, labels = load_raw(cfg.data_root)
    make_splits(cfg, labels)


def check_gpu() -> None:
    import torch

    print(f"\ntorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA device found. Install cu128 PyTorch wheel:")
        print("  pip install torch>=2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128")
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Device: {name}")
    print(f"Compute capability: {cap[0]}.{cap[1]}")

    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = a @ b
    print(f"Matmul smoke test: {c.shape} — kernel OK")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-root", type=Path, default=None,
        help="Path to data/raw/. Defaults to data_root from configs/baseline.yaml.",
    )
    p.add_argument("--check-gpu", action="store_true", help="Run CUDA smoke test.")
    args = p.parse_args()

    if args.data_root is not None:
        data_root = args.data_root
    else:
        from wafer_mixed.config import MixedConfig, REPO_ROOT
        cfg = MixedConfig.from_yaml(REPO_ROOT / "configs" / "baseline.yaml")
        data_root = cfg.data_root

    npz_path = data_root / "MixedWM38.npz"
    if npz_path.exists():
        print(f"{npz_path} already present — skipping download.")
    else:
        download(npz_path)

    verify(npz_path)
    write_splits(data_root)

    if args.check_gpu:
        check_gpu()
