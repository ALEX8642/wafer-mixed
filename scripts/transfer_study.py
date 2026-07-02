"""
transfer_study.py — Phase 2 transfer sweep: 3 init arms × train fractions × seeds.

Arms:
    scratch     random init (the Phase 1 configuration)
    supervised  WM-811K supervised backbone (wafer-defect-classifier outputs/best.pt)
    simclr      wafer-ssl SimCLR backbone (wafer-ssl outputs/pretrained_backbone.pt)

Every run shares baseline.yaml hyperparameters; the three arms inside a
(fraction, seed) cell differ only in initialisation. The train subsample is
seeded by `seed` alone (see data.subsample_indices), so all three arms of a
cell train on the identical maps: deltas are paired comparisons, not
resampling noise.

Budget policy: "same epochs" is not "same budget" once the train set shrinks
— 30 epochs at 1 % is ~90 gradient steps, and the baseline's patience-7
early stop fires during the slow start (observed: a scratch 1 % run killed
at epoch 10 with val F1 0.08). Sub-fraction cells therefore scale epochs by
1/fraction (≈ equal gradient steps, capped at MAX_EPOCHS) with patience
widened to SMALL_PATIENCE, so every run trains to plateau and early stop
only cuts genuinely converged runs. The full-data cell keeps the exact
Phase 1 budget for comparability with the baseline.

Seed policy: fraction 1.0 runs a single seed (Phase 1 showed the full-data
regime saturates; the interesting variance lives lower), 0.1 and 0.01 run
three seeds each — small-sample variance is real exactly where transfer has
room to show anything.

Rows append to outputs/transfer/results.csv; (arm, fraction, seed) cells
already present are skipped, so the sweep is resumable and extensible.

Run (5090):
    python scripts/transfer_study.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafer_mixed.config import REPO_ROOT, MixedConfig
from wafer_mixed.data import get_dataloaders
from wafer_mixed.evaluate import collect_probs
from wafer_mixed.metrics import exact_match_ratio, macro_f1, predict_multihot
from wafer_mixed.model import build_model
from wafer_mixed.train import train

WORKSPACE = REPO_ROOT.parent  # the three wafer repos are siblings

ARMS: dict[str, str] = {
    "scratch": "",
    "supervised": str(WORKSPACE / "wafer-defect-classifier" / "outputs" / "best.pt"),
    "simclr": str(WORKSPACE / "wafer-ssl" / "outputs" / "pretrained_backbone.pt"),
}
FRACTIONS = [1.0, 0.1, 0.01]
SEEDS_FULL = [42]
SEEDS_SMALL = [42, 43, 44]
MAX_EPOCHS = 300      # cap on the 1/fraction epoch scaling
SMALL_PATIENCE = 30   # early-stop patience for sub-fraction cells

CSV_FIELDS = [
    "arm", "fraction", "seed", "n_train", "epochs_run",
    "best_val_f1", "test_macro_f1", "test_exact_match",
]


def build_cfg(yaml_path: Path, **overrides) -> MixedConfig:
    """baseline.yaml + overrides → MixedConfig (single construction)."""
    with open(yaml_path) as f:
        merged: dict = yaml.safe_load(f)
    merged.update(overrides)
    return MixedConfig(**merged)


def eval_test(cfg: MixedConfig, ckpt_path: Path, test_loader) -> tuple[float, float]:
    """Test-set (macro-F1, exact-match) for one finished run. The test loader
    is identical for every run (fractions touch the train split only), so the
    caller builds it once instead of re-reading the raw arrays per cell."""
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    model = build_model(cfg).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, probs = collect_probs(model, test_loader, cfg.device, desc="test", leave=False)
    y_pred = predict_multihot(probs)
    return macro_f1(y_true, y_pred), exact_match_ratio(y_true, y_pred)


def done_cells(csv_path: Path) -> set[tuple[str, float, int]]:
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {
            (r["arm"], float(r["fraction"]), int(r["seed"]))
            for r in csv.DictReader(f)
        }


def append_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="wafer-mixed Phase 2 transfer sweep")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "baseline.yaml")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    args = p.parse_args()

    for arm in args.arms:
        if ARMS[arm] and not Path(ARMS[arm]).exists():
            sys.exit(f"Donor checkpoint for arm {arm!r} not found: {ARMS[arm]}")

    transfer_dir = REPO_ROOT / "outputs" / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    csv_path = transfer_dir / "results.csv"
    done = done_cells(csv_path)

    # Smallest fraction first: the cheap cells exercise every arm (incl. donor
    # loading) within minutes, so a bad configuration fails before the
    # expensive full-data runs start.
    runs = [
        (arm, frac, seed)
        for frac in sorted(args.fractions)
        for seed in (SEEDS_FULL if frac >= 1.0 else SEEDS_SMALL)
        for arm in args.arms
    ]
    print(f"Sweep: {len(runs)} cells ({len(done)} already in {csv_path.name})")
    base_cfg = build_cfg(args.config)
    base_epochs = base_cfg.num_epochs
    _, _, test_loader = get_dataloaders(base_cfg)

    for i, (arm, frac, seed) in enumerate(runs, 1):
        if (arm, frac, seed) in done:
            print(f"[{i:2d}/{len(runs)}] {arm} f={frac} s={seed} — done, skipping")
            continue
        print(f"\n[{i:2d}/{len(runs)}] ===== arm={arm}  fraction={frac}  seed={seed} =====")
        overrides = dict(
            seed=seed,
            train_fraction=frac,
            backbone_ckpt_path=ARMS[arm],
            output_dir=str(transfer_dir / f"{arm}_f{frac}_s{seed}"),
        )
        if frac < 1.0:  # ≈ equal gradient-step budget; see module docstring
            overrides["num_epochs"] = min(round(base_epochs / frac), MAX_EPOCHS)
            overrides["patience"] = SMALL_PATIENCE
        cfg = build_cfg(args.config, **overrides)
        result = train(cfg)
        test_f1, test_em = eval_test(cfg, result["ckpt_path"], test_loader)
        append_row(csv_path, {
            "arm": arm, "fraction": frac, "seed": seed,
            "n_train": result["n_train"], "epochs_run": result["epochs_run"],
            "best_val_f1": f"{result['best_val_f1']:.4f}",
            "test_macro_f1": f"{test_f1:.4f}",
            "test_exact_match": f"{test_em:.4f}",
        })
        print(f"→ test macro-F1 {test_f1:.4f}  exact-match {test_em:.4f}  "
              f"(logged to {csv_path.name})")

    print(f"\nSweep complete. Results: {csv_path}")


if __name__ == "__main__":
    main()
