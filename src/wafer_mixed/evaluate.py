"""
evaluate.py — Test-set evaluation for the multi-label baseline.

Entry point: python -m wafer_mixed.evaluate [--config configs/baseline.yaml]
             [--checkpoint path]

Reports (all at the fixed 0.5 sigmoid threshold — per-label tuned thresholds
are Phase 3):
    - Per-label precision/recall/F1/support and macro-F1 over the 8 labels.
    - Exact-match ratio (all 8 labels simultaneously correct).
    - Single-vs-mixed breakdown: same metrics on single-defect and
      mixed-defect subsets, plus per-label recall in each — the study's
      central question is whether superposition degrades recognition.
    - Spurious-activation matrix on the mixed subset: S[i,j] = how often
      the presence of defect i drags in a false prediction of j. This is
      the multi-label analogue of the confusion matrix.

Artifacts written to outputs/: per_label_metrics.csv, metrics.json,
spurious_matrix.png.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from wafer_mixed.config import MixedConfig, build_arg_parser
from wafer_mixed.data import LABEL_NAMES, get_dataloaders
from wafer_mixed.metrics import (
    exact_match_ratio,
    per_label_recall_by_subset,
    predict_multihot,
    spurious_activation_matrix,
    subset_breakdown,
    subset_masks,
)
from wafer_mixed.model import build_model


def _fmt(x: float, spec: str = ".4f") -> str:
    """Format a metric value, rendering metrics.py's NaN sentinel as a dash."""
    return "—" if np.isnan(x) else format(x, spec)


def evaluate(cfg: MixedConfig, checkpoint_path: Path | None = None) -> None:
    if checkpoint_path is None:
        checkpoint_path = cfg.output_dir / "best.pt"

    ckpt = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)

    # Honour every setting that shapes the model or its input from the
    # checkpoint, so an edited baseline.yaml between training and evaluation
    # can't mismatch state-dict keys — or silently evaluate at a different
    # resolution than the model was trained on (adaptive pooling would accept
    # it without complaint). New model hyperparameters must be added here.
    saved_cfg = ckpt.get("cfg", {})
    cfg.arch = str(saved_cfg.get("arch", cfg.arch))
    cfg.cbam = bool(saved_cfg.get("cbam", cfg.cbam))
    cfg.cbam_reduction = int(saved_cfg.get("cbam_reduction", cfg.cbam_reduction))
    cfg.input_size = int(saved_cfg.get("input_size", cfg.input_size))

    model = build_model(cfg).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint : {checkpoint_path}  (epoch {ckpt.get('epoch', '?')}, "
          f"val macro-F1 {ckpt.get('val_macro_f1', float('nan')):.4f})")

    _, _, test_loader = get_dataloaders(cfg)

    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for inputs, targets in tqdm(test_loader, desc="Evaluating"):
        with torch.no_grad():
            logits = model(inputs.to(cfg.device, non_blocking=True))
        all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_targets.append(targets.numpy().astype(np.int64))

    probs  = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    y_pred = predict_multihot(probs)

    # Single computation path for the F1 numbers: headline macro-F1 is the
    # mean of the same per-label array printed in the table below.
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    m_f1  = float(f1.mean())
    exact = exact_match_ratio(y_true, y_pred)

    print("\n" + "=" * 64)
    print("TEST SET RESULTS  (multi-label @ sigmoid 0.5)")
    print("=" * 64)
    print(f"  Macro-F1 (8 labels) : {m_f1:.4f}  ← headline metric")
    print(f"  Exact-match ratio   : {exact:.4f}  (all 8 labels correct)")

    # --- per-label table ---------------------------------------------------
    print(f"\n  {'label':<10} {'prec':>7} {'recall':>7} {'f1':>7} {'support':>8}")
    for i, name in enumerate(LABEL_NAMES):
        flag = "  (small support)" if support[i] < 200 else ""
        print(f"  {name:<10} {prec[i]:>7.4f} {rec[i]:>7.4f} {f1[i]:>7.4f} "
              f"{support[i]:>8d}{flag}")

    # --- single-vs-mixed breakdown ------------------------------------------
    breakdown = subset_breakdown(y_true, y_pred)
    print("\n  Subset breakdown (macro-F1 averages only labels with support "
          "in the subset):")
    print(f"  {'subset':<8} {'n':>6} {'exact-match':>12} {'macro-F1':>10}")
    for name, e in breakdown.items():
        print(f"  {name:<8} {e['n']:>6d} {e['exact_match']:>12.4f} "
              f"{_fmt(e['macro_f1']):>10}")

    recalls = per_label_recall_by_subset(y_true, y_pred)
    print(f"\n  Per-label recall, single vs mixed (— = label never in subset):")
    print(f"  {'label':<10} {'single':>8} {'mixed':>8}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<10} {_fmt(recalls['single'][i]):>8} "
              f"{_fmt(recalls['mixed'][i]):>8}")

    # --- confusion inside mixes ----------------------------------------------
    S = spurious_activation_matrix(y_true, y_pred, subset_masks(y_true)["mixed"])
    pairs = [
        (S[i, j], LABEL_NAMES[i], LABEL_NAMES[j])
        for i in range(len(LABEL_NAMES))
        for j in range(len(LABEL_NAMES))
        if not np.isnan(S[i, j]) and S[i, j] > 0
    ]
    pairs.sort(reverse=True)
    print("\n  Top spurious activations inside mixes "
          "(true label → falsely predicted label):")
    for rate, ti, pj in pairs[:6]:
        print(f"    {ti} → +{pj}: {rate:.3f}")
    if not pairs:
        print("    (none — no false positives conditioned on a co-present label)")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(prec, rec, f1, support, m_f1, cfg.output_dir)
    _save_json(m_f1, exact, breakdown, recalls, cfg.output_dir)
    _save_spurious_heatmap(S, cfg.output_dir)


def _save_csv(prec, rec, f1, support, m_f1, output_dir: Path) -> None:
    csv_path = output_dir / "per_label_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "precision", "recall", "f1_score", "support"])
        for i, name in enumerate(LABEL_NAMES):
            w.writerow([name, f"{prec[i]:.4f}", f"{rec[i]:.4f}",
                        f"{f1[i]:.4f}", int(support[i])])
        w.writerow(["macro_avg", f"{prec.mean():.4f}", f"{rec.mean():.4f}",
                    f"{m_f1:.4f}", ""])
    print(f"\nPer-label CSV : {csv_path}")


def _save_json(m_f1, exact, breakdown, recalls, output_dir: Path) -> None:
    def _clean(x):
        return None if isinstance(x, float) and np.isnan(x) else x

    payload = {
        "macro_f1": m_f1,
        "exact_match": exact,
        "subsets": {
            k: {kk: _clean(float(vv)) if kk != "n" else vv for kk, vv in e.items()}
            for k, e in breakdown.items()
        },
        "recall_by_subset": {
            k: [_clean(float(v)) for v in arr] for k, arr in recalls.items()
        },
        "label_names": LABEL_NAMES,
    }
    json_path = output_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Metrics JSON  : {json_path}")


def _save_spurious_heatmap(S: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.nan_to_num(S, nan=0.0), cmap="Reds", vmin=0.0)
    ax.set_title("Spurious activation inside mixes\n"
                 "S[i,j] = P(predict j | i true, j not true)", fontsize=11)
    ax.set_xlabel("Falsely predicted label")
    ax.set_ylabel("True (present) label")
    ticks = range(len(LABEL_NAMES))
    ax.set_xticks(ticks); ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(LABEL_NAMES, fontsize=8)
    plt.colorbar(im, ax=ax)
    for i in ticks:
        for j in ticks:
            cell = _fmt(S[i, j], ".2f")
            ax.text(j, i, cell, ha="center", va="center", fontsize=7,
                    color="white" if (not np.isnan(S[i, j]) and S[i, j] > 0.5 * np.nanmax(S)) else "black")
    plt.tight_layout()
    png_path = output_dir / "spurious_matrix.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Spurious matrix: {png_path}")


if __name__ == "__main__":
    parser = build_arg_parser("wafer-mixed evaluate")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to .pt checkpoint (default: outputs/best.pt)")
    args = parser.parse_args()
    cfg  = MixedConfig.from_yaml_and_args(args.config, args)
    evaluate(cfg, checkpoint_path=args.checkpoint)
