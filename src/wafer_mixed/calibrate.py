"""
calibrate.py — per-label temperature scaling, per-label ECE, tuned decision
thresholds, and cost-of-quality framing for the multi-label model.

Ported from wafer-defect-classifier's calibrate.py. Multi-label makes every
piece per-logit, which is simpler than the softmax version:
    - Each of the 8 sigmoid outputs is its own binary probability, so
      temperature is a *vector* T (one scalar per label) fit by minimising
      BCE-with-logits on the val set — no shared softmax coupling.
    - ECE is the binary calibration error per label: bin samples by predicted
      probability p_j; within a bin, |mean(p_j) − observed positive rate|.
      (The multi-class version bins by max-softmax confidence instead.)
    - The decision rule is already per-label (sigmoid > τ), so threshold
      tuning is an independent 1-D grid search per label on the calibrated
      val probabilities — no argmax suppression trick needed.

Cost-of-quality framing (mixed maps change the bookkeeping):
    With superposed defects the escape definition is per-*label*, not
    per-wafer: missing Scratch on an Edge-Ring+Scratch map is an escape of
    the scratch information even though the wafer itself would still be
    flagged (for Edge-Ring) and routed to review. The wafer-level analogue —
    a defective map predicted fully clean, so nothing flags it at all — is
    reported separately as "clean escapes".
      Escape (per label)      = label present, not predicted.  Cost: the
        defect signature goes unrecorded → wrong root-cause routing, and for
        clean escapes possibly a shipped defective wafer.
      False alarm (per label) = label predicted, not present.  Cost: review
        time chasing a signature that isn't there.
    ESCAPE_COST:FALSE_ALARM_COST = 10:1 as in the main repo — conservative
    for high-volume fabs; the framework matters, not the multiplier.

Entry point: python -m wafer_mixed.calibrate [--checkpoint outputs/best.pt]
    Fits T and thresholds on the val split, reports before/after on the test
    split. Writes outputs/calibration.json, outputs/thresholds.json, and
    assets/reliability_diagram.png.

Coupling: the tuned thresholds are fit on temperature-SCALED probabilities.
A consumer of thresholds.json must first divide logits by the per-label
temperatures in calibration.json (thresholds.json embeds them under
"_temperatures" so the file is self-contained).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

from wafer_mixed.config import REPO_ROOT, MixedConfig, build_arg_parser
from wafer_mixed.data import LABEL_NAMES, get_dataloaders
from wafer_mixed.evaluate import collect_logits
from wafer_mixed.metrics import (
    DEFAULT_THRESHOLD,
    exact_match_ratio,
    macro_f1,
    predict_multihot,
)
from wafer_mixed.model import load_checkpoint_model

ESCAPE_COST = 10       # relative units per per-label escape (missed defect signature)
FALSE_ALARM_COST = 1   # relative units per per-label false alarm

# Threshold grid: same range/step as the main repo's tune_thresholds.
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)


# ---------------------------------------------------------------------------
# Per-label temperature scaling
# ---------------------------------------------------------------------------

class PerLabelTemperature(nn.Module):
    """
    One temperature per label, applied to logits before the sigmoid.
    BCE decomposes over labels, so the joint LBFGS fit is equivalent to 8
    independent 1-D fits — done jointly for simplicity.
    """

    def __init__(self, n_labels: int = len(LABEL_NAMES)) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(n_labels) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)

    def fit(self, logits: torch.Tensor, targets: torch.Tensor) -> "PerLabelTemperature":
        """Find T minimising BCE-with-logits on the provided (val) set."""
        bce = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=100)

        def _eval():
            optimizer.zero_grad()
            loss = bce(self.forward(logits), targets)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self

    def values(self) -> np.ndarray:
        """Clamped per-label temperatures as a (n_labels,) array."""
        return self.temperature.detach().clamp(min=0.05).numpy().copy()


def scale_probs(logits: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    """
    Sigmoid probabilities from logits under per-label temperatures. expit is
    the numerically stable sigmoid: a perfectly separated label fits T near
    the clamp floor, and logits/T then overflows a naive exp.
    """
    return expit(logits / temperatures)


# ---------------------------------------------------------------------------
# Per-label ECE and reliability diagram
# ---------------------------------------------------------------------------

def binary_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """
    Binary Expected Calibration Error: bin by predicted probability; within
    each bin, |mean predicted probability − observed positive rate|, weighted
    by bin occupancy. Lower is better; perfect calibration = 0.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p > lo) & (p <= hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def per_label_ece(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> np.ndarray:
    """binary_ece of each label column. Shape (n_labels,)."""
    return np.array(
        [binary_ece(probs[:, j], y_true[:, j], n_bins) for j in range(probs.shape[1])]
    )


def plot_reliability_grid(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    y_true: np.ndarray,
    save_path: Path,
    n_bins: int = 15,
    min_bin_count: int = 10,
) -> None:
    """
    Small-multiple reliability diagram: one panel per label, before/after
    calibration curves. Bins with fewer than min_bin_count samples are
    omitted from the curves — sigmoid outputs pile up at 0 and 1, so the
    near-empty middle bins are 2-3-sample noise that reads as wild
    miscalibration. The printed ECE still weights every occupied bin.
    """
    ink, muted, grid_c = "#0b0b0b", "#898781", "#e1e0d9"
    c_before, c_after = "#2a78d6", "#eb6834"
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    def _curve(p: np.ndarray, y: np.ndarray) -> tuple[list, list]:
        # NaN for omitted bins breaks the line there: segments only connect
        # adjacent occupied bins, instead of drawing invented geometry across
        # the empty middle of a saturated label.
        xs, ys = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p > lo) & (p <= hi)
            if mask.sum() >= min_bin_count:
                xs.append(p[mask].mean())
                ys.append(y[mask].mean())
            else:
                xs.append(np.nan)
                ys.append(np.nan)
        return xs, ys

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for j, (ax, name) in enumerate(zip(axes.ravel(), LABEL_NAMES)):
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=muted)
        for probs, color, label in (
            (probs_before, c_before, "before"),
            (probs_after, c_after, "after"),
        ):
            xs, ys = _curve(probs[:, j], y_true[:, j])
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=2,
                    color=color, label=label)
        eb = binary_ece(probs_before[:, j], y_true[:, j], n_bins)
        ea = binary_ece(probs_after[:, j], y_true[:, j], n_bins)
        support = int(y_true[:, j].sum())
        ax.set_title(f"{name}  (n+={support})\nECE {eb:.4f} → {ea:.4f}",
                     fontsize=10, color=ink)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(color=grid_c, linewidth=0.6)
        ax.tick_params(colors=muted, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(grid_c)
        if j == 0:
            ax.legend(fontsize=9, loc="upper left", frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("Mean predicted probability", fontsize=9, color=muted)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed positive rate", fontsize=9, color=muted)
    fig.suptitle(
        "Per-label reliability, test split — before vs after temperature scaling\n"
        f"(curves omit bins with <{min_bin_count} samples; "
        "ECE weights every occupied bin)",
        fontsize=12, color=ink,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Reliability diagram: {save_path}")


# ---------------------------------------------------------------------------
# Per-label threshold tuning
# ---------------------------------------------------------------------------

def tune_thresholds(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Per-label decision thresholds maximising each label's binary F1 on the
    provided (calibrated val) probabilities. Returns a (n_labels,) vector.

    Ties are broken toward the LOWER threshold: at equal F1 a lower τ trades
    false negatives for false positives, and escapes cost 10× false alarms.
    """
    thresholds = np.full(probs.shape[1], DEFAULT_THRESHOLD)
    for j in range(probs.shape[1]):
        p, y = probs[:, j], y_true[:, j]
        best_f1 = -1.0
        for tau in THRESHOLD_GRID:
            pred = p > tau
            tp = float((pred & (y == 1)).sum())
            fp = float((pred & (y == 0)).sum())
            fn = float((~pred & (y == 1)).sum())
            f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-9)
            if f1 > best_f1:
                best_f1, thresholds[j] = f1, float(tau)
    return thresholds


# ---------------------------------------------------------------------------
# Cost-of-quality
# ---------------------------------------------------------------------------

def cost_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Per-label escapes (label present, not predicted) and false alarms (label
    predicted, not present), the 10:1 cost-weighted error per wafer, and
    wafer-level clean escapes (defective map predicted fully clean — nothing
    flags the wafer at all).
    """
    escapes = ((y_true == 1) & (y_pred == 0)).sum(axis=0)
    false_alarms = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    n = len(y_true)
    return {
        "escapes_per_label": escapes,
        "false_alarms_per_label": false_alarms,
        "escapes_total": int(escapes.sum()),
        "false_alarms_total": int(false_alarms.sum()),
        "clean_escapes": int(((y_true.sum(axis=1) > 0)
                              & (y_pred.sum(axis=1) == 0)).sum()),
        "cost_weighted_error": float(
            (escapes.sum() * ESCAPE_COST + false_alarms.sum() * FALSE_ALARM_COST) / n
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def calibrate(cfg: MixedConfig, checkpoint_path: Path | None = None) -> None:
    if checkpoint_path is None:
        checkpoint_path = cfg.output_dir / "best.pt"

    model, ckpt = load_checkpoint_model(cfg, checkpoint_path)
    print(f"Checkpoint : {checkpoint_path}  (epoch {ckpt.get('epoch', '?')}, "
          f"val macro-F1 {ckpt.get('val_macro_f1', float('nan')):.4f})")

    _, val_loader, test_loader = get_dataloaders(cfg)
    y_val, val_logits = collect_logits(model, val_loader, cfg.device, desc="val logits")
    y_test, test_logits = collect_logits(model, test_loader, cfg.device, desc="test logits")

    # --- fit per-label temperature on val ---
    scaler = PerLabelTemperature().fit(
        torch.from_numpy(val_logits), torch.from_numpy(y_val).float()
    )
    T = scaler.values()

    # --- per-label ECE before/after, on test ---
    probs_before = scale_probs(test_logits, np.ones_like(T))
    probs_after = scale_probs(test_logits, T)
    ece_before = per_label_ece(probs_before, y_test)
    ece_after = per_label_ece(probs_after, y_test)

    # --- per-label thresholds on calibrated val probs ---
    thresholds = tune_thresholds(scale_probs(val_logits, T), y_val)

    print("\n" + "=" * 64)
    print("PER-LABEL CALIBRATION  (T, τ fit on val; ECE on test)")
    print("=" * 64)
    print(f"  {'label':<10} {'T':>7} {'ECE before':>11} {'ECE after':>10} {'τ':>6}")
    for j, name in enumerate(LABEL_NAMES):
        print(f"  {name:<10} {T[j]:>7.3f} {ece_before[j]:>11.4f} "
              f"{ece_after[j]:>10.4f} {thresholds[j]:>6.2f}")
    print(f"  {'mean':<10} {'':>7} {ece_before.mean():>11.4f} "
          f"{ece_after.mean():>10.4f}")

    # --- test-set decision comparison: raw @0.5 vs calibrated @tuned τ ---
    pred_base = predict_multihot(probs_before)                 # DEFAULT_THRESHOLD
    pred_cal = predict_multihot(probs_after, thresholds)
    cost_base = cost_analysis(y_test, pred_base)
    cost_cal = cost_analysis(y_test, pred_cal)

    print("\n" + "=" * 64)
    print(f"TEST SET  raw @ {DEFAULT_THRESHOLD} vs calibrated @ tuned τ")
    print("=" * 64)
    print(f"  {'':<22} {'raw@0.5':>10} {'calibrated':>11}")
    print(f"  {'macro-F1':<22} {macro_f1(y_test, pred_base):>10.4f} "
          f"{macro_f1(y_test, pred_cal):>11.4f}")
    print(f"  {'exact-match':<22} {exact_match_ratio(y_test, pred_base):>10.4f} "
          f"{exact_match_ratio(y_test, pred_cal):>11.4f}")
    for key, label in [
        ("escapes_total", "escapes (label-level)"),
        ("false_alarms_total", "false alarms"),
        ("clean_escapes", "clean escapes (wafer)"),
    ]:
        print(f"  {label:<22} {cost_base[key]:>10d} {cost_cal[key]:>11d}")
    print(f"  {'cost-weighted error':<22} {cost_base['cost_weighted_error']:>10.4f} "
          f"{cost_cal['cost_weighted_error']:>11.4f}"
          f"   (escape {ESCAPE_COST}× : FA {FALSE_ALARM_COST}×, per wafer)")

    print(f"\n  Per-label escapes / false alarms (raw@0.5 → calibrated):")
    print(f"  {'label':<10} {'escapes':>14} {'false alarms':>14}")
    for j, name in enumerate(LABEL_NAMES):
        e0, e1 = cost_base["escapes_per_label"][j], cost_cal["escapes_per_label"][j]
        f0, f1_ = cost_base["false_alarms_per_label"][j], cost_cal["false_alarms_per_label"][j]
        print(f"  {name:<10} {f'{e0} → {e1}':>14} {f'{f0} → {f1_}':>14}")

    # --- artifacts ---
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cal_path = cfg.output_dir / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(
            {
                "temperatures": {n: float(t) for n, t in zip(LABEL_NAMES, T)},
                "ece_before": {n: float(e) for n, e in zip(LABEL_NAMES, ece_before)},
                "ece_after": {n: float(e) for n, e in zip(LABEL_NAMES, ece_after)},
                "ece_mean_before": float(ece_before.mean()),
                "ece_mean_after": float(ece_after.mean()),
            },
            f, indent=2,
        )
    print(f"\nCalibration JSON: {cal_path}")

    thresh_path = cfg.output_dir / "thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(
            {
                # τ were tuned on temperature-scaled probs — ship T alongside
                # so the file can't be applied to raw sigmoids by mistake.
                "_temperatures": {n: float(t) for n, t in zip(LABEL_NAMES, T)},
                "thresholds": {n: float(t) for n, t in zip(LABEL_NAMES, thresholds)},
            },
            f, indent=2,
        )
    print(f"Thresholds JSON : {thresh_path}")

    plot_reliability_grid(
        probs_before, probs_after, y_test,
        REPO_ROOT / "assets" / "reliability_diagram.png",
    )


if __name__ == "__main__":
    parser = build_arg_parser("wafer-mixed calibrate")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to .pt checkpoint (default: outputs/best.pt)")
    args = parser.parse_args()
    cfg = MixedConfig.from_yaml_and_args(args.config, args)
    calibrate(cfg, checkpoint_path=args.checkpoint)
