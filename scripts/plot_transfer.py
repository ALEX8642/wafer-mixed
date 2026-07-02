"""
plot_transfer.py — Phase 2 story figure: test macro-F1 vs training-set size,
one line per init arm, min–max seed band where multiple seeds ran.

Reads outputs/transfer/results.csv (written by scripts/transfer_study.py)
and writes assets/transfer_curves.png. Colors are the first three slots of a
CVD-validated categorical palette in fixed order; per its relief rule the
low-contrast slots get direct labels, and docs/TRANSFER.md carries the full
numbers table.

Run:
    python scripts/plot_transfer.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafer_mixed.config import REPO_ROOT

RESULTS = REPO_ROOT / "outputs" / "transfer" / "results.csv"
OUT_PNG = REPO_ROOT / "assets" / "transfer_curves.png"

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
ARM_STYLE = {  # fixed slot order — identity follows the arm, never rank
    "scratch":    ("#2a78d6", "random init (scratch)"),
    "supervised": ("#1baf7a", "WM-811K supervised init"),
    "simclr":     ("#eda100", "wafer-ssl SimCLR init"),
}


def load_results(path: Path) -> dict[str, dict[int, list[float]]]:
    """arm → {n_train: [test macro-F1 per seed]}"""
    by_arm: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            by_arm[r["arm"]][int(r["n_train"])].append(float(r["test_macro_f1"]))
    return by_arm


def main() -> None:
    if not RESULTS.exists():
        sys.exit(f"{RESULTS} not found — run scripts/transfer_study.py first.")
    by_arm = load_results(RESULTS)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for arm, (color, label) in ARM_STYLE.items():
        if arm not in by_arm:
            continue
        sizes = sorted(by_arm[arm])
        mean = [float(np.mean(by_arm[arm][n])) for n in sizes]
        lo   = [float(np.min(by_arm[arm][n])) for n in sizes]
        hi   = [float(np.max(by_arm[arm][n])) for n in sizes]
        ax.fill_between(sizes, lo, hi, color=color, alpha=0.10, linewidth=0)
        ax.plot(sizes, mean, color=color, linewidth=2,
                solid_capstyle="round", solid_joinstyle="round", label=label,
                marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3)
        # Direct label at the low-data end, where the arms separate most.
        ax.annotate(label, (sizes[0], mean[0]), textcoords="offset points",
                    xytext=(10, -4 if arm == "scratch" else 8), fontsize=9,
                    color=INK_2)

    ax.set_xscale("log")
    all_sizes = sorted({n for d in by_arm.values() for n in d})
    ax.set_xticks(all_sizes)
    ax.set_xticklabels(
        [f"{n:,}\n({n / max(all_sizes):.0%})" for n in all_sizes], fontsize=9
    )
    ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.set_xlabel("Training maps (fraction of full train split)", color=INK_2)
    ax.set_ylabel("Test macro-F1 (8 labels @ 0.5)", color=INK_2)
    ax.set_title("Does pretraining help MixedWM38? Test macro-F1 vs training-set size",
                 color=INK, fontsize=12, pad=12)

    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)

    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.text(0.01, 0.01,
             "Band = min–max over 3 seeds (42/43/44) at 10% and 1%; single seed (42) at 100%. "
             "Identical subsample per (fraction, seed) across arms.",
             fontsize=7.5, color=MUTED)

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close()
    print(f"Figure written: {OUT_PNG}")


if __name__ == "__main__":
    main()
