"""
metrics.py — multi-label metrics for the 8 basic defect labels.

All functions take multi-hot indicator arrays of shape (N, 8):
    y_true — ground-truth {0,1} labels
    y_pred — hard {0,1} predictions (already thresholded; Phase 1 uses 0.5,
             Phase 3 will supply per-label tuned thresholds)

Metric choices:
    - Per-label F1 / macro-F1 over the 8 labels: the headline, analogous to
      the 9-class macro-F1 in wafer-defect-classifier. Each label is scored
      as its own binary problem, so rare labels (Near-full: 149 maps,
      Random: 866, neither appears in any mix) count equally — flagged as
      small-support in reports.
    - Exact-match ratio: strictest view — every one of the 8 labels correct
      simultaneously. This is the "did we identify the full combination"
      number a fab cares about for routing.
    - Single-vs-mixed breakdown: the study's central question is whether
      superposed defects degrade recognition, so the same metrics are
      reported on the single-defect and mixed-defect subsets separately.
      Macro-F1 is not reported on the normal subset (no positive labels —
      every per-label F1 is degenerate there).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score

# Phase 1 decision rule: fixed sigmoid threshold. Phase 3 replaces this with
# per-label tuned thresholds — change it HERE so train early-stopping and
# evaluation always share the same rule.
DEFAULT_THRESHOLD = 0.5


def predict_multihot(
    probs: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """Hard {0,1} multi-hot predictions from sigmoid probabilities."""
    return (probs > threshold).astype(np.int64)


def per_label_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """F1 of each label as an independent binary problem. Shape (8,)."""
    return f1_score(y_true, y_pred, average=None, zero_division=0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of the 8 per-label F1 scores."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def exact_match_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of samples where all 8 labels are simultaneously correct."""
    return float((y_true == y_pred).all(axis=1).mean())


def subset_masks(y_true: np.ndarray) -> Dict[str, np.ndarray]:
    """Boolean masks by true-label count: normal (0), single (1), mixed (≥2)."""
    counts = y_true.sum(axis=1)
    return {
        "normal": counts == 0,
        "single": counts == 1,
        "mixed": counts >= 2,
    }


def subset_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict]:
    """
    Per-subset metrics: n, exact-match, macro-F1.

    Subset macro-F1 averages only labels with positive support in the subset:
    Near-full and Random never appear in mixes, so including their degenerate
    zero-support F1 (= 0) would drag the mixed-subset number down for no
    informational gain. False positives on absent labels are therefore not
    reflected here — the spurious-activation matrix captures those.
    macro-F1 is NaN for the normal subset (no label has support there).
    """
    out: Dict[str, Dict] = {}
    for name, mask in subset_masks(y_true).items():
        entry = {"n": int(mask.sum()), "exact_match": np.nan, "macro_f1": np.nan}
        if entry["n"]:
            t, p = y_true[mask], y_pred[mask]
            entry["exact_match"] = exact_match_ratio(t, p)
            present = t.sum(axis=0) > 0
            if present.any():
                entry["macro_f1"] = float(per_label_f1(t, p)[present].mean())
        out[name] = entry
    return out


def per_label_recall_by_subset(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Recall of each label computed separately on single-defect and mixed-defect
    samples: does label X get harder to detect when superposed with others?
    NaN where a label has no positives in the subset (Near-full and Random
    never appear in mixes).
    """
    masks = subset_masks(y_true)
    out: Dict[str, np.ndarray] = {}
    for name in ("single", "mixed"):
        t, p = y_true[masks[name]], y_pred[masks[name]]
        support = t.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            recall = (t * p).sum(axis=0) / support
        recall[support == 0] = np.nan
        out[name] = recall
    return out


def spurious_activation_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    S[i, j] = P(label j predicted | label i true AND label j not true),
    i.e. how often the presence of defect i drags in a false prediction of j.
    This is the multi-label analogue of a confusion matrix: run it on the
    mixed subset to see which basic labels get confused inside mixes.
    Diagonal is NaN (i true and i not-true is contradictory); NaN also where
    the conditioning set is empty.
    """
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    n_labels = y_true.shape[1]
    S = np.full((n_labels, n_labels), np.nan)
    for i in range(n_labels):
        cond_i = y_true[:, i] == 1
        for j in range(n_labels):
            if j == i:
                continue
            cond = cond_i & (y_true[:, j] == 0)
            if cond.sum():
                S[i, j] = y_pred[cond, j].mean()
    return S
