"""
Tests for the multi-label metrics module — all against hand-computed values.
"""
import numpy as np
import pytest

from wafer_mixed.metrics import (
    DEFAULT_THRESHOLD,
    exact_match_ratio,
    macro_f1,
    per_label_f1,
    per_label_recall_by_subset,
    predict_multihot,
    spurious_activation_matrix,
    subset_breakdown,
    subset_masks,
)

# 4 samples × 8 labels:
#   row 0: normal (all zero), predicted perfectly
#   row 1: single defect (label 0), predicted perfectly
#   row 2: mix (labels 0+2), label 2 missed
#   row 3: mix (labels 1+3), spurious label 4 added
Y_TRUE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
])
Y_PRED = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0],
])


def test_predict_multihot_threshold():
    probs = np.array([[0.49, 0.51, DEFAULT_THRESHOLD, 1.0]])
    # strict >: a probability exactly at the threshold is NOT a positive
    np.testing.assert_array_equal(predict_multihot(probs), [[0, 1, 0, 1]])
    np.testing.assert_array_equal(predict_multihot(probs, threshold=0.4),
                                  [[1, 1, 1, 1]])
    assert predict_multihot(probs).dtype == np.int64


def test_exact_match_ratio():
    # rows 0 and 1 fully correct, rows 2 and 3 not → 2/4
    assert exact_match_ratio(Y_TRUE, Y_PRED) == 0.5
    assert exact_match_ratio(Y_TRUE, Y_TRUE) == 1.0


def test_per_label_f1_hand_computed():
    f1 = per_label_f1(Y_TRUE, Y_PRED)
    assert f1[0] == 1.0                      # label 0: 2 TP, no errors
    assert f1[1] == 1.0                      # label 1: 1 TP
    assert f1[2] == 0.0                      # label 2: 1 FN, no TP
    assert f1[3] == 1.0                      # label 3: 1 TP
    assert f1[4] == 0.0                      # label 4: 1 FP, no TP
    assert (f1[5:] == 0.0).all()             # never true, never predicted → 0 via zero_division


def test_macro_f1_is_mean_of_per_label():
    assert macro_f1(Y_TRUE, Y_PRED) == pytest.approx(per_label_f1(Y_TRUE, Y_PRED).mean())


def test_subset_masks():
    masks = subset_masks(Y_TRUE)
    np.testing.assert_array_equal(masks["normal"], [True, False, False, False])
    np.testing.assert_array_equal(masks["single"], [False, True, False, False])
    np.testing.assert_array_equal(masks["mixed"],  [False, False, True, True])


def test_subset_breakdown():
    b = subset_breakdown(Y_TRUE, Y_PRED)
    assert b["normal"] == {"n": 1, "exact_match": 1.0, "macro_f1": pytest.approx(np.nan, nan_ok=True)}
    assert b["single"]["n"] == 1 and b["single"]["exact_match"] == 1.0
    assert b["mixed"]["n"] == 2 and b["mixed"]["exact_match"] == 0.0
    # mixed macro-F1 averages only labels with support there (0,1,2,3):
    # f1 = [1, 1, 0, 1] → 0.75. Label 4's FP is excluded (zero support in mixes).
    assert b["mixed"]["macro_f1"] == pytest.approx(0.75)


def test_per_label_recall_by_subset():
    r = per_label_recall_by_subset(Y_TRUE, Y_PRED)
    # single subset: only label 0 present, recalled → 1.0; others NaN
    assert r["single"][0] == 1.0
    assert np.isnan(r["single"][1:]).all()
    # mixed subset: label 0 recalled (row 2), label 2 missed → 0.0
    assert r["mixed"][0] == 1.0
    assert r["mixed"][2] == 0.0
    assert r["mixed"][1] == 1.0 and r["mixed"][3] == 1.0
    assert np.isnan(r["mixed"][4])           # label 4 never true in mixes


def test_spurious_activation_matrix():
    S = spurious_activation_matrix(Y_TRUE, Y_PRED, subset_masks(Y_TRUE)["mixed"])
    assert np.isnan(np.diag(S)).all()        # diagonal undefined by construction
    # In row-3 mix (labels 1+3 true), label 4 was falsely predicted:
    assert S[1, 4] == 1.0 and S[3, 4] == 1.0
    # Label 1 present (row 3) never dragged in label 0:
    assert S[1, 0] == 0.0
    # Conditioning on a label never present in mixes → all NaN row:
    assert np.isnan(S[5, :4]).all()


def test_spurious_matrix_no_mask_uses_all_rows():
    S = spurious_activation_matrix(Y_TRUE, Y_PRED)
    # Label 0 true in rows 1 and 2; label 2 never falsely added there:
    assert S[0, 2] == 0.0
