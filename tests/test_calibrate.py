"""
Tests for the calibration module — temperature recovery on synthetic
miscalibrated logits, hand-computed ECE and cost counts, threshold tuning
against a known optimum, and the per-label threshold vector path through
predict_multihot.
"""
import numpy as np
import torch

from wafer_mixed.calibrate import (
    ESCAPE_COST,
    FALSE_ALARM_COST,
    PerLabelTemperature,
    binary_ece,
    cost_analysis,
    per_label_ece,
    scale_probs,
    tune_thresholds,
)
from wafer_mixed.metrics import predict_multihot

N_LABELS = 8


def _synthetic_overconfident(n: int = 4000, t_true: float = 3.0):
    """
    Labels drawn from Bernoulli(sigmoid(z)) with z ~ N(0, 2), logits reported
    as t_true * z: perfectly calibrated only after dividing by t_true.
    Fresh seeded generator per call — no shared state, so every test sees the
    same data regardless of which other tests ran first.
    """
    rng = np.random.default_rng(42)
    z = rng.normal(0.0, 2.0, size=(n, N_LABELS))
    y = (rng.uniform(size=z.shape) < 1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    return (t_true * z).astype(np.float32), y


def test_temperature_recovers_true_scale():
    logits, y = _synthetic_overconfident(t_true=3.0)
    scaler = PerLabelTemperature().fit(torch.from_numpy(logits), torch.from_numpy(y))
    T = scaler.values()
    assert T.shape == (N_LABELS,)
    # Each label's fit should land near the generating temperature.
    np.testing.assert_allclose(T, 3.0, rtol=0.15)


def test_temperature_fit_reduces_bce_and_ece():
    logits, y = _synthetic_overconfident(t_true=3.0)
    scaler = PerLabelTemperature().fit(torch.from_numpy(logits), torch.from_numpy(y))
    T = scaler.values()

    bce = torch.nn.BCEWithLogitsLoss()
    lt, yt = torch.from_numpy(logits), torch.from_numpy(y)
    assert bce(lt / torch.from_numpy(T), yt) < bce(lt, yt)

    ece_before = per_label_ece(scale_probs(logits, np.ones(N_LABELS)), y)
    ece_after = per_label_ece(scale_probs(logits, T), y)
    assert (ece_after < ece_before).all()


def test_binary_ece_hand_computed():
    # 4 samples in two occupied bins (15 bins of width 1/15):
    #   p=0.10, 0.12 → bin (0.0667, 0.1333]: mean p 0.11, positive rate 0.5
    #   p=0.90, 0.92 → bin (0.8667, 0.9333]: mean p 0.91, positive rate 1.0
    # ECE = 0.5*|0.11-0.5| + 0.5*|0.91-1.0| = 0.195 + 0.045 = 0.24
    p = np.array([0.10, 0.12, 0.90, 0.92])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    assert np.isclose(binary_ece(p, y, n_bins=15), 0.24)


def test_binary_ece_perfect_calibration_is_zero():
    # Within each occupied bin, mean predicted probability equals the
    # observed positive rate exactly.
    p = np.array([0.5, 0.5, 0.5, 0.5])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert binary_ece(p, y) == 0.0


def test_tune_thresholds_finds_separating_boundary():
    # Label 0: positives at p ≥ 0.30, negatives at p ≤ 0.20 — any τ in
    # [0.20, 0.29] gives F1=1.0; the ascending grid with strict improvement
    # keeps the lowest such τ (documented escape-favouring tie-break).
    # Label 1: separation at high p.
    probs = np.zeros((6, N_LABELS))
    y = np.zeros((6, N_LABELS))
    probs[:, 0] = [0.05, 0.10, 0.20, 0.30, 0.55, 0.90]
    y[:, 0] = [0, 0, 0, 1, 1, 1]
    probs[:, 1] = [0.1, 0.2, 0.7, 0.8, 0.9, 0.95]
    y[:, 1] = [0, 0, 0, 0, 1, 1]
    taus = tune_thresholds(probs, y)
    assert np.isclose(taus[0], 0.20)
    assert np.isclose(taus[1], 0.80)
    # Untouched labels (no positives anywhere) keep a grid value, not NaN.
    assert np.isfinite(taus).all()


def test_tune_thresholds_beats_default_when_default_is_wrong():
    # Positives live at p ∈ [0.25, 0.45]: the 0.5 default catches none of
    # them (F1=0); tuning must find a τ below 0.25.
    probs = np.zeros((8, N_LABELS))
    y = np.zeros((8, N_LABELS))
    probs[:, 2] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45]
    y[:, 2] = [0, 0, 0, 0, 1, 1, 1, 1]
    taus = tune_thresholds(probs, y)
    assert taus[2] < 0.25
    pred = predict_multihot(probs, taus)
    np.testing.assert_array_equal(pred[:, 2], y[:, 2])


def test_predict_multihot_per_label_vector():
    probs = np.array([[0.4, 0.6, 0.4, 0.6, 0.5, 0.5, 0.5, 0.5]])
    taus = np.array([0.3, 0.7, 0.5, 0.5, 0.49, 0.51, 0.5, 0.5])
    # strict >: column crosses only if p exceeds its own τ
    np.testing.assert_array_equal(
        predict_multihot(probs, taus), [[1, 0, 0, 1, 1, 0, 0, 0]]
    )


def test_cost_analysis_hand_computed():
    y_true = np.zeros((4, N_LABELS), dtype=np.int64)
    y_pred = np.zeros((4, N_LABELS), dtype=np.int64)
    # row 0: label 0 present, missed entirely → 1 escape, 1 clean escape
    y_true[0, 0] = 1
    # row 1: labels 1+2 present, only 1 predicted → 1 escape, wafer still flagged
    y_true[1, [1, 2]] = 1
    y_pred[1, 1] = 1
    # row 2: clean wafer, label 3 predicted → 1 false alarm
    y_pred[2, 3] = 1
    # row 3: perfect clean wafer
    c = cost_analysis(y_true, y_pred)
    assert c["escapes_total"] == 2
    assert c["false_alarms_total"] == 1
    assert c["clean_escapes"] == 1
    assert c["escapes_per_label"][0] == 1
    assert c["escapes_per_label"][2] == 1
    assert c["false_alarms_per_label"][3] == 1
    expected = (2 * ESCAPE_COST + 1 * FALSE_ALARM_COST) / 4
    assert np.isclose(c["cost_weighted_error"], expected)


def test_scale_probs_matches_torch_sigmoid():
    logits = np.random.default_rng(7).normal(size=(16, N_LABELS)).astype(np.float32)
    T = np.full(N_LABELS, 2.0)
    expected = torch.sigmoid(torch.from_numpy(logits) / 2.0).numpy()
    np.testing.assert_allclose(scale_probs(logits, T), expected, atol=1e-6)
