"""Metric helpers (paper section 6.3).

Threshold-independent metrics (AUROC, AUPRC) plus the calibrated, clinically
meaningful operating-point metrics that Table VI needs:
  - sensitivity at a fixed specificity (default 90%),
  - F1 at that calibrated threshold,
all with bootstrap 95% confidence intervals so reported numbers carry
uncertainty rather than bare point estimates.

The clinically relevant binary event is "abnormal" (SVEB or VEB window) versus
"normal". `to_deterioration_binary` (name kept for backward compatibility) collapses
the 3-class labels/scores into that binary view for the operating-point metrics.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Threshold-independent (multiclass, macro one-vs-rest)
# ---------------------------------------------------------------------------
def auroc_multiclass(y_true, y_score):
    """Macro one-vs-rest AUROC; safe for imbalanced multiclass labels."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    K = y_score.shape[1]
    aucs = []
    for k in range(K):
        y_bin = (y_true == k).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        aucs.append(roc_auc_score(y_bin, y_score[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def auprc_multiclass(y_true, y_score):
    """Macro one-vs-rest AUPRC (average precision). Robust under imbalance."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    K = y_score.shape[1]
    aps = []
    for k in range(K):
        y_bin = (y_true == k).astype(int)
        if y_bin.sum() == 0:
            continue
        aps.append(average_precision_score(y_bin, y_score[:, k]))
    return float(np.mean(aps)) if aps else float("nan")


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))


# ---------------------------------------------------------------------------
# Binary "abnormal vs normal" view for operating-point metrics
# ---------------------------------------------------------------------------
def to_deterioration_binary(y_true, y_score, stable_class: int = 0):
    """Collapse 3-class -> binary abnormal-vs-normal.

    Returns (y_bin, s_bin) where y_bin = 1 if the true class is NOT `stable_class`
    (SVEB or VEB), and s_bin is the predicted abnormality score = 1 - P(normal).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_bin = (y_true != stable_class).astype(int)
    s_bin = 1.0 - y_score[:, stable_class]
    return y_bin, s_bin


# ---------------------------------------------------------------------------
# Calibrated operating-point metrics
# ---------------------------------------------------------------------------
def threshold_at_specificity(y_bin, s_bin, target_specificity: float = 0.90):
    """Smallest threshold whose specificity >= target (maximizes sensitivity
    subject to the specificity floor). Returns (threshold, sensitivity, specificity)."""
    y_bin = np.asarray(y_bin)
    s_bin = np.asarray(s_bin)
    fpr, tpr, thr = roc_curve(y_bin, s_bin)
    spec = 1.0 - fpr
    ok = np.where(spec >= target_specificity)[0]
    if len(ok) == 0:
        return 1.0, 0.0, 1.0
    j = ok[np.argmax(tpr[ok])]
    return float(thr[j]), float(tpr[j]), float(spec[j])


def sensitivity_at_specificity(y_bin, s_bin, target_specificity: float = 0.90):
    """Sensitivity (recall) at a fixed specificity. Standard early-warning metric."""
    _, sens, _ = threshold_at_specificity(y_bin, s_bin, target_specificity)
    return sens


def f1_at_threshold(y_bin, s_bin, threshold: float):
    y_pred = (np.asarray(s_bin) >= threshold).astype(int)
    return float(f1_score(np.asarray(y_bin), y_pred, zero_division=0))


def calibrated_operating_point(y_val, s_val, y_test, s_test,
                               target_specificity: float = 0.90):
    """Calibrate the threshold on validation at a target specificity, then
    report sensitivity and F1 on the test split at that fixed threshold.

    Inputs are the binary abnormal view (use `to_deterioration_binary`).
    Returns dict with threshold, sensitivity, specificity, f1, alarm_rate.
    """
    thr, _, _ = threshold_at_specificity(y_val, s_val, target_specificity)
    y_test = np.asarray(y_test)
    s_test = np.asarray(s_test)
    y_pred = (s_test >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "threshold": float(thr),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": f1_at_threshold(y_test, s_test, thr),
        "alarm_rate": float(y_pred.mean()),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(values, alpha: float = 0.05):
    """Percentile 95% CI over a list of per-seed scalar values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    lo = float(np.percentile(v, 100 * alpha / 2))
    hi = float(np.percentile(v, 100 * (1 - alpha / 2)))
    return (float(v.mean()), lo, hi)


def bootstrap_metric(y_true, y_score, metric_fn, n_boot: int = 1000,
                     alpha: float = 0.05, seed: int = 0):
    """Bootstrap a metric over resampled examples. metric_fn(y_true, y_score)->float."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    point = float(metric_fn(y_true, y_score))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats.append(float(metric_fn(y_true[idx], y_score[idx])))
        except ValueError:
            continue
    if not stats:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def fmt_ci(mean, lo, hi, prec: int = 3):
    return f"{mean:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


# ---------------------------------------------------------------------------
# Legacy helpers (unchanged)
# ---------------------------------------------------------------------------
def bytes_per_decision(rate_bits_per_sample: float, samples_per_window: int) -> float:
    return rate_bits_per_sample * samples_per_window / 8.0


def energy_per_decision_mJ(comp_mJ, crypto_mJ, radio_mJ):
    return float(comp_mJ + crypto_mJ + radio_mJ)
