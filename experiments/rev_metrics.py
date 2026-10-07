"""Metrics (numpy + scikit-learn only). Same definitions as the published
metrics.py, plus per-class diagnostics requested by Reviewer 2 (#4)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve


def auroc_multiclass(y, P):
    y, P = np.asarray(y), np.asarray(P)
    if P.shape[1] == 2:
        return float(roc_auc_score((y == 1).astype(int), P[:, 1]))
    a = [roc_auc_score((y == k).astype(int), P[:, k]) for k in range(P.shape[1])
         if 0 < (y == k).sum() < len(y)]
    return float(np.mean(a)) if a else float("nan")


def auprc_multiclass(y, P):
    y, P = np.asarray(y), np.asarray(P)
    if P.shape[1] == 2:
        return float(average_precision_score((y == 1).astype(int), P[:, 1]))
    a = [average_precision_score((y == k).astype(int), P[:, k]) for k in range(P.shape[1]) if (y == k).sum() > 0]
    return float(np.mean(a)) if a else float("nan")


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro"))


def per_class_auroc(y, P):
    y, P = np.asarray(y), np.asarray(P)
    return [float(roc_auc_score((y == k).astype(int), P[:, k])) if 0 < (y == k).sum() < len(y) else float("nan")
            for k in range(P.shape[1])]


def per_class_recall(y, pred, K):
    y, pred = np.asarray(y), np.asarray(pred)
    return [float((pred[y == k] == k).mean()) if (y == k).sum() else float("nan") for k in range(K)]


def binary_view(y, P):
    """abnormal-vs-normal: score = 1 - P(normal)."""
    return (np.asarray(y) != 0).astype(int), 1.0 - np.asarray(P)[:, 0]


def threshold_at_specificity(yb, sb, target=0.90):
    fpr, tpr, thr = roc_curve(yb, sb)
    ok = np.where(1 - fpr >= target)[0]
    if len(ok) == 0:
        return 1.0
    return float(thr[ok[np.argmax(tpr[ok])]])


def operating_point(y_va, P_va, y_te, P_te, target=0.90):
    yv, sv = binary_view(y_va, P_va); yt, st = binary_view(y_te, P_te)
    thr = threshold_at_specificity(yv, sv, target)
    pred = (st >= thr).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
    return dict(threshold=thr, sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                f1=float(f1_score(yt, pred, zero_division=0)), alarm_rate=float(pred.mean()))


def full_metrics(y_va, P_va, y_te, P_te, n_classes=3, target_spec=0.90):
    pred = np.asarray(P_te).argmax(1)
    res = dict(auroc=auroc_multiclass(y_te, P_te), auprc=auprc_multiclass(y_te, P_te),
               macro_f1=macro_f1(y_te, pred), per_class_auroc=per_class_auroc(y_te, P_te),
               per_class_recall=per_class_recall(y_te, pred, n_classes))
    if n_classes == 3:
        op = operating_point(y_va, P_va, y_te, P_te, target_spec)
        res.update(sens_at_spec=op["sensitivity"], spec_at_op=op["specificity"], f1_at_op=op["f1"],
                   alarm_rate=op["alarm_rate"], threshold=op["threshold"])
    return res


def bootstrap_metric(y, P, fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); y, P = np.asarray(y), np.asarray(P); n = len(y)
    point = float(fn(y, P)); st = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            st.append(float(fn(y[i], P[i])))
        except ValueError:
            pass
    return point, float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))
