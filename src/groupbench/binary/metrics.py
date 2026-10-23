"""Fixed binary probability scores with explicit weighting and arithmetic."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import validate_binary_target

EPSILON = float(np.finfo(np.float64).eps)


def metric_contract():
    return {"primary": "group_brier", "brier_scale": "binary_[0,1]",
            "group_weighting": "equal_groups_then_equal_rows_within_group",
            "log_loss": "metric_only_clipped_binary_natural_log",
            "log_loss_epsilon": EPSILON, "threshold": 0.5,
            "threshold_comparison": ">=", "undefined_threshold_metrics": None,
            "positive_loss_rounded_to_zero": "error"}


def _probability_array(values):
    original = np.asarray(values)
    # Only real numeric types representable without narrowing are supported.
    # In particular, do not discard imaginary components or narrow longdouble
    # values outside [0,1] into valid-looking endpoints.
    if (original.dtype.kind not in "iuf"
            or (original.dtype.kind == "f" and original.dtype.itemsize > 8)
            or not np.isfinite(original).all() or np.any(original < 0) or np.any(original > 1)):
        raise ValueError("Probabilities must be real finite numeric values in [0,1] at float64 or lower precision; no narrowing repair is applied.")
    return original.astype(np.float64, copy=False)


def checked_probabilities(probabilities, n_rows, classes):
    labels = validate_binary_target(classes, 2)
    if not np.array_equal(labels, np.array([0, 1])):
        raise ValueError("Model probability classes must be exactly [0, 1] in that order.")
    p = _probability_array(probabilities)
    if (p.shape != (n_rows, 2) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1.0, atol=1e-12, rtol=0)):
        raise ValueError("Model probabilities must be finite n-by-2 values in [0,1] summing to one; no repair is applied.")
    return p


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        value = float(np.mean(values))
    if value == 0.0 and np.any(values > 0):
        raise ValueError("Positive probability loss is below float64 scoring resolution; refusing a zero score.")
    if not np.isfinite(value):
        raise ValueError("Probability metric arithmetic produced a non-finite result.")
    return value


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def binary_metrics(y_true, probability_1):
    p = _probability_array(probability_1)
    y = validate_binary_target(y_true, len(p) if p.ndim else 0, require_both=False)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Binary metrics require finite matching one-dimensional probabilities in [0,1].")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            residual = p - y
            losses = residual * residual
            if np.any((residual != 0) & (losses == 0)):
                raise ValueError("Positive Brier error is below float64 scoring resolution; refusing a zero score.")
            brier = _mean(losses)
            q = np.clip(p, EPSILON, 1 - EPSILON)
            logarithmic = np.empty(len(y), dtype=np.float64)
            positive = y == 1
            logarithmic[positive] = -np.log(q[positive])
            logarithmic[~positive] = -np.log1p(-q[~positive])
            log_loss = _mean(logarithmic)
    except FloatingPointError as exc:
        raise ValueError("Binary metric arithmetic overflowed or became invalid in float64.") from exc
    if not 0 <= brier <= 1 or log_loss < 0:
        raise ValueError("Probability metric result is outside its defined range.")
    pred = p >= 0.5
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    recall, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    return {"n_rows": len(y), "n_true_0": tn + fp, "n_true_1": tp + fn,
            "n_predicted_0": tn + fn, "n_predicted_1": tp + fp,
            "positive_prevalence": float(np.mean(y)), "mean_probability_1": float(np.mean(p)),
            "brier": brier, "log_loss_clipped": log_loss,
            "probability_below_log_epsilon_count": int(np.sum(p < EPSILON)),
            "probability_above_log_one_minus_epsilon_count": int(np.sum(p > 1 - EPSILON)),
            "probability_zero_count": int(np.sum(p == 0)), "probability_one_count": int(np.sum(p == 1)),
            "threshold": 0.5, "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "accuracy": (tp + tn) / len(y), "recall_1": recall, "specificity_0": specificity,
            "precision_1": _ratio(tp, tp + fp),
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None}


def grouped_binary_metrics(y_true, probability_1, groups):
    p = _probability_array(probability_1)
    y = validate_binary_target(y_true, len(p) if p.ndim else 0, require_both=False)
    g = np.asarray(groups)
    if g.ndim != 1 or len(g) != len(y) or pd.isna(g).any():
        raise ValueError("groups must contain one non-missing value per scored row.")
    overall = binary_metrics(y, p)
    per_group = []
    # Group positions once instead of scanning all rows for each group.
    for name, positions in pd.Series(np.arange(len(g))).groupby(g, sort=False).groups.items():
        idx = np.asarray(positions, dtype=int)
        per_group.append({"group": str(name), **binary_metrics(y[idx], p[idx])})
    group_brier = _mean([row["brier"] for row in per_group])
    worst_brier = max(row["brier"] for row in per_group)
    if not np.isfinite(group_brier) or not 0 <= group_brier <= 1 or not 0 <= worst_brier <= 1:
        raise ValueError("Grouped Brier metric is outside [0,1].")
    overall.update(n_groups=len(per_group), group_brier=group_brier,
                   group_log_loss_clipped=_mean([row["log_loss_clipped"] for row in per_group]),
                   worst_group_brier=worst_brier,
                   worst_group_log_loss_clipped=max(row["log_loss_clipped"] for row in per_group))
    return overall, per_group
