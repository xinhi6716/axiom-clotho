"""Regression metrics with explicit row and group weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if y.ndim != 1 or pred.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require matching, non-empty finite one-dimensional arrays.")
    residual = pred - y
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "n_rows": len(y),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / total) if len(y) > 1 and total > 0 else None,
        "bias": float(np.mean(residual)),
    }


def grouped_metrics(y_true, y_pred, groups) -> tuple[dict, list[dict]]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    g = np.asarray(groups)
    if len(g) != len(y):
        raise ValueError("groups and targets must have equal lengths.")
    per_group = []
    for group in pd.unique(g):
        mask = g == group
        per_group.append({"group": str(group), **regression_metrics(y[mask], pred[mask])})
    overall = regression_metrics(y, pred)
    overall.update({
        "n_groups": len(per_group),
        "group_mae": float(np.mean([row["mae"] for row in per_group])),
        "group_rmse": float(np.mean([row["rmse"] for row in per_group])),
        "worst_group_mae": float(max(row["mae"] for row in per_group)),
    })
    return overall, per_group
