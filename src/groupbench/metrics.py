"""Regression metrics with explicit row and group weighting."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

from .data import validate_groups


@contextmanager
def _checked_arithmetic():
    """Fail closed rather than rank models using corrupted float64 scores."""
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            yield
    except FloatingPointError as exc:
        raise ValueError(
            "Metric arithmetic overflowed, underflowed, or became invalid in float64. "
            "Rescale targets and predictions to practical units before evaluation."
        ) from exc


def regression_metrics(y_true, y_pred) -> dict:
    """Return finite scores, or raise when float64 intermediate arithmetic fails."""
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if y.ndim != 1 or pred.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require matching, non-empty finite one-dimensional arrays.")
    with _checked_arithmetic():
        residual = pred - y
        # Detect genuinely undefined R² without a potentially overflowing mean
        # or a roundoff-created variance for identical floating-point targets.
        constant = bool(np.all(y == y[0]))
        # Center in coordinates relative to a representable observed value.
        # Computing the mean at a large shared offset can round away half an
        # ULP and materially corrupt SST even when all final scores are finite.
        if constant:
            total = 0.0
        else:
            anchored = y - y[0]
            total = float(np.sum((anchored - anchored.mean()) ** 2))
        return {
            "n_rows": len(y),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1 - np.sum(residual**2) / total) if not constant else None,
            "bias": float(np.mean(residual)),
        }


def grouped_metrics(y_true, y_pred, groups) -> tuple[dict, list[dict]]:
    """Score rows using the same exact group-identity contract as evaluation."""
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    # Validate before NumPy can promote mixed numeric IDs to a lossy dtype.
    g = validate_groups(groups, len(y))
    per_group = []
    for group in pd.unique(g):
        mask = g == group
        per_group.append({"group": str(group), **regression_metrics(y[mask], pred[mask])})
    overall = regression_metrics(y, pred)
    with _checked_arithmetic():
        overall.update({
            "n_groups": len(per_group),
            "group_mae": float(np.mean([row["mae"] for row in per_group])),
            "group_rmse": float(np.mean([row["rmse"] for row in per_group])),
            "worst_group_mae": float(max(row["mae"] for row in per_group)),
        })
    return overall, per_group
