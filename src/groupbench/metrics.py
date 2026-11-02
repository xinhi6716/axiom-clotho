"""Regression metrics with explicit row and group weighting."""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Integral

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



def _metric_vector(values):
    """Validate source values before a float64 conversion can erase errors."""
    message = (
        "Metrics require non-empty finite real numeric one-dimensional arrays "
        "without booleans, complex values or precision above float64. Integer "
        "values must be exactly representable in float64; rescale or explicitly "
        "convert inputs after reviewing their precision."
    )
    # np.asarray strips a masked array's mask; never score hidden observations.
    if np.ma.isMaskedArray(values):
        raise ValueError(message)
    try:
        # Preserve each scalar in Python sequences: ordinary NumPy inference
        # can already round a large integer when the sequence also has floats.
        original = np.asarray(values) if isinstance(values, (np.ndarray, pd.Series)) else np.asarray(values, dtype=object)
        if original.ndim != 1 or not len(original):
            raise ValueError(message)
        kind = original.dtype.kind
        if kind == "f":
            if original.dtype.itemsize > 8 or not np.isfinite(original).all():
                raise ValueError(message)
            return original.astype(np.float64, copy=False)
        if kind in "iu":
            converted = original.astype(np.float64)
            # Comparison to a float would itself round integers. Compare Python
            # integers instead, only for values outside the contiguous exact range.
            large = (original > 2**53) | (original < -(2**53))
            if any(int(value) != int(result) for value, result in zip(original[large], converted[large])):
                raise ValueError(message)
            return converted
        if kind != "O":
            raise ValueError(message)
        converted = np.empty(len(original), dtype=np.float64)
        for position, value in enumerate(original):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(message)
            if isinstance(value, Integral):
                result = float(value)
                if not np.isfinite(result) or int(result) != int(value):
                    raise ValueError(message)
            elif isinstance(value, (float, np.floating)):
                if isinstance(value, np.floating) and value.dtype.itemsize > 8:
                    raise ValueError(message)
                result = float(value)
                if not np.isfinite(result):
                    raise ValueError(message)
            else:
                raise ValueError(message)
            converted[position] = result
        return converted
    except (TypeError, ValueError, OverflowError):
        # Do not echo potentially private source values through conversion errors.
        raise ValueError(message) from None


def _metric_inputs(y_true, y_pred):
    y, pred = _metric_vector(y_true), _metric_vector(y_pred)
    if pred.shape != y.shape:
        raise ValueError("Metrics require matching, non-empty finite one-dimensional arrays.")
    return y, pred


def regression_metrics(y_true, y_pred) -> dict:
    """Return finite scores, or raise when float64 intermediate arithmetic fails."""
    y, pred = _metric_inputs(y_true, y_pred)
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
    y, pred = _metric_inputs(y_true, y_pred)
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
