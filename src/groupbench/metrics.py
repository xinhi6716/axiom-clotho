"""Regression metrics with explicit row and group weighting."""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Integral
from fractions import Fraction
import math

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
    # Request positional arrays directly, avoiding a pandas Index per group.
    # Preserve encounter order and row order within each normalized group.
    for group, positions in pd.Series(np.arange(len(g))).groupby(g, sort=False).indices.items():
        idx = np.asarray(positions, dtype=int)
        per_group.append({"group": str(group), **regression_metrics(y[idx], pred[idx])})
    overall = regression_metrics(y, pred)
    with _checked_arithmetic():
        overall.update({
            "n_groups": len(per_group),
            "group_mae": float(np.mean([row["mae"] for row in per_group])),
            "group_rmse": float(np.mean([row["rmse"] for row in per_group])),
            "worst_group_mae": float(max(row["mae"] for row in per_group)),
        })
    return overall, per_group


def paired_absolute_error_difference(prediction_a, prediction_b, target):
    """Exact B-minus-A absolute-error contrast of validated binary64 inputs.

    Callers must validate conversion/precision before calling. This preserves
    differences lost by subtracting rounded errors; it cannot recover decimal
    precision lost when inputs became floats. Keep the result exact to aggregate.
    """
    a, b, y = (Fraction.from_float(float(value))
               for value in (prediction_a, prediction_b, target))
    return abs(b - y) - abs(a - y)


def contrast_float(value):
    """Round one exact contrast for display, rejecting false zero or infinity."""
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("paired contrast outside supported representable range") from None
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ValueError("paired contrast outside supported representable range")
    return result


def _bounded_paired_sequence(sequence):
    return type(sequence) in (tuple, list) and 1 <= len(sequence) <= 20_000


def _paired_label(value):
    return (type(value) is str and 0 < len(value) <= 256
            and bool(value.strip()) and value == value.strip())


def _exact_paired_number(value):
    if type(value) not in (int, float):
        raise ValueError("Loss inputs and weights must be real finite numbers.")
    try:
        converted = float(value)
    except OverflowError:
        raise ValueError("Loss input or weight is outside binary64 range.") from None
    if not math.isfinite(converted) or (type(value) is int and int(converted) != value):
        raise ValueError("Loss input or weight must be representable as binary64.")
    return Fraction.from_float(converted)


def scheduled_phase_paired_mae(rows, *, expected_roster, expected_cells, phase_weights):
    """Compare aligned forecasts on a predeclared device × phase panel.

    Each row is (row_id, device, phase, target, forecast_a, forecast_b).
    Pair by row ID before calling; each row carries both forecasts, so no
    positional join occurs here. Supply an independently frozen roster of
    (row_id, device, phase) tuples and a schedule of unique (device, phase)
    pairs with one positive weight per phase. Row order is irrelevant.
    Every device must have every declared phase, with at least one observed
    row per cell, and no extras. IDs are nonblank strings; numeric inputs
    are built-in ints exactly representable as binary64, or finite floats.
    Each ordered container is bounded to 20,000 entries.

    The fixed-phase estimand averages loss within cell, weights phases by
    their declared relative weights, and then averages across devices.
    This finite-roster diagnostic does not infer missing cells, establish
    physical independence, or verify when the supplied roster was frozen.
    Reconstructing the expected roster from scored rows cannot detect omission.
    """
    if not all(map(_bounded_paired_sequence, (rows, expected_roster, expected_cells))):
        raise ValueError("Rows, expected roster and expected cells must be nonempty bounded lists or tuples.")
    if type(phase_weights) is not dict or not 1 <= len(phase_weights) <= 20_000:
        raise ValueError("Supply an explicit bounded phase-weight mapping.")
    weights = {}
    for phase, value in phase_weights.items():
        if not _paired_label(phase):
            raise ValueError("Invalid scheduled phase identifier.")
        weights[phase] = _exact_paired_number(value)
        if weights[phase] <= 0:
            raise ValueError("Phase weights must be strictly positive.")
    expected, devices = set(), set()
    for cell in expected_cells:
        if type(cell) is not tuple or len(cell) != 2 or not all(map(_paired_label, cell)):
            raise ValueError("Expected cells must be (device, phase) tuples with valid identifiers.")
        if cell in expected or cell[1] not in weights:
            raise ValueError("Expected cells must be unique and use declared phases.")
        expected.add(cell)
        devices.add(cell[0])
    if len(expected) != len(devices) * len(weights):
        raise ValueError("Expected cells must declare every phase for every device.")

    roster = {}
    for entry in expected_roster:
        if type(entry) is not tuple or len(entry) != 3 or not all(map(_paired_label, entry)):
            raise ValueError("Expected roster entries must be (row ID, device, phase) tuples.")
        row_id, device, phase = entry
        if row_id in roster:
            raise ValueError("Duplicate expected roster row ID.")
        if (device, phase) not in expected:
            raise ValueError("Expected roster entry outside the declared schedule.")
        roster[row_id] = (device, phase)

    cells, ids = {}, set()
    for row in rows:
        if type(row) is not tuple or len(row) != 6 or not all(map(_paired_label, row[:3])):
            raise ValueError("Each row must provide a row ID, device, phase, target, and both forecasts.")
        row_id, device, phase, y, a, b = row
        if row_id in ids:
            raise ValueError("Duplicate paired row ID.")
        ids.add(row_id)
        if (device, phase) not in expected:
            raise ValueError("Observed row outside the declared schedule.")
        if row_id not in roster:
            raise ValueError("Unexpected paired row ID outside the expected roster.")
        if roster[row_id] != (device, phase):
            raise ValueError("Paired row ID is rebound to a different device or phase.")
        y, a, b = map(_exact_paired_number, (y, a, b))
        state = cells.setdefault((device, phase), [0, Fraction(), Fraction()])
        state[0] += 1
        state[1] += abs(a - y)
        state[2] += abs(b - y)
    if len(cells) != len(expected):
        raise ValueError(f"Incomplete scheduled roster: {len(expected) - len(cells)} missing cell(s).")

    if len(ids) != len(roster):
        raise ValueError(f"Incomplete expected roster: {len(roster) - len(ids)} missing row(s).")

    by_device, micro, standard = {}, [Fraction(), Fraction()], [Fraction(), Fraction()]
    weight_sum = sum(weights.values())
    for (device, phase), (n, a, b) in cells.items():
        device_sum = by_device.setdefault(device, [0, Fraction(), Fraction()])
        device_sum[0] += n
        for j, loss in enumerate((a, b)):
            micro[j] += loss
            device_sum[j + 1] += loss
            standard[j] += weights[phase] / weight_sum * (loss / n)

    def summary(a, b):
        difference = b - a
        return {"mae_a": contrast_float(a), "mae_b": contrast_float(b),
                "mae_b_minus_a": contrast_float(difference),
                "better": "A" if difference > 0 else "B" if difference < 0 else "tie"}

    return {
        "n_rows": len(rows), "n_devices": len(devices), "n_scheduled_cells": len(expected),
        "micro": summary(*(total / len(rows) for total in micro)),
        "equal_device": summary(*(sum(v[j] / v[0] for v in by_device.values()) / len(devices)
                                   for j in (1, 2))),
        "fixed_phase_equal_device": summary(*(total / len(devices) for total in standard)),
    }


def paired_parent_mae_influence(rows, *, expected_roster):
    """Describe same-roster MAE and the effect of omitting one declared parent.

    Supply (row_id, device, parent, target, forecast_a, forecast_b) tuples
    and an independently frozen (row_id, device, parent) roster. IDs are exact
    nonblank Unicode scalar strings; use source-qualified canonical IDs.
    Rows, devices within parent, then parents each receive equal weight at
    their own level. All bounds are descriptive, not intervals or resampling.
    Parent labels do not verify independence or resolve aliases. The per_parent
    records return supplied parent labels verbatim; use pseudonymous labels
    when sharing the report.
    """
    def identifier(value):
        return _paired_label(value) and not any(0xD800 <= ord(char) <= 0xDFFF for char in value)

    if not all(map(_bounded_paired_sequence, (rows, expected_roster))):
        raise ValueError("Rows and expected roster must be nonempty bounded lists or tuples with valid Unicode identifiers.")
    roster, device_parent = {}, {}
    for entry in expected_roster:
        if type(entry) is not tuple or len(entry) != 3 or not all(map(identifier, entry)):
            raise ValueError("Expected roster entries must be (row ID, device, parent) tuples with valid Unicode identifiers.")
        row, device, parent = entry
        if row in roster:
            raise ValueError("Duplicate expected roster row ID.")
        if device_parent.setdefault(device, parent) != parent:
            raise ValueError("Each declared device must belong to exactly one parent.")
        roster[row] = (device, parent)

    seen, devices = set(), {}
    for row in rows:
        if type(row) is not tuple or len(row) != 6 or not all(map(identifier, row[:3])):
            raise ValueError("Each row must contain an ID, device, parent, target and both forecasts, with valid Unicode identifiers.")
        row_id, device, parent, y, a, b = row
        if row_id in seen:
            raise ValueError("Duplicate paired row ID.")
        if row_id not in roster:
            raise ValueError("Unexpected paired row ID outside the expected roster.")
        if roster[row_id] != (device, parent):
            raise ValueError("Paired row ID is rebound to a different device or parent.")
        seen.add(row_id)
        y, a, b = map(_exact_paired_number, (y, a, b))
        loss = devices.setdefault(device, [0, Fraction(), Fraction()])
        loss[0] += 1
        loss[1] += abs(a - y)
        loss[2] += abs(b - y)
    if len(seen) != len(roster):
        raise ValueError(f"Incomplete expected roster: {len(roster) - len(seen)} missing row(s).")

    parents = {}
    for device, (n, loss_a, loss_b) in devices.items():
        pool = parents.setdefault(device_parent[device], [0, Fraction(), Fraction()])
        pool[0] += 1
        pool[1] += loss_a / n
        pool[2] += loss_b / n
    k = len(parents)
    total_a, total_b, effects, details = Fraction(), Fraction(), [], []
    for parent, (n, loss_a, loss_b) in sorted(parents.items()):
        a, b = loss_a / n, loss_b / n
        total_a += a
        total_b += b
        effects.append(b - a)
        details.append({"parent": parent, "n_devices": n, "mae_b_minus_a": contrast_float(b - a)})
    effect_sum = total_b - total_a
    loo = None
    if k > 1:
        loo = {"min": contrast_float((effect_sum - max(effects)) / (k - 1)),
               "max": contrast_float((effect_sum - min(effects)) / (k - 1))}
    return {"n_rows": len(rows), "n_devices": len(devices), "n_declared_parents": k,
            "mae_a": contrast_float(total_a / k), "mae_b": contrast_float(total_b / k),
            "mae_b_minus_a": contrast_float(effect_sum / k), "per_parent": details,
            "leave_one_parent_out_mae_b_minus_a": loo}
