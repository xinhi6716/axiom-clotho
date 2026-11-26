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
        "without booleans, timedelta or complex values, or precision above float64. Integer "
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
            # NumPy durations are Integral scalars, but converting them would
            # score their raw unit-dependent tick count rather than a number.
            if isinstance(value, (bool, np.bool_, np.timedelta64)):
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
    return _regression_metrics_from_validated(*_metric_inputs(y_true, y_pred))


def _regression_metrics_from_validated(y, pred):
    """Score nonempty matching float64 vectors already checked by the caller."""
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
        mae = float(np.mean(np.abs(residual)))
        squared_residual = residual**2
        return {
            "n_rows": len(y),
            "mae": mae,
            "rmse": float(np.sqrt(np.mean(squared_residual))),
            "r2": float(1 - np.sum(squared_residual) / total) if not constant else None,
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
        score = _regression_metrics_from_validated(y[idx], pred[idx])
        per_group.append({"group": str(group), **score})
    return _add_group_aggregates(_regression_metrics_from_validated(y, pred), per_group), per_group


def _add_group_aggregates(row_metrics, per_group):
    """Add equal-group summaries in the supplied group encounter order."""
    with _checked_arithmetic():
        row_metrics.update({
            "n_groups": len(per_group),
            "group_mae": float(np.mean([row["mae"] for row in per_group])),
            "group_rmse": float(np.mean([row["rmse"] for row in per_group])),
            "worst_group_mae": float(max(row["mae"] for row in per_group)),
        })
    return row_metrics


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


def _validated_paired_float(value):
    if type(value) not in (int, float):
        raise ValueError("Loss inputs and weights must be real finite numbers.")
    try:
        converted = float(value)
    except OverflowError:
        raise ValueError("Loss input or weight is outside binary64 range.") from None
    if not math.isfinite(converted) or (type(value) is int and int(converted) != value):
        raise ValueError("Loss input or weight must be representable as binary64.")
    return converted


def _exact_paired_number(value):
    return Fraction.from_float(_validated_paired_float(value))


# Every finite binary64 value is an integer multiple of this quantum.
_BINARY64_QUANTUM_EXP = 1074
_BINARY64_QUANTUM_DENOMINATOR = 1 << _BINARY64_QUANTUM_EXP


def _paired_binary64_units(value):
    numerator, denominator = _validated_paired_float(value).as_integer_ratio()
    return numerator << (_BINARY64_QUANTUM_EXP + 1 - denominator.bit_length())


def _paired_roster_id(value):
    return _paired_label(value) and not any(0xD800 <= ord(c) <= 0xDFFF for c in value)


def paired_mae_missing_bounds(rows, *, expected_roster, truth_bounds=None):
    """Bound the full equal-row B-minus-A MAE on a frozen paired roster.

    Supply (row_id, target_or_none, forecast_a, forecast_b) tuples and an
    independently declared, nonempty roster of unique row IDs. Row and roster
    order are irrelevant; both forecasts are required even if truth is None.
    Numeric inputs follow the finite built-in binary64 contract. Exact means
    and endpoints are returned as Fraction values for those input numbers,
    not as rounded binary64 values. The available-case mean is None when
    no target is observed.

    Optionally supply a dict mapping missing row IDs to (lower, upper)
    closed truth limits in the same target units. None on either side
    denotes an unbounded side. Unlisted missing rows remain unrestricted.
    A bound on an observed row is rejected. A point bound does not count
    as an observation.

    The B-minus-A loss difference is monotone or constant in truth, so
    evaluate it at the declared endpoints (or its constant tails). For an
    unrestricted row the range is [−|a−b|, |a−b|]. Averaging row endpoints
    is sharp under the Cartesian product of the supplied intervals; with
    extra cross-row constraints it is only an outer bound. This is not
    imputation, a confidence interval, or physical independence evidence.
    """
    if not _bounded_paired_sequence(rows):
        raise ValueError("Rows must be a nonempty bounded list or tuple.")
    if (type(expected_roster) not in (list, tuple, set, frozenset)
            or not 1 <= len(expected_roster) <= 20_000):
        raise ValueError("Expected roster must be a nonempty bounded collection of row IDs.")

    expected = set()
    for row_id in expected_roster:
        if not _paired_roster_id(row_id):
            raise ValueError("Invalid expected roster row ID.")
        if row_id in expected:
            raise ValueError("Duplicate expected roster row ID.")
        expected.add(row_id)

    bounds = {}
    if truth_bounds is not None:
        if type(truth_bounds) is not dict or len(truth_bounds) > len(expected):
            raise ValueError("Truth bounds must be a bounded dict keyed by roster row ID.")
        for row_id, limits in truth_bounds.items():
            if not _paired_roster_id(row_id) or row_id not in expected:
                raise ValueError("Truth bounds must identify only expected roster rows.")
            if type(limits) is not tuple or len(limits) != 2:
                raise ValueError("Each truth bound must be a (lower, upper) tuple.")
            lower, upper = (None if value is None else _paired_binary64_units(value)
                            for value in limits)
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("Truth bound lower endpoint must not exceed its upper endpoint.")
            bounds[row_id] = lower, upper

    ids = set()
    observed_count = 0
    # Accumulate dyadic sums in exact integer quantum units; construct only
    # the final ratios rather than repeatedly reducing row-level Fractions.
    observed_sum = unrestricted_radius = missing_lower = missing_upper = 0
    for row in rows:
        if type(row) is not tuple or len(row) != 4 or not _paired_roster_id(row[0]):
            raise ValueError("Each row must provide a row ID, target or None, and both forecasts.")
        row_id, target, pred_a, pred_b = row
        if row_id in ids:
            raise ValueError("Duplicate paired row ID.")
        if row_id not in expected:
            raise ValueError("Unexpected paired row ID outside the expected roster.")
        ids.add(row_id)
        a, b = map(_paired_binary64_units, (pred_a, pred_b))
        if target is None:
            limits = bounds.get(row_id)
            if limits is None:
                unrestricted_radius += abs(a - b)
                continue
            lower, upper = limits
            left = b - a if lower is None else abs(b - lower) - abs(a - lower)
            right = a - b if upper is None else abs(b - upper) - abs(a - upper)
            missing_lower += min(left, right)
            missing_upper += max(left, right)
        else:
            if row_id in bounds:
                raise ValueError("Truth bounds cannot be supplied for an observed target.")
            y = _paired_binary64_units(target)
            observed_count += 1
            observed_sum += abs(b - y) - abs(a - y)
    if ids != expected:
        raise ValueError("Incomplete expected roster: missing paired rows.")
    total = len(expected)
    def mean(units, count):
        return Fraction(units, count * _BINARY64_QUANTUM_DENOMINATOR)

    return {
        "n_rows": total,
        "n_observed": observed_count,
        "n_missing": total - observed_count,
        "available_mae_b_minus_a": mean(observed_sum, observed_count) if observed_count else None,
        "full_mae_b_minus_a_lower": mean(observed_sum + missing_lower - unrestricted_radius, total),
        "full_mae_b_minus_a_upper": mean(observed_sum + missing_upper + unrestricted_radius, total),
    }


def paired_mae_parent_bounds(rows, *, expected_roster, truth_bounds=None):
    """Bound paired B-minus-A MAE with one equal weight per declared parent.

    Supply row tuples as in paired_mae_missing_bounds and an independent
    roster of (row_id, parent_id) tuples. Every row ID must occur once in
    both inputs; membership is not inferred from scoring rows or labels.
    Bounds use the same closed or unbounded interval contract. The output
    keeps every parent, including ones with no observed targets. The equal-
    parent observed-only result is None unless every parent is represented.
    This is a descriptive sensitivity analysis, not a test of independence
    or a claim that declared parents are physically disjoint.
    """
    if not _bounded_paired_sequence(rows) or not _bounded_paired_sequence(expected_roster):
        raise ValueError("Rows and parent roster must be nonempty bounded lists or tuples.")

    parent_by_row, roster_by_parent = {}, {}
    for entry in expected_roster:
        if type(entry) is not tuple or len(entry) != 2 or not all(map(_paired_roster_id, entry)):
            raise ValueError("Parent roster entries must be (row ID, parent ID) tuples with valid IDs.")
        row_id, parent_id = entry
        if row_id in parent_by_row:
            raise ValueError("Duplicate row ID in the parent roster.")
        parent_by_row[row_id] = parent_id
        roster_by_parent.setdefault(parent_id, []).append(row_id)

    rows_by_parent, seen = {}, set()
    for row in rows:
        if type(row) is not tuple or len(row) != 4 or not _paired_roster_id(row[0]):
            raise ValueError("Each row must provide a row ID, target or None, and both forecasts.")
        row_id = row[0]
        if row_id in seen:
            raise ValueError("Duplicate paired row ID.")
        if row_id not in parent_by_row:
            raise ValueError("Unexpected paired row ID outside the parent roster.")
        seen.add(row_id)
        rows_by_parent.setdefault(parent_by_row[row_id], []).append(row)
    if len(seen) != len(parent_by_row):
        raise ValueError("Incomplete parent roster: missing paired rows.")

    bounds_by_parent = {}
    if truth_bounds is not None:
        if type(truth_bounds) is not dict or len(truth_bounds) > len(parent_by_row):
            raise ValueError("Truth bounds must be a bounded dict keyed by roster row ID.")
        for row_id, limits in truth_bounds.items():
            if not _paired_roster_id(row_id) or row_id not in parent_by_row:
                raise ValueError("Truth bounds must identify only expected roster rows.")
            bounds_by_parent.setdefault(parent_by_row[row_id], {})[row_id] = limits

    per_parent = []
    n_observed = n_with_observed = 0
    available = lower = upper = Fraction()
    for parent_id in sorted(roster_by_parent):
        score = paired_mae_missing_bounds(
            rows_by_parent[parent_id], expected_roster=roster_by_parent[parent_id],
            truth_bounds=bounds_by_parent.get(parent_id),
        )
        per_parent.append({"parent": parent_id, **score})
        lower += score["full_mae_b_minus_a_lower"]
        upper += score["full_mae_b_minus_a_upper"]
        n_observed += score["n_observed"]
        if score["n_observed"]:
            n_with_observed += 1
            available += score["available_mae_b_minus_a"]

    n_parents = len(per_parent)
    return {
        "n_rows": len(rows), "n_observed": n_observed, "n_missing": len(rows) - n_observed,
        "n_parents": n_parents, "n_parents_with_observations": n_with_observed,
        "available_equal_parent_mae_b_minus_a": available / n_parents if n_with_observed == n_parents else None,
        "full_equal_parent_mae_b_minus_a_lower": lower / n_parents,
        "full_equal_parent_mae_b_minus_a_upper": upper / n_parents,
        "per_parent": per_parent,
    }


def paired_mae_shared_shift_bounds(rows, *, expected_roster, shift_bounds):
    """Exact finite-roster B-minus-A MAE under one shared target shift.

    Supply (row_id, observed_target, forecast_a, forecast_b, positive_weight)
    tuples, the independently frozen row IDs, and finite closed shift endpoints.
    For log targets the common shift can represent log gain. Predictions and
    weights remain fixed. This is a sensitivity range, not a calibration
    estimate or a confidence interval. Independent row bounds answer a
    different question.

    Each absolute loss is linear between residual breakpoints. An exact
    slope sweep evaluates only the interval endpoints and internal knots
    in O(n log n) time. Ties return the smallest attaining shift.
    """
    if not _bounded_paired_sequence(rows):
        raise ValueError("Rows must be a nonempty bounded list or tuple.")
    if (type(expected_roster) not in (list, tuple, set, frozenset)
            or not 1 <= len(expected_roster) <= 20_000
            or not all(_paired_roster_id(key) for key in expected_roster)):
        raise ValueError("Expected roster must be a nonempty bounded collection of row IDs.")
    expected = set(expected_roster)
    if len(expected) != len(expected_roster):
        raise ValueError("Expected row IDs must be unique.")
    if type(shift_bounds) is not tuple or len(shift_bounds) != 2:
        raise ValueError("Shift bounds must be a finite ordered pair.")
    low, high = map(_paired_binary64_units, shift_bounds)
    if low > high:
        raise ValueError("Shift bounds must be a finite ordered pair.")

    seen, jumps = set(), {}
    gap = slope = weight_sum = 0
    for row in rows:
        if type(row) is not tuple or len(row) != 5 or not _paired_roster_id(row[0]):
            raise ValueError("Rows must be (ID, target, A, B, weight) tuples.")
        key, target, a, b, weight = row
        if key not in expected or key in seen:
            raise ValueError("Rows must match the expected roster exactly.")
        seen.add(key)
        y, pa, pb, w = map(_paired_binary64_units, (target, a, b, weight))
        if w <= 0:
            raise ValueError("Row weights must be positive.")
        weight_sum += w
        for knot, signed_w in ((pb - y, w), (pa - y, -w)):
            gap += signed_w * abs(knot - low)
            slope += signed_w if knot <= low else -signed_w
            if low < knot < high:
                jumps[knot] = jumps.get(knot, 0) + 2 * signed_w
    if seen != expected:
        raise ValueError("Rows must match the expected roster exactly.")

    minimum = maximum = (gap, low)
    initial, previous = gap, low
    for x in sorted(jumps) + [high]:
        gap += slope * (x - previous)
        if gap < minimum[0]:
            minimum = (gap, x)
        if gap > maximum[0]:
            maximum = (gap, x)
        slope += jumps.get(x, 0)
        previous = x
    quantum = _BINARY64_QUANTUM_DENOMINATOR
    divisor = weight_sum * quantum
    return {
        "n_rows": len(seen),
        "at_lower_shift_mae_b_minus_a": Fraction(initial, divisor),
        "at_upper_shift_mae_b_minus_a": Fraction(gap, divisor),
        "min_mae_b_minus_a": Fraction(minimum[0], divisor),
        "max_mae_b_minus_a": Fraction(maximum[0], divisor),
        "min_attaining_shift": Fraction(minimum[1], quantum),
        "max_attaining_shift": Fraction(maximum[1], quantum),
    }


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


def _paired_parent_losses(rows, expected_roster):
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
    return devices, parents


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
    devices, parents = _paired_parent_losses(rows, expected_roster)
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



def paired_parent_mae_deletion_bounds(rows, *, expected_roster, max_deletions):
    """Exact B-minus-A nested MAE sensitivity to whole-parent removal.

    Use the same frozen row/device/parent roster and nested weighting as
    paired_parent_mae_influence. Choose the deletion budget independently of
    model outcomes. Every budget retains at least one parent; the full
    comparison is included at budget zero. Each interval is the exact
    descriptive range over all omissions of up to that many parents. These
    ranges are not confidence intervals or a reason to discard sources.
    """
    if type(max_deletions) is not int or max_deletions < 0:
        raise ValueError("The whole-parent deletion budget must be a nonnegative integer.")
    devices, parents = _paired_parent_losses(rows, expected_roster)
    k = len(parents)
    if max_deletions >= k:
        raise ValueError("The whole-parent deletion budget must retain at least one parent.")

    # The smallest retained mean removes the largest effects, and vice
    # versa, because each parent gets equal weight. The extrema are nested:
    # deleting a largest (smallest) remaining value cannot raise (lower)
    # the corresponding mean. Exactly d deletions therefore also bounds
    # every subset with at most d deletions.
    effects = sorted((loss_b - loss_a) / n for n, loss_a, loss_b in parents.values())
    total = sum(effects, Fraction())
    remaining_low, remaining_high = total, total
    bounds = [(total / k, total / k)]
    for d in range(1, max_deletions + 1):
        remaining_low -= effects[-d]
        remaining_high -= effects[d - 1]
        bounds.append((remaining_low / (k - d), remaining_high / (k - d)))
    return {
        "n_rows": len(rows),
        "n_devices": len(devices),
        "n_declared_parents": k,
        "max_deletions": max_deletions,
        "mae_b_minus_a_bounds_by_budget": tuple(bounds),
    }


def _paired_group_residual_units(rows, expected_roster):
    if not all(map(_bounded_paired_sequence, (rows, expected_roster))):
        raise ValueError("Rows and frozen roster must be bounded nonempty lists or tuples.")
    roster = {}
    for membership in expected_roster:
        if (type(membership) is not tuple or len(membership) != 2
                or not all(map(_paired_roster_id, membership))):
            raise ValueError("Frozen roster memberships require valid (row ID, group ID) tuples.")
        row_id, group = membership
        if row_id in roster:
            raise ValueError("Frozen roster row IDs must be unique.")
        roster[row_id] = group
    seen = set()
    for row in rows:
        if type(row) is not tuple or len(row) != 4 or not _paired_roster_id(row[0]):
            raise ValueError("Paired rows require an ID, observation and both predictions.")
        row_id, target, a, b = row
        if row_id not in roster or row_id in seen:
            raise ValueError("Paired rows must match the frozen roster exactly.")
        seen.add(row_id)
        y, pa, pb = map(_paired_binary64_units, (target, a, b))
        yield roster[row_id], pa - y, pb - y
    if len(seen) != len(roster):
        raise ValueError("Paired rows must match the frozen roster exactly.")


def _paired_group_error_units(rows, expected_roster):
    for group, residual_a, residual_b in _paired_group_residual_units(rows, expected_roster):
        yield group, abs(residual_a), abs(residual_b)


def paired_practical_loss_rates(rows, *, expected_roster, margin):
    """Describe how often either forecast beats the other by a fixed margin.

    Supply one (row_id, observed, prediction_a, prediction_b) tuple per
    independently frozen (row_id, group_id) mapping. The nonnegative
    absolute loss margin is specified in the common target units before
    outcomes are compared. Differences within or on either margin boundary
    count as within-margin, not as wins. Every declared row needs both
    predictions and an observation; none is dropped or imputed.

    Results are exact Fractions over the represented binary64 inputs, with
    equal-row and equal-declared-group weighting reported separately.
    Within-margin is a descriptive category, not statistical equivalence.
    Declared groups do not prove independent replication or future accuracy.
    """
    if not all(map(_bounded_paired_sequence, (rows, expected_roster))):
        raise ValueError("Rows and frozen roster must be bounded nonempty lists or tuples.")
    margin_units = _paired_binary64_units(margin)
    if margin_units < 0:
        raise ValueError("Practical absolute-error margin must be nonnegative.")
    by_group = {}
    for group, error_a, error_b in _paired_group_error_units(rows, expected_roster):
        gap = error_b - error_a
        index = 0 if gap > margin_units else 2 if gap < -margin_units else 1
        by_group.setdefault(group, [0, 0, 0])[index] += 1
    names = ("a_lower_loss_beyond_margin", "within_margin", "b_lower_loss_beyond_margin")
    row_rates = {name: Fraction(sum(counts[i] for counts in by_group.values()), len(rows))
                 for i, name in enumerate(names)}
    equal_group_rates = {name: sum((Fraction(counts[i], sum(counts)) for counts in by_group.values()),
                                   Fraction()) / len(by_group) for i, name in enumerate(names)}
    return {
        "n_rows": len(rows), "n_declared_groups": len(by_group),
        "margin_exact": Fraction(margin_units, _BINARY64_QUANTUM_DENOMINATOR),
        "row_weighted": row_rates, "equal_group": equal_group_rates,
    }


def paired_absolute_squared_loss_tradeoff(rows, *, expected_roster, error_scale):
    """Describe when a fixed paired ranking depends on the loss definition.

    Rows are (row_id, observed, prediction_a, prediction_b) and each ID
    has one entry in an independently frozen (row_id, group) roster. Declare
    a positive error scale in common target units independently of outcomes.
    Compare (1-w)*|error|/scale + w*(error/scale)**2 for w in [0,1].
    Endpoints and any exact tie weight are returned without selecting w.
    Negative B-minus-A means lower loss for B. A tie at 0 or 1 need not
    imply an interior rank reversal. This descriptive cohort sensitivity
    does not establish population performance or physical independence.
    """
    scale = _paired_binary64_units(error_scale)
    if scale <= 0:
        raise ValueError("The absolute/squared loss comparison requires a positive error scale.")
    groups = {}
    for group, error_a, error_b in _paired_group_error_units(rows, expected_roster):
        gap = error_b - error_a
        counts = groups.setdefault(group, [0, 0, 0])
        counts[0] += 1
        counts[1] += gap
        counts[2] += gap * (error_a + error_b)

    def summarize(a, b):
        # a and b are exact differences at squared-loss weights zero and one.
        return {
            "normalized_absolute_b_minus_a": a,
            "normalized_squared_b_minus_a": b,
            "tie_squared_weight": a / (a - b) if a != b and min(a, b) <= 0 <= max(a, b) else None,
            "all_weights_tied": a == b == 0,
            "strict_interior_rank_reversal": a < 0 < b or b < 0 < a,
        }

    return {
        "n_rows": len(rows), "n_declared_groups": len(groups),
        "error_scale_exact": Fraction(scale, _BINARY64_QUANTUM_DENOMINATOR),
        "row_weighted": summarize(
            Fraction(sum(group[1] for group in groups.values()), len(rows) * scale),
            Fraction(sum(group[2] for group in groups.values()), len(rows) * scale**2),
        ),
        "equal_group": summarize(
            sum(Fraction(a, n * scale) for n, a, _ in groups.values()) / len(groups),
            sum(Fraction(b, n * scale**2) for n, _, b in groups.values()) / len(groups),
        ),
    }


def paired_group_mse_components(rows, *, expected_roster):
    """Decompose fixed-cohort MSE for both forecasts and their exact difference.

    Rows are (row_id, observed, prediction_a, prediction_b), on a separately
    frozen (row_id, group_id) roster. All rows must be complete, with shared
    target units. Under either row or equal-group weighting, MSE splits into
    within-group centered dispersion, between-group squared bias dispersion
    and squared overall bias. All terms are exact Fractions for represented
    binary64 values; the signed overall bias is in units of the target and
    other terms are in squared units.

    Between-group dispersion is descriptive, not evidence of physical group
    effects or independence. No test outcomes are used to correct predictions,
    change cohorts, or choose which weighting to report.
    """
    totals = {}
    for group, residual_a, residual_b in _paired_group_residual_units(rows, expected_roster):
        accum = totals.setdefault(group, [0, 0, 0, 0, 0])
        accum[0] += 1
        accum[1] += residual_a
        accum[2] += residual_b
        accum[3] += residual_a * residual_a
        accum[4] += residual_b * residual_b
    quantum = _BINARY64_QUANTUM_DENOMINATOR
    unit_square = quantum**2
    k = len(totals)

    def summarize(column, row_weighted):
        bias = mse = mean_group_bias_square = Fraction()
        for state in totals.values():
            n, error_sum, squares = state[0], state[column], state[column + 2]
            weight = Fraction(n, len(rows)) if row_weighted else Fraction(1, k)
            group_bias = Fraction(error_sum, n * quantum)
            bias += weight * group_bias
            mse += weight * Fraction(squares, n * unit_square)
            mean_group_bias_square += weight * group_bias**2
        bias_square = bias**2
        return {
            "bias": bias,
            "mse": mse,
            "within_group_centered_mse": mse - mean_group_bias_square,
            "between_group_bias_dispersion": mean_group_bias_square - bias_square,
            "overall_bias_squared": bias_square,
        }

    def view(row_weighted):
        a, b = (summarize(column, row_weighted) for column in (1, 2))
        return {"a": a, "b": b, "b_minus_a": {key: b[key] - a[key] for key in a}}

    return {"n_rows": len(rows), "n_declared_groups": k,
            "row_weighted": view(True), "equal_group": view(False)}
