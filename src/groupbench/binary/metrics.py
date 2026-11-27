"""Fixed binary probability scores with explicit weighting and arithmetic."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..data import validate_groups
from .data import _reject_active_masks, validate_binary_target

EPSILON = float(np.finfo(np.float64).eps)


def metric_contract():
    return {"primary": "group_brier", "brier_scale": "binary_[0,1]",
            "group_weighting": "equal_groups_then_equal_rows_within_group",
            "log_loss": "metric_only_clipped_binary_natural_log",
            "log_loss_epsilon": EPSILON, "threshold": 0.5,
            "threshold_comparison": ">=", "undefined_threshold_metrics": None,
            "positive_loss_rounded_to_zero": "error"}



def _reject_boolean_probabilities(values):
    """Preserve source Boolean types until they can be rejected explicitly.

    Called after mask preflight, which already bounds sequence depth and rejects
    cycles. Inspect only known container storage/metadata, never a generator or
    arbitrary iterable. Numeric arrays already promoted by the caller cannot
    reveal their former scalar types.
    """
    pending, seen = [values], set()
    typed_arrays = (np.ndarray, pd.Series, pd.Index, pd.api.extensions.ExtensionArray)
    while pending:
        value = pending.pop()
        boolean = isinstance(value, (bool, np.bool_))
        if isinstance(value, typed_arrays):
            boolean = boolean or pd.api.types.is_bool_dtype(value.dtype)
        elif isinstance(value, memoryview):
            boolean = boolean or value.format.lstrip("@=<>!") == "?"
        if boolean:
            raise ValueError("Probabilities must be numeric, not Boolean; supply explicit numeric 0/1 for probability endpoints. No labels are remapped.")
        if isinstance(value, (list, tuple)) and id(value) not in seen:
            seen.add(id(value))
            items = list.__iter__(value) if isinstance(value, list) else tuple.__iter__(value)
            pending.extend(items)


def _probability_array(values):
    _reject_active_masks(values, "Probabilities")
    _reject_boolean_probabilities(values)
    original = np.asarray(values)
    # pandas 2.0 exposes nullable numeric arrays as object even without missing
    # values. Preserve their declared native dtype, not generic object contents.
    # Missing entries still reject; Boolean/string/custom extension arrays do
    # not enter this conversion, and range checks run before float64 narrowing.
    if original.dtype.kind == "O":
        array = values.array if isinstance(values, (pd.Series, pd.Index)) else values
        if type(array) in (pd.arrays.FloatingArray, pd.arrays.IntegerArray) and not array.isna().any():
            original = array.to_numpy(dtype=array.dtype.numpy_dtype)
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
    if p.shape != y.shape:
        raise ValueError("Binary metrics require finite matching one-dimensional probabilities in [0,1].")
    return _metrics_on_validated_arrays(y, p)


def _metrics_on_validated_arrays(y, p):
    """Score already validated, nonempty, aligned 1-D int64/float64 arrays."""
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
            # Diagnostic rounding follows float64, independent of caller seterr.
            mean_probability = float(np.mean(p))
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
            "positive_prevalence": float(np.mean(y)), "mean_probability_1": mean_probability,
            "brier": brier, "log_loss_clipped": log_loss,
            "probability_below_log_epsilon_count": int(np.sum(p < EPSILON)),
            "probability_above_log_one_minus_epsilon_count": int(np.sum(p > 1 - EPSILON)),
            "probability_zero_count": int(np.sum(p == 0)), "probability_one_count": int(np.sum(p == 1)),
            "threshold": 0.5, "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "accuracy": (tp + tn) / len(y), "recall_1": recall, "specificity_0": specificity,
            "precision_1": _ratio(tp, tp + fp),
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None}


def grouped_binary_metrics(y_true, probability_1, groups):
    """Score rows using the same exact group-identity contract as evaluation."""
    p = _probability_array(probability_1)
    y = validate_binary_target(y_true, len(p) if p.ndim else 0, require_both=False)
    # Validate before NumPy can promote mixed numeric IDs to a lossy dtype.
    g = validate_groups(groups, len(y))
    if p.shape != y.shape:
        raise ValueError("Binary metrics require finite matching one-dimensional probabilities in [0,1].")
    overall = _metrics_on_validated_arrays(y, p)
    per_group = []
    # Group positions once instead of scanning all rows for each group.
    for name, positions in pd.Series(np.arange(len(g))).groupby(g, sort=False).groups.items():
        idx = np.asarray(positions, dtype=int)
        per_group.append({"group": str(name), **_metrics_on_validated_arrays(y[idx], p[idx])})
    group_brier = _mean([row["brier"] for row in per_group])
    worst_brier = max(row["brier"] for row in per_group)
    if not np.isfinite(group_brier) or not 0 <= group_brier <= 1 or not 0 <= worst_brier <= 1:
        raise ValueError("Grouped Brier metric is outside [0,1].")
    overall.update(n_groups=len(per_group), group_brier=group_brier,
                   group_log_loss_clipped=_mean([row["log_loss_clipped"] for row in per_group]),
                   worst_group_brier=worst_brier,
                   worst_group_log_loss_clipped=max(row["log_loss_clipped"] for row in per_group))
    return overall, per_group


def paired_brier_prevalence_sensitivity(rows, *, expected_roster, prevalence_range):
    """Reweight a complete paired Brier comparison over a fixed prevalence range.

    Supply (row_id, observed_label, positive_probability_a, positive_probability_b)
    tuples and a separately frozen (row_id, group_id) roster. Labels must be
    literal integer 0/1 and both labels must occur. Probabilities and closed
    prevalence endpoints are finite built-in binary64 or exact integer values
    in [0, 1]. The prevalence range is externally declared, never inferred
    from the comparison.

    Row weighting fixes the observed loss difference within each class.
    Equal-group weighting first fixes each group's conditional loss difference,
    then averages groups equally within class; it is unavailable if any group
    lacks either class. The resulting mixtures are hypothetical sensitivity
    calculations, not evidence of prevalence, label shift or transportability.
    """
    from fractions import Fraction

    from ..metrics import (
        _BINARY64_QUANTUM_DENOMINATOR, _bounded_paired_sequence,
        _paired_binary64_units, _paired_roster_id,
    )

    if not all(map(_bounded_paired_sequence, (rows, expected_roster))):
        raise ValueError("Paired rows and frozen roster require bounded nonempty sequences.")
    if type(prevalence_range) is not tuple or len(prevalence_range) != 2:
        raise ValueError("Prevalence range requires a closed pair of endpoints.")
    unit = _BINARY64_QUANTUM_DENOMINATOR
    low, high = map(_paired_binary64_units, prevalence_range)
    if not 0 <= low <= high <= unit:
        raise ValueError("Prevalence endpoints must be ordered and within [0, 1].")
    lo, hi = Fraction(low, unit), Fraction(high, unit)
    roster = {}
    for entry in expected_roster:
        if type(entry) is not tuple or len(entry) != 2 or not all(map(_paired_roster_id, entry)):
            raise ValueError("Frozen roster entries require valid (row ID, group ID) tuples.")
        row_id, group = entry
        if row_id in roster:
            raise ValueError("Frozen roster row IDs must be unique.")
        roster[row_id] = group

    seen, group_totals = set(), {}
    for row in rows:
        if type(row) is not tuple or len(row) != 4 or not _paired_roster_id(row[0]):
            raise ValueError("Paired rows require one ID, one label and two probabilities.")
        row_id, label, a, b = row
        if row_id not in roster or row_id in seen:
            raise ValueError("Paired rows must match the frozen roster exactly.")
        seen.add(row_id)
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Paired Brier labels must be literal integers 0 or 1.")
        a, b = map(_paired_binary64_units, (a, b))
        if not (0 <= a <= unit and 0 <= b <= unit):
            raise ValueError("Paired Brier probabilities must be within [0, 1].")
        stats = group_totals.setdefault(roster[row_id], [[0, 0], [0, 0]])[label]
        stats[0] += 1
        stats[1] += (b - a) * (b + a - 2 * label * unit)
    if len(seen) != len(roster):
        raise ValueError("Paired rows must match the frozen roster exactly.")

    counts = tuple(sum(group[i][0] for group in group_totals.values()) for i in (0, 1))
    if not all(counts):
        raise ValueError("Prevalence sensitivity requires both observed classes.")
    square = unit**2

    def report(d0, d1):
        def at(p):
            return d0 + p * (d1 - d0)
        left, right = at(lo), at(hi)
        cross = d0 / (d0 - d1) if d0 != d1 else None
        return {
            "conditional_brier_b_minus_a": (d0, d1),
            "brier_b_minus_a_at_prevalence_endpoints": (left, right),
            "brier_b_minus_a_bounds": (min(left, right), max(left, right)),
            "tie_prevalence_in_declared_range": cross if cross is not None and lo <= cross <= hi else None,
            "all_prevalences_tied": d0 == d1 == 0,
        }

    supporting_groups = tuple(sum(group[i][0] > 0 for group in group_totals.values()) for i in (0, 1))
    n_groups = len(group_totals)
    equal_group = None
    if supporting_groups == (n_groups, n_groups):
        equal_group = report(*(sum(Fraction(g[i][1], g[i][0] * square)
                                   for g in group_totals.values()) / n_groups for i in (0, 1)))
    return {
        "n_rows": len(seen), "n_declared_groups": n_groups, "n_rows_by_label": counts,
        "n_declared_groups_by_label": supporting_groups, "prevalence_range_exact": (lo, hi),
        "row_within_label": report(*(Fraction(sum(g[i][1] for g in group_totals.values()),
                                               counts[i] * square) for i in (0, 1))),
        "equal_group_within_label": equal_group,
    }


def paired_auc_tie_audit(rows, *, expected_roster):
    """Audit exact positive-negative score ordering on a frozen binary cohort.

    Pass (row ID, literal integer 0/1 label, score A, score B) tuples and an
    independently frozen (row ID, group ID) roster. Scores are finite binary64
    or exactly representable integers; higher ranks indicate class 1.
    Pooled AUC permits cross-group comparisons; within-group AUC weights
    each group's own AUC equally and is unavailable if any group lacks a
    class. Midrank AUC awards half credit for ties. Each model's own tie
    envelope freezes all strict orderings and permits arbitrary ordering
    inside tied blocks; its endpoints are attainable but not joint paired
    bounds or confidence intervals.
    """
    from fractions import Fraction

    from ..metrics import _bounded_paired_sequence, _paired_binary64_units, _paired_roster_id

    if not all(map(_bounded_paired_sequence, (rows, expected_roster))):
        raise ValueError("Paired rows and frozen roster require bounded nonempty sequences.")
    roster = {}
    for entry in expected_roster:
        if type(entry) is not tuple or len(entry) != 2 or not all(map(_paired_roster_id, entry)):
            raise ValueError("Frozen roster entries require valid (row ID, group ID) tuples.")
        row_id, group = entry
        if row_id in roster:
            raise ValueError("Frozen roster row IDs must be unique.")
        roster[row_id] = group

    groups, pooled, seen = {}, [], set()
    for row in rows:
        if type(row) is not tuple or len(row) != 4 or not _paired_roster_id(row[0]):
            raise ValueError("Paired rows require one ID, one label and two scores.")
        row_id, label, a, b = row
        if row_id not in roster or row_id in seen:
            raise ValueError("Paired rows must match the frozen roster exactly.")
        seen.add(row_id)
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Paired AUC labels must be literal integers 0 or 1.")
        sample = (label, _paired_binary64_units(a), _paired_binary64_units(b))
        groups.setdefault(roster[row_id], []).append(sample)
        pooled.append(sample)
    if len(seen) != len(roster):
        raise ValueError("Paired rows must match the frozen roster exactly.")

    def model_report(samples, column, pairs):
        blocks = {}
        for row in samples:
            counts = blocks.setdefault(row[column], [0, 0])
            counts[row[0]] += 1
        wins = ties = negatives_below = 0
        for score in sorted(blocks):
            neg, pos = blocks[score]
            wins += pos * negatives_below
            ties += neg * pos
            negatives_below += neg
        return {"wins": wins, "ties": ties, "losses": pairs - wins - ties,
                "auc": Fraction(2 * wins + ties, 2 * pairs),
                "tie_refinement_auc_bounds": (Fraction(wins, pairs), Fraction(wins + ties, pairs))}

    def report(samples):
        pos = sum(row[0] for row in samples)
        pairs = pos * (len(samples) - pos)
        if not pairs:
            return None
        a, b = (model_report(samples, column, pairs) for column in (1, 2))
        return {"n_positive_negative_pairs": pairs, "a": a, "b": b,
                "auc_b_minus_a": b["auc"] - a["auc"]}

    pooled_report = report(pooled)
    if pooled_report is None:
        raise ValueError("Pooled paired AUC requires both observed classes.")
    per_group = {group: report(groups[group]) for group in sorted(groups)}
    incomplete = tuple(group for group, item in per_group.items() if item is None)
    balanced = None
    if not incomplete:
        def balanced_model(model):
            return {
                "auc": sum(item[model]["auc"] for item in per_group.values()) / len(per_group),
                "tie_refinement_auc_bounds": tuple(
                    sum(item[model]["tie_refinement_auc_bounds"][j] for item in per_group.values())
                    / len(per_group) for j in (0, 1)),
            }
        a, b = map(balanced_model, ("a", "b"))
        balanced = {"a": a, "b": b, "auc_b_minus_a": b["auc"] - a["auc"]}
    return {"n_rows": len(rows), "pooled": pooled_report, "per_group": per_group,
            "groups_without_comparable_classes": incomplete, "equal_group": balanced}
