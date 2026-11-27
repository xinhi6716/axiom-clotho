"""Aggregate source-feature checks, separate from fitted models and transforms."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from .data import validate_prediction_features, _align_normalized_prediction_features


# Explicit handshake for the separate optional numeric-only extension package.
# Defining the contract does not import, build, or select a native backend.
NUMERIC_SUPPORT_PROTOCOL = "numeric-source-support-v1"


def prediction_input_support(training: pd.DataFrame, new_rows: pd.DataFrame) -> dict:
    """Compare new rows with validated, normalized training features only.

    Callers supply training features from ``prepare_data``. Apply the existing
    prediction validation to reorder columns and accommodate all-missing input
    columns, without mutating either frame. No target/group, estimator state,
    imputed values or prediction-batch statistics define the reference.

    Work is bounded by the existing row/feature limits. Only per-feature counts
    and aggregate row counts leave this function, never category inventories,
    numeric extrema, row identifiers or feature values. Counts may overlap.
    """
    incoming = validate_prediction_features(
        new_rows, training.columns, [str(training[c].dtype) for c in training])
    return _input_support_aligned(training, incoming)


def _prediction_input_support_normalized(training, new_rows):
    """Internal fast path for unchanged ``validate_features`` output only.

    Public diagnostics always validate raw inputs above. Planning and inference
    already own normalized copies; repeat only schema/type alignment here.
    """
    incoming = _align_normalized_prediction_features(
        new_rows, training.columns, [str(training[c].dtype) for c in training])
    return _input_support_aligned(training, incoming)


def _input_support_aligned(training, incoming):
    """Count support against normalized training and aligned prediction copies."""
    flags = {name: np.zeros(len(incoming), dtype=bool) for name in (
        "missing_value", "numeric_outside_training_range", "unseen_category",
        "non_missing_value_in_all_missing_training_feature")}
    features = []
    for name in training:
        source, query = training[name], incoming[name]
        source_missing, query_missing = source.isna(), query.isna().to_numpy()
        n_source_missing = int(source_missing.sum())
        all_missing = n_source_missing == len(source)
        n_query_missing = int(query_missing.sum())
        flags["missing_value"] |= query_missing
        if all_missing:
            flags["non_missing_value_in_all_missing_training_feature"] |= ~query_missing
        numeric = pd.api.types.is_numeric_dtype(source)
        feature = {
            "name": name, "kind": "numeric" if numeric else "categorical",
            "training_missing_count": n_source_missing,
            "training_support": "unavailable_all_missing" if all_missing else "observed_non_missing_values",
            "prediction_missing_count": n_query_missing,
            "prediction_non_missing_without_training_values_count":
                len(incoming) - n_query_missing if all_missing else 0,
        }
        if numeric:
            # Normalization already rejects infinities. Direct comparisons
            # avoid overflow from subtracting extreme but finite endpoints.
            feature["below_training_min_count"] = None
            feature["above_training_max_count"] = None
            if not all_missing:
                if source.dtype == np.dtype(np.float64):
                    # The canonical numeric dtype carries missingness as NaN.
                    # Reduce the existing buffer instead of copying observed
                    # rows into a pandas Series just to take its extrema.
                    values = source.to_numpy(copy=False)
                    minimum, maximum = np.nanmin(values), np.nanmax(values)
                else:
                    observed = source[~source_missing]
                    minimum, maximum = observed.min(), observed.max()
                q = query.to_numpy()
                below, above = q < minimum, q > maximum
                feature["below_training_min_count"] = int(below.sum())
                feature["above_training_max_count"] = int(above.sum())
                flags["numeric_outside_training_range"] |= below | above
        else:
            # Source categories, not imputed/encoded categories: a literal
            # missing marker is a normal string; NaN is counted separately.
            observed = source[~source_missing]
            unseen = ~query_missing & ~query.isin(observed.unique()).to_numpy()
            feature["unseen_category_count"] = int(unseen.sum())
            flags["unseen_category"] |= unseen
        features.append(feature)
    any_flag = np.zeros(len(incoming), dtype=bool)
    for mask in flags.values():
        any_flag |= mask
    return {
        "schema_version": "1.0",
        "reference": "full_training_features_before_imputation",
        "action": "diagnostic_only_no_input_changes",
        "n_training_rows": len(training), "n_prediction_rows": len(incoming),
        "rows_with_any_flag": int(any_flag.sum()),
        "row_flag_counts": {name: int(mask.sum()) for name, mask in flags.items()},
        "features": features,
    }


def outer_fold_input_support(X: pd.DataFrame, groups, *, outer_splits: int = 5,
                             random_state: int = 0) -> dict:
    """Return optional target-free support diagnostics for benchmark outer folds.

    Pass exactly the selected feature frame, row order and group labels used by
    the benchmark. This uses its GroupKFold splitter, but does not validate inner
    selection feasibility or fit models. Marginal ranges do not detect holes,
    joint feature support, target shift, or establish future accuracy.
    """
    from .data import validate_features, validate_groups, validate_seed
    from .splitting import GroupKFold

    features = validate_features(X)
    labels = validate_groups(groups, len(features))
    seed = validate_seed(random_state)
    folds = []
    for fold, (train, test) in enumerate(
            GroupKFold(outer_splits, seed).split(features, groups=labels)):
        # Both slices came from the same normalized, schema-aligned frame;
        # do not re-normalize test rows or coerce their already-shared dtypes.
        support = _input_support_aligned(features.iloc[train], features.iloc[test])
        support["reference"] = "outer_training_features_before_imputation"
        folds.append({
            "fold": fold,
            "train_group_count": len(set(labels[train])),
            "test_group_count": len(set(labels[test])),
            "flagged_row_fraction": support["rows_with_any_flag"] / len(test),
            "input_support": support,
        })
    return {
        "schema_version": "1.0",
        "diagnostic": "outer_fold_source_support",
        "action": "diagnostic_only_no_input_changes",
        "splitter": "GroupKFold",
        "random_state": seed,
        "outer_splits": len(folds),
        "n_rows": len(features),
        "n_groups": len(set(labels)),
        "folds": folds,
        "limitations": [
            "Marginal feature checks do not detect interior holes or joint support gaps.",
            "Flags are not an OOD test, calibrated uncertainty, or evidence of future accuracy.",
            "No model selection, predictions, or benchmark outputs are changed.",
            "Inner selection feasibility is not checked by this target-free diagnostic.",
        ],
    }


def audit_numeric_pair_hull(training, query) -> dict:
    """Check closed convex-hull membership for two preselected numeric features.

    Supply bounded sequences of (x, y) tuples in matched units. Coordinates
    must be finite built-in binary64 floats or ints exactly representable as
    binary64. Incomplete coordinates fail rather than being dropped or
    imputed. Selection and unit provenance are the caller's responsibility.

    Membership and affine dimension are exact for the represented inputs.
    Repeated training points do not change the hull. Marginal ranges and
    hull containment are descriptive only: inside the hull does not prove
    local density, a feasible recipe, group independence or transfer skill.
    Only counts are returned, not coordinates, row labels or hull vertices.
    """
    from .data import MAX_ROWS
    from .metrics import _paired_binary64_units

    def points(values):
        if type(values) not in (list, tuple) or not 1 <= len(values) <= MAX_ROWS:
            raise ValueError("Each coordinate roster must be a bounded nonempty list or tuple.")
        result = []
        for pair in values:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("Each numeric point must be an ordered two-coordinate tuple.")
            result.append(tuple(_paired_binary64_units(number) for number in pair))
        return result

    fitted, candidates = points(training), points(query)
    vertices = sorted(set(fitted))

    def turn(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def chain(sequence):
        edge = []
        for point in sequence:
            while len(edge) > 1 and turn(edge[-2], edge[-1], point) <= 0:
                edge.pop()
            edge.append(point)
        return edge

    hull = vertices if len(vertices) == 1 else chain(vertices)[:-1] + chain(reversed(vertices))[:-1]
    n = len(hull)
    xmin, xmax = vertices[0][0], vertices[-1][0]
    ymin, ymax = min(point[1] for point in vertices), max(point[1] for point in vertices)

    def on_segment(a, b, q):
        # Collinearity was checked by the caller; lexicographic order then
        # bounds both axes, including vertical segments.
        return min(a, b) <= q <= max(a, b)

    def inside(q):
        if n == 1:
            return q == hull[0]
        left = turn(hull[0], hull[1], q)
        if n == 2:
            return left == 0 and on_segment(hull[0], hull[1], q)
        right = turn(hull[0], hull[-1], q)
        if left < 0 or right > 0:
            return False
        if left == 0:
            return on_segment(hull[0], hull[1], q)
        if right == 0:
            return on_segment(hull[0], hull[-1], q)
        lower, upper = 1, n - 1
        while upper - lower > 1:
            middle = (lower + upper) // 2
            if turn(hull[0], hull[middle], q) >= 0:
                lower = middle
            else:
                upper = middle
        return turn(hull[lower], hull[upper], q) >= 0

    outside_marginal = outside_joint_inside_marginal = 0
    for q in candidates:
        if not (xmin <= q[0] <= xmax and ymin <= q[1] <= ymax):
            outside_marginal += 1
        elif not inside(q):
            outside_joint_inside_marginal += 1
    return {
        "n_training_rows": len(fitted), "n_distinct_training_points": len(vertices),
        "training_affine_dimension": min(n - 1, 2), "n_query_rows": len(candidates),
        "n_outside_marginal_ranges": outside_marginal,
        "n_outside_hull_inside_marginal_ranges": outside_joint_inside_marginal,
        "n_inside_or_on_hull": len(candidates) - outside_marginal - outside_joint_inside_marginal,
    }


def audit_complete_feature_support_by_group(features: pd.DataFrame, row_groups) -> dict:
    """Count complete feature support and contrast it with marginal support.

    Supply exactly the predeclared features and matching group identifiers
    in positional row order. Use original observations, not imputed data.
    A group with at least one observation per feature need not contain any
    row with *every* feature observed. Only aggregate counts are returned;
    no targets, group IDs, values, model fits or imputation are involved.

    Groups are caller-declared counting units. A complete row does not
    establish independent samples, acquisition provenance or predictability.
    """
    from .data import validate_features, validate_groups

    selected = validate_features(features)
    groups = validate_groups(row_groups, len(selected))
    present = selected.notna().to_numpy()
    complete_rows = present.all(axis=1)
    group_positions = {}
    inverse = np.fromiter(
        (group_positions.setdefault(group, len(group_positions)) for group in groups),
        dtype=np.intp, count=len(groups),
    )
    n_groups = len(group_positions)
    feature_seen = np.zeros((n_groups, present.shape[1]), dtype=bool)
    np.logical_or.at(feature_seen, inverse, present)
    rows_per_group = np.bincount(inverse, minlength=n_groups)
    complete_per_group = np.bincount(inverse[complete_rows], minlength=n_groups)
    all_seen = feature_seen.all(axis=1)

    return {
        "n_rows": len(selected), "n_features": present.shape[1], "n_declared_groups": n_groups,
        "n_complete_rows": int(complete_rows.sum()),
        "n_groups_with_every_feature_observed_marginally": int(all_seen.sum()),
        "n_groups_with_marginal_but_no_complete_rows": int((all_seen & (complete_per_group == 0)).sum()),
        "n_groups_with_at_least_one_complete_row": int((complete_per_group > 0).sum()),
        "n_groups_with_only_complete_rows": int((complete_per_group == rows_per_group).sum()),
    }


def _valid_categorical_support_level(value):
    return (type(value) is str and 1 <= len(value) <= 1024 and bool(value.strip())
            and "\x00" not in value
            and not any(0xd800 <= ord(c) <= 0xdfff for c in value))


def audit_categorical_pair_group_support(training, query, *, training_groups,
                                         minimum_training_groups: int) -> dict:
    """Count declared training groups supporting each preselected category pair.

    Supply explicit ordered two-string tuples in the same feature order,
    without imputations, collapse or case normalization. Group identifiers
    follow the normal group contract and must align to training row positions.
    Choose the positive minimum group count before examining query results.
    Repeated rows from one group do not establish additional group support.

    Distinguish a new marginal level, a previously unseen combination of
    known levels, an observed pair below the group threshold, and a pair
    meeting that threshold. These descriptive counts do not establish
    physical independence, adequate uncertainty, or transfer accuracy.
    """
    from .data import MAX_ROWS, validate_groups

    def pairs(values):
        if type(values) not in (list, tuple) or not 1 <= len(values) <= MAX_ROWS:
            raise ValueError("Training and query pairs require bounded nonempty sequences.")
        # Keep each literal, including meaningful spaces or case. Reject
        # malformed text rather than stringify numeric IDs or missing values.
        for pair in values:
            if (type(pair) is not tuple or len(pair) != 2
                    or not all(_valid_categorical_support_level(value) for value in pair)):
                raise ValueError("Categorical pairs require two bounded, nonblank string values.")
        return values

    if type(minimum_training_groups) is not int or not 1 <= minimum_training_groups <= MAX_ROWS:
        raise ValueError("The minimum training group count must be a bounded positive integer.")
    train, queries = pairs(training), pairs(query)
    groups = validate_groups(training_groups, len(train))
    supporting = {}
    levels_left, levels_right = set(), set()
    for pair, group in zip(train, groups):
        supporting.setdefault(pair, set()).add(group)
        levels_left.add(pair[0])
        levels_right.add(pair[1])

    novel_level = novel_pair = below_minimum = supported = 0
    for pair in queries:
        if pair[0] not in levels_left or pair[1] not in levels_right:
            novel_level += 1
        elif pair not in supporting:
            novel_pair += 1
        elif len(supporting[pair]) < minimum_training_groups:
            below_minimum += 1
        else:
            supported += 1
    return {
        "n_training_rows": len(train), "n_declared_training_groups": len(set(groups)),
        "n_distinct_training_pairs": len(supporting), "n_query_rows": len(queries),
        "minimum_training_groups": minimum_training_groups,
        "n_query_rows_with_unseen_level": novel_level,
        "n_query_rows_with_unseen_pair_of_known_levels": novel_pair,
        "n_query_rows_with_pair_below_group_threshold": below_minimum,
        "n_query_rows_with_pair_at_group_threshold_or_above": supported,
    }


def audit_categorical_tuple_group_support(training, query, *, training_groups,
                                          minimum_training_groups: int) -> dict:
    """Audit full tuples against their pair projections on frozen feature order.

    Supply tuples of 3–8 complete literal strings in common feature order
    and training group IDs aligned by position. A row meets the full threshold
    only when the same complete tuple appears in that many distinct declared
    groups. A group meets the all-pairs condition when every positional pair
    occurs somewhere in its training rows, potentially on different rows.
    Counts only; no fit, imputation, physical independence or accuracy claim.
    """
    from itertools import combinations

    from .data import MAX_ROWS, validate_groups

    if any(type(rows) not in (list, tuple) or not 1 <= len(rows) <= MAX_ROWS
           for rows in (training, query)):
        raise ValueError("Training and query tuples require bounded nonempty sequences.")
    if type(minimum_training_groups) is not int or not 1 <= minimum_training_groups <= MAX_ROWS:
        raise ValueError("The minimum training group count must be a bounded positive integer.")
    if type(training[0]) is not tuple or not 3 <= len(training[0]) <= 8:
        raise ValueError("Categorical tuple support requires three to eight declared fields.")
    width = len(training[0])
    positions = tuple(combinations(range(width), 2))
    if (len(training) + len(query)) * len(positions) > 500_000:
        raise ValueError("Categorical projection work exceeds the audit bound.")
    for rows in (training, query):
        for row in rows:
            if (type(row) is not tuple or len(row) != width
                    or not all(_valid_categorical_support_level(value) for value in row)):
                raise ValueError("Categorical tuples require a common width and bounded nonblank strings.")
    groups = validate_groups(training_groups, len(training))

    def projections(row):
        return ((i, j, row[i], row[j]) for i, j in positions)

    full, pairs, query_counts = {}, {}, {}
    for row in query:
        query_counts[row] = query_counts.get(row, 0) + 1
        for pair in projections(row):
            if pair not in pairs:
                pairs[pair] = set()
    for row, group in zip(training, groups):
        if row in query_counts:
            full.setdefault(row, set()).add(group)
        for pair in projections(row):
            match = pairs.get(pair)
            if match is not None:
                match.add(group)

    full_supported = marginal_supported = common_group_supported = 0
    work = 0
    for row, multiplicity in query_counts.items():
        if row in full and len(full[row]) >= minimum_training_groups:
            full_supported += multiplicity
            marginal_supported += multiplicity
            common_group_supported += multiplicity
            continue
        if not all(len(pairs[pair]) >= minimum_training_groups for pair in projections(row)):
            continue
        marginal_supported += multiplicity
        support_sets = sorted((pairs[pair] for pair in projections(row)), key=len)
        # Bound all membership work even for large overlapping categories.
        work += len(support_sets[0]) * (len(support_sets) - 1)
        if work > 4_000_000:
            raise ValueError("Categorical common-group intersection work exceeds the audit bound.")
        count = 0
        seed, *others = support_sets
        for group in seed:
            if all(group in other for other in others):
                count += 1
                if count >= minimum_training_groups:
                    common_group_supported += multiplicity
                    break

    return {
        "n_training_rows": len(training), "n_query_rows": len(query),
        "n_declared_training_groups": len(set(groups)), "n_fields": width,
        "minimum_training_groups": minimum_training_groups,
        "n_query_rows_meeting_threshold_in_each_pair": marginal_supported,
        "n_query_rows_meeting_threshold_in_common_pair_groups": common_group_supported,
        "n_query_rows_meeting_threshold_for_full_tuple": full_supported,
        "n_query_rows_with_common_pair_support_but_not_full_support": common_group_supported - full_supported,
    }


def audit_numeric_group_range_support(training, query, *, training_groups,
                                      minimum_training_groups: int) -> dict:
    """Describe the declared group ranges supporting one preselected feature.

    Provide one complete finite value per training/query position in common
    units. Each group's closed minimum-to-maximum interval contributes at
    most one vote per query; duplicates do not add votes. Use a group-count
    threshold fixed before examining the query results. Only aggregate counts
    are returned, with no group IDs or source extrema.

    Within a group's range does not imply an observed value, sufficient
    local density, physical independence, or accurate predictions.
    """
    from bisect import bisect_left, bisect_right

    from .data import MAX_ROWS, validate_groups
    from .metrics import _validated_paired_float

    if any(type(values) not in (list, tuple) or not 1 <= len(values) <= MAX_ROWS
           for values in (training, query)):
        raise ValueError("Training and query values require bounded nonempty sequences.")
    if type(minimum_training_groups) is not int or not 1 <= minimum_training_groups <= MAX_ROWS:
        raise ValueError("The minimum training group count must be a bounded positive integer.")
    groups = validate_groups(training_groups, len(training))
    queries = [_validated_paired_float(value) for value in query]

    ranges = {}
    for value, group in zip(training, groups):
        x = _validated_paired_float(value)
        previous = ranges.get(group)
        ranges[group] = (x, x) if previous is None else (min(x, previous[0]), max(x, previous[1]))
    starts = sorted(lo for lo, _ in ranges.values())
    ends = sorted(hi for _, hi in ranges.values())

    external = gap = below_threshold = meets_threshold = 0
    for x in queries:
        if x < starts[0] or x > ends[-1]:
            external += 1
            continue
        # Count intervals which have started, subtract only intervals whose
        # end is strictly below x. Singleton intervals include their point.
        count = bisect_right(starts, x) - bisect_left(ends, x)
        if count == 0:
            gap += 1
        elif count < minimum_training_groups:
            below_threshold += 1
        else:
            meets_threshold += 1
    return {
        "n_training_rows": len(training), "n_query_rows": len(queries),
        "n_declared_training_groups": len(ranges),
        "minimum_training_groups": minimum_training_groups,
        "n_query_rows_outside_pooled_range": external,
        "n_query_rows_inside_pooled_range_without_group_support": gap,
        "n_query_rows_with_group_support_below_threshold": below_threshold,
        "n_query_rows_with_group_support_at_threshold_or_above": meets_threshold,
    }


def quantize_positive_operand(value: Fraction, step: Fraction) -> Fraction | None:
    """Snap a positive exact-rational operand to the zero-origin uniform lattice.

    Exact halfway ties go to the larger code. A zero code is returned as
    None: the operand is unscorable for a downstream positive/log transform,
    not a normal-coverage row. Floats are rejected so callers must declare
    how original numeric tokens were converted before applying a lattice.
    The step is caller-supplied (e.g. estimated on frozen reference only);
    this function does not see calibration, outcomes or the query roster.
    """
    if not isinstance(value, Fraction) or not isinstance(step, Fraction):
        raise TypeError("value and step must be exact Fraction operands")
    if value <= 0 or step <= 0:
        raise ValueError("value and step must be strictly positive")
    ratio = value / step
    whole, rem = divmod(ratio.numerator, ratio.denominator)
    code = whole + (2 * rem >= ratio.denominator)
    return code * step if code else None
