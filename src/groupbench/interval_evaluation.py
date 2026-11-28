"""Descriptive, no-fit evaluation of saved prediction intervals on one cohort."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import math
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

from .budget_comparison import _read_input
from .data import validate_groups


MAX_ROWS = 20_000
MAX_INTEGER = 10**100
META = {"target_definition", "value_unit", "group_definition", "cohort_definition"}
NUMERIC_SEMANTICS = (
    "JSON integers/decimals and Python int/Decimal inputs are exact; Python float "
    "inputs mean their exact stored binary64 values. Comparisons and arithmetic "
    "use exact rational values. Metrics ending in _approx are rounded binary64, "
    "not exact rankings; nonfinite or nonzero-to-zero output is rejected. Exact "
    "source ratios are retained: rounding those endpoints can change apparent "
    "coverage. Already lost caller-side precision cannot be recovered."
)
LIMITATIONS = [
    "Descriptive evaluation of supplied intervals only; no fitting, calibration or interval construction.",
    "No exchangeability, future/population coverage, confidence interval or conformal-prediction guarantee.",
    "Alpha declares central equal-tailed endpoints at alpha/2 and 1-alpha/2; this declaration is not verified.",
    "Nominal coverage is a declaration, not observed coverage or a guarantee.",
    "Target, common units, group meaning and cohort completeness are user declarations; hidden mixed units or targets cannot be detected from numbers.",
    "Rows omitted from both expected_row_ids and rows cannot be detected; do not remove failed observations to make input pass.",
    "Equal-group summaries weight declared groups equally; they do not establish independence or repair a sampling design.",
    "Compare interval scores only at the same alpha, target, units, cohort and weighting definition; do not sum or rank scores across different datasets, units or alpha levels. Narrower intervals alone do not select a better model.",
]


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must have exactly these keys: {sorted(expected)}")


def _require_unicode_scalars(value):
    # Python str can contain surrogate code points. An explicit surrogate
    # pair and the corresponding supplementary scalar are distinct Python
    # strings but serialize identically under json.dumps(ensure_ascii=True).
    if any("\ud800" <= char <= "\udfff" for char in value):
        raise ValueError("Interval text must contain only Unicode scalar values; surrogate code points are not allowed")


def _label(value, limit=256):
    if (not isinstance(value, str) or not value or len(value) > limit
            or value != value.strip() or any(ord(c) < 32 for c in value)):
        raise ValueError("Labels must be bounded, nonempty, trimmed text without control characters")
    _require_unicode_scalars(value)
    return value


def _bounded_decimal(value):
    # Inspect the exponent before Fraction can expand a huge power of ten.
    if not value.is_finite():
        raise ValueError("Numbers must be finite; missing/nonfinite values reject the whole cohort")
    parts = value.as_tuple()
    if len(parts.digits) > 100 or abs(parts.exponent) > 400:
        raise ValueError("Decimal precision/exponent exceeds the 100-digit/400-exponent limits")


def load_document(text):
    """Parse bounded JSON numbers without an initial binary64 conversion."""
    if not isinstance(text, str) or len(text) > 8_000_000:
        raise ValueError("Input must be text within the 8-million-character limit")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def integer(token):
        if len(token.lstrip("-")) > 100:
            raise ValueError("JSON integers must contain at most 100 decimal digits")
        return int(token)

    def decimal(token):
        if len(token) > 1024:
            raise ValueError("JSON decimal token exceeds its 1024-character limit")
        try:
            value = Decimal(token)
        except InvalidOperation as exc:
            raise ValueError("Unsupported JSON decimal") from exc
        _bounded_decimal(value)
        return value

    def constant(_value):
        raise ValueError("Nonfinite JSON constant")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_int=integer,
                          parse_float=decimal, parse_constant=constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds supported depth") from exc


def _approx(value):
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("Numeric result exceeds binary64 reporting range; rescale units") from exc
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ValueError("Numeric result overflows or underflows binary64 reporting range; rescale units")
    return result


def _number(value):
    # Do not silently coerce bool, strings, complex, NumPy or custom scalars.
    if type(value) is int:
        if not -MAX_INTEGER < value < MAX_INTEGER:
            raise ValueError("Integers must contain at most 100 decimal digits")
        exact = Fraction(value)
    elif type(value) is Decimal:
        _bounded_decimal(value)
        exact = Fraction(value)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Numbers must be finite")
        exact = Fraction.from_float(value)
    else:
        raise ValueError("Numeric values must be int, Decimal or finite Python float, not missing values, strings or booleans")
    _approx(exact)
    return exact


def _groups(values):
    bounded = []
    for value in values:
        if type(value) is Decimal:
            _bounded_decimal(value)
            if value != value.to_integral_value():
                raise ValueError("Decimal group identifiers must be integers; use text IDs for fractional decimal identifiers")
            if not -MAX_INTEGER < value < MAX_INTEGER:
                raise ValueError("Numeric group integers must contain at most 100 decimal digits")
            value = int(value)
        if isinstance(value, Integral) and not -MAX_INTEGER < value < MAX_INTEGER:
            raise ValueError("Numeric group integers must contain at most 100 decimal digits")
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                raise ValueError("Group identifiers must be finite")
            numerator, denominator = value.as_integer_ratio()
            if numerator.bit_length() > 4096 or denominator.bit_length() > 4096:
                raise ValueError("Floating group identifiers exceed the 4096-bit exact-ratio limit; use text IDs")
        if isinstance(value, str):
            if len(value) > 1024:
                raise ValueError("Text group identifiers must not exceed 1024 characters")
            _require_unicode_scalars(value)
        bounded.append(value)
    # No NumPy numeric promotion before the shared exact-identity validator.
    return validate_groups(bounded, len(bounded))


def _mean(values):
    return sum(values, Fraction()) / len(values)


def _group_statistics(rows, alpha):
    covered = below = above = 0
    width = score = Fraction()
    for row in rows:
        covered += row["covered"]
        below += row["below_lower"]
        above += row["above_upper"]
        width += row["width"]
        if alpha is not None:
            score += row["score"]
    return len(rows), covered, below, above, width, score


def _summary(stats, alpha):
    n, covered, _, _, width, score = stats
    return {
        "n_rows": n,
        "n_covered": covered,
        "empirical_coverage_approx": _approx(Fraction(covered, n)),
        "mean_width_approx": _approx(width / n),
        "mean_interval_score_approx": None if alpha is None else _approx(score / n),
    }


def _tail_misses(total, grouped):
    """Describe direction of misses, without testing equal-tail calibration."""
    def summarize(stats):
        n, _, below, above, _, _ = stats
        return {"n_rows": n, "below_lower": below, "above_upper": above,
                "below_lower_rate_approx": _approx(Fraction(below, n)),
                "above_upper_rate_approx": _approx(Fraction(above, n))}

    return {
        "rule": "below_lower: observed < lower; above_upper: observed > upper; endpoints are covered",
        "interpretation": "Descriptive tail misses only; no calibration test, independence claim or future-coverage guarantee.",
        "row_weighted": summarize(total),
        "groups": [{"group": group, **summarize(stats)} for group, stats in grouped.items()],
        "equal_group": {"n_groups": len(grouped), **{
            key + "_rate_approx": _approx(_mean([
                Fraction(stats[i], stats[0]) for stats in grouped.values()]))
            for key, i in (("below_lower", 2), ("above_upper", 3))
        }},
    }


def evaluate_saved_intervals(document):
    """Evaluate every declared row, or reject the entire document with ValueError.

    This separate document contract never alters training or saved model bundles.
    ``alpha`` is fixed miscoverage for declared central equal-tailed intervals;
    null omits interval scores. See docs/SAVED_INTERVAL_EVALUATION.md.
    """
    _keys(document, {"schema_version", "metadata", "expected_row_ids", "alpha", "rows"}, "document")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    metadata = document["metadata"]
    _keys(metadata, META, "metadata")
    for value in metadata.values():
        _label(value, 2048)
    expected, source = document["expected_row_ids"], document["rows"]
    for values in (expected, source):
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_ROWS:
            raise ValueError("expected_row_ids and rows must be nonempty lists of at most 20000 entries")
    expected = [_label(value) for value in expected]
    if len(set(expected)) != len(expected):
        raise ValueError("Duplicate expected row identifier")
    alpha = None if document["alpha"] is None else _number(document["alpha"])
    if alpha is not None and not 0 < alpha < 1:
        raise ValueError("alpha must be null or one fixed miscoverage strictly between 0 and 1")
    ids = []
    for row in source:
        _keys(row, {"row_id", "group", "observed", "lower", "upper", "status"}, "row")
        ids.append(_label(row["row_id"]))
        # Array equality can produce a truthy single-element mask rather
        # than a literal success marker in the direct Python API.
        status = row["status"]
        if type(status) is not str or status != "ok":
            raise ValueError("Failed/non-ok row: entire cohort rejected; no failures are dropped")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate observed row identifier")
    if set(ids) != set(expected):
        raise ValueError("Missing or extra row: rows must exactly match the declared expected_row_ids cohort")
    groups = _groups([row["group"] for row in source])
    evaluated = []
    for row, group in zip(source, groups):
        y, lower, upper = [_number(row[key]) for key in ("observed", "lower", "upper")]
        if lower > upper:
            raise ValueError("Reversed interval bounds: lower must not exceed upper")
        width = upper - lower
        score = None if alpha is None else width + 2 * max(lower - y, y - upper, 0) / alpha
        _approx(width)
        if score is not None:
            _approx(score)
        evaluated.append({"row_id": row["row_id"], "group": str(group),
                          "observed_exact": str(y), "lower_exact": str(lower), "upper_exact": str(upper),
                          "covered": lower <= y <= upper,
                          "below_lower": y < lower, "above_upper": y > upper,
                          "width": width, "score": score})
    evaluated.sort(key=lambda row: row["row_id"])
    by_group = {}
    for row in evaluated:
        by_group.setdefault(row["group"], []).append(row)
    # One exact accumulation per group feeds all report views. Summing
    # group totals and averaging exact means are separate weighting rules.
    grouped = {group: _group_statistics(rows, alpha) for group, rows in sorted(by_group.items())}
    total = tuple(sum(column) for column in zip(*grouped.values()))
    group_rows = [{"group": group, **_summary(stats, alpha)} for group, stats in grouped.items()]
    equal = {
        "n_groups": len(grouped),
        "empirical_coverage_approx": _approx(_mean([
            Fraction(stats[1], stats[0]) for stats in grouped.values()])),
        "mean_width_approx": _approx(_mean([stats[4] / stats[0] for stats in grouped.values()])),
        "mean_interval_score_approx": None if alpha is None else _approx(_mean([
            stats[5] / stats[0] for stats in grouped.values()])),
    }
    observations = [{key: value for key, value in row.items() if key not in ("width", "score")} |
                    {"width_approx": _approx(row["width"]),
                     "interval_score_approx": None if alpha is None else _approx(row["score"])} for row in evaluated]
    return {
        "schema_version": 1, "status": "complete_descriptive_interval_evaluation",
        "metadata": dict(metadata), "provenance_verified": False,
        "numeric_semantics": NUMERIC_SEMANTICS, "boundary_rule": "closed: lower <= observed <= upper",
        "alpha_exact": None if alpha is None else str(alpha), "alpha_approx": None if alpha is None else _approx(alpha),
        "nominal_coverage_exact": None if alpha is None else str(1 - alpha),
        "nominal_coverage_approx": None if alpha is None else _approx(1 - alpha),
        "cohort": {"expected_rows": len(expected), "input_rows": len(source), "scored_rows": len(evaluated), "omitted_rows": 0},
        "row_weighted": _summary(total, alpha), "equal_group": equal,
        "groups": group_rows, "observations": observations,
        "tail_misses": _tail_misses(total, grouped),
        "limitations": list(LIMITATIONS),
    }



def compare_saved_interval_scores(document_a, document_b):
    """Compare two declared central intervals on the same frozen cohort.

    Existing single-arm validation is applied to both documents. Expected
    roster order is immaterial; scored identities, normalized groups,
    observations, alpha and all metadata must match exactly. Calculations use
    exact rational arithmetic until reporting and do not fit or resample.
    """
    a, b = map(evaluate_saved_intervals, (document_a, document_b))
    if a["metadata"] != b["metadata"]:
        raise ValueError("Paired interval metadata must match exactly.")
    if a["alpha_exact"] is None or a["alpha_exact"] != b["alpha_exact"]:
        raise ValueError("Paired interval scores require the same nonnull exact alpha.")
    fields = ("row_id", "group", "observed_exact")
    def keys(report):
        return [tuple(row[field] for field in fields) for row in report["observations"]]
    # Each validated document has homogeneous text or numeric groups. The
    # single-arm display strings alone cannot distinguish those namespaces.
    different_group_kind = (isinstance(document_a["rows"][0]["group"], str)
                            != isinstance(document_b["rows"][0]["group"], str))
    if different_group_kind or keys(a) != keys(b):
        raise ValueError("Paired interval roster, row-to-group bindings and exact observations must match.")
    alpha = Fraction(a["alpha_exact"])

    def components(row):
        y, lower, upper = (Fraction(row[field]) for field in
                           ("observed_exact", "lower_exact", "upper_exact"))
        lower_distance, upper_distance = max(lower - y, 0), max(y - upper, 0)
        width = upper - lower
        return (Fraction(lower <= y <= upper), width, lower_distance, upper_distance,
                width + 2 * (lower_distance + upper_distance) / alpha)

    # Accumulate exact group and pooled sums to avoid rounding each group's mean.
    names = ("coverage", "width", "lower_miss_distance", "upper_miss_distance", "interval_score")
    sums = [Fraction()] * (2 * len(names))
    groups = {}
    for one, two in zip(a["observations"], b["observations"]):
        values = components(one) + components(two)
        count, group_sums = groups.get(one["group"], (0, None))
        if group_sums is None:
            group_sums = [Fraction()] * len(sums)
        for i, value in enumerate(values):
            sums[i] += value
            group_sums[i] += value
        groups[one["group"]] = count + 1, group_sums

    def report(means):
        def rendered(values):
            return {name + "_approx": _approx(value) for name, value in zip(names, values)} | {
                "interval_score_exact": str(values[-1])}
        n = len(names)
        return {"a": rendered(means[:n]), "b": rendered(means[n:]),
                "b_minus_a": rendered([means[n + i] - means[i] for i in range(n)])}

    n_rows = len(a["observations"])
    return {
        "status": "complete_descriptive_paired_interval_scores",
        "n_rows": n_rows, "n_declared_groups": len(groups), "metadata": a["metadata"],
        "alpha_exact": a["alpha_exact"],
        "row_weighted": report([total / n_rows for total in sums]),
        "equal_group": report([_mean([totals[i] / count for count, totals in groups.values()])
                               for i in range(len(sums))]),
        "limitations": [
            "Miss distances are zero for covered rows and averaged across all scored rows.",
            "These descriptive comparisons do not establish calibration, independence or future coverage.",
            "Completeness and identity are checked against the declared roster, not externally verified.",
        ],
    }


def _paired_bounded_interval_rows(rows, expected_roster):
    if not all(type(value) in (list, tuple) and 1 <= len(value) <= MAX_ROWS
               for value in (rows, expected_roster)):
        raise ValueError("Rows and the expected roster must be bounded nonempty sequences.")
    ids, groups = [], []
    for entry in expected_roster:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError("Expected roster entries must be (row_id, group) tuples.")
        ids.append(_label(entry[0]))
        groups.append(entry[1])
    if len(set(ids)) != len(ids):
        raise ValueError("Expected row identifiers must be unique.")
    expected = dict(zip(ids, _groups(groups)))

    seen = set()
    for row in rows:
        if type(row) is not tuple or len(row) != 7:
            raise ValueError("Each row must supply one ID and six closed endpoints.")
        key = _label(row[0])
        if key not in expected or key in seen:
            raise ValueError("Rows must match the expected roster exactly.")
        seen.add(key)
        lo, hi, al, ah, bl, bh = map(_number, row[1:])
        if lo > hi or al > ah or bl > bh:
            raise ValueError("Truth and forecast endpoints must be ordered.")
        yield expected[key], lo, hi, al, ah, bl, bh
    if len(seen) != len(expected):
        raise ValueError("Rows must match the expected roster exactly.")


def bound_paired_interval_scores(rows, *, expected_roster, alpha):
    """Sharp B-minus-A interval-score bounds for independently bounded truths.

    Rows are (row_id, truth_lower, truth_upper, a_lower, a_upper, b_lower,
    b_upper) tuples, one per frozen expected (row_id, group) tuple. Group
    assignments come only from that independent roster. The exact interval
    limits may coincide for known truths. Predictions, bounds and common
    alpha must be fixed outside this audit. No rows are dropped or imputed.

    The exact finite-roster extrema allow each truth to vary independently
    within its supplied range; a shared calibration error is a different
    model. These sensitivity bounds establish neither coverage nor inference.
    """
    alpha = _number(alpha)
    if not 0 < alpha < 1:
        raise ValueError("Paired interval bounds require one fixed miscoverage in (0, 1).")
    slope = 2 / alpha

    def score(y, lo, hi):
        return hi - lo + slope * max(lo - y, y - hi, 0)

    group_totals = {}
    for group, lo, hi, al, ah, bl, bh in _paired_bounded_interval_rows(rows, expected_roster):
        # The difference is linear between any successive forecast endpoints.
        # Include internal knots, not just truth-range endpoints.
        candidates = {lo, hi} | {point for point in (al, ah, bl, bh) if lo < point < hi}
        gaps = [score(y, bl, bh) - score(y, al, ah) for y in candidates]
        count, lower, upper = group_totals.get(group, (0, Fraction(), Fraction()))
        group_totals[group] = count + 1, lower + min(gaps), upper + max(gaps)

    return {
        "n_rows": len(rows),
        "n_declared_groups": len(group_totals),
        "alpha_exact": alpha,
        "row_weighted_b_minus_a_bounds": tuple(
            sum(stats[i] for stats in group_totals.values()) / len(rows) for i in (1, 2)),
        "equal_group_b_minus_a_bounds": tuple(
            _mean([stats[i] / stats[0] for stats in group_totals.values()]) for i in (1, 2)),
    }



def bound_paired_interval_scores_common_offset(rows, *, expected_roster, alpha, offset):
    """Sharp B-minus-A interval-score bounds for one common additive offset.

    Each row is (row_id, reported_truth, a_lower, a_upper, b_lower, b_upper).
    The same unknown additive offset applies to every reported truth, in the
    declared units; it is not added to forecasts. A separately supplied closed
    offset interval is the complete feasible set, with no residual row error.
    A different offset per row or group needs a different model. Neither the
    offset certificate nor probabilistic/future coverage is inferred.
    """
    alpha = _number(alpha)
    if not 0 < alpha < 1:
        raise ValueError("Paired interval bounds require one fixed miscoverage in (0, 1).")
    if type(offset) not in (list, tuple) or len(offset) != 2:
        raise ValueError("Common offset requires two ordered closed endpoints.")
    left, right = map(_number, offset)
    if left > right:
        raise ValueError("Common offset requires two ordered closed endpoints.")
    if type(rows) not in (list, tuple) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("Rows must be a bounded nonempty sequence.")
    wrapped = []
    for row in rows:
        if type(row) is not tuple or len(row) != 6:
            raise ValueError("Each row must supply one ID, one reported truth and four forecast endpoints.")
        wrapped.append((row[0], row[1], *row[1:]))
    parsed = list(_paired_bounded_interval_rows(wrapped, expected_roster))
    counts = {}
    for group, *_ in parsed:
        counts[group] = counts.get(group, 0) + 1

    # A score has two convex hinges. Each hinge raises its slope by 2/alpha
    # at its knot; for the negative score, both changes have the opposite sign.
    # Weight and aggregate knot changes before sweeping the shared interval.
    scale = 2 / alpha
    current = [Fraction(), Fraction()]
    derivatives = [Fraction(), Fraction()]
    events = {}
    n, g = len(parsed), len(counts)
    for group, reported, _, al, ah, bl, bh in parsed:
        row_z = Fraction(1, n)
        group_z = Fraction(1, g * counts[group])
        for lo0, hi0, sign in ((bl, bh, 1), (al, ah, -1)):
            lo, hi = lo0 - reported, hi0 - reported
            score = sign * (hi - lo + scale * max(lo - left, left - hi, 0))
            derivative = sign * scale * (int(left >= hi) - int(left < lo))
            current[0] += row_z * score
            current[1] += group_z * score
            derivatives[0] += row_z * derivative
            derivatives[1] += group_z * derivative
            for knot in (lo, hi):
                if left < knot < right:
                    a, b = events.get(knot, (Fraction(), Fraction()))
                    events[knot] = (a + row_z * sign * scale, b + group_z * sign * scale)

    lower, upper = current[:], current[:]
    at_lower, at_upper = [left, left], [left, left]
    previous = left
    for knot in [*sorted(events), right]:
        for k in range(2):
            current[k] += (knot - previous) * derivatives[k]
            if current[k] < lower[k]:
                lower[k], at_lower[k] = current[k], knot
            if current[k] > upper[k]:
                upper[k], at_upper[k] = current[k], knot
        if knot in events:
            derivatives[0] += events[knot][0]
            derivatives[1] += events[knot][1]
        previous = knot
    return {
        "n_rows": n,
        "n_declared_groups": g,
        "alpha_exact": alpha,
        "common_offset_exact": (left, right),
        "row_weighted_b_minus_a_bounds": (lower[0], upper[0]),
        "equal_group_b_minus_a_bounds": (lower[1], upper[1]),
        "row_weighted_extremizing_offsets": (at_lower[0], at_upper[0]),
        "equal_group_extremizing_offsets": (at_lower[1], at_upper[1]),
    }


def bound_paired_failure_risk_at_horizon(rows, *, expected_roster, horizon):
    """Sharp descriptive B-minus-A first-failure risk at a fixed horizon.

    Each row is (pair_id, a_last_event_free, a_first_failed,
    b_last_event_free, b_first_failed) for two separately observed matched
    attempts. Use None for an unobserved bound, not for a confirmed event-free
    read. The event must be an absorbing first failure or a separately logged
    onset; a current pass does not rule out earlier recoverable excursions.
    Horizon and times are nonnegative on the same elapsed scale, using
    T > last_event_free, T <= first_failed and event indicator T <= horizon.
    The association roster is fixed before outcomes. Bounds are sharp on the
    product of the declared feasible time ranges, without cross-attempt time
    restrictions, not confidence intervals or a causal effect.
    """
    h = _number(horizon)
    if h < 0:
        raise ValueError("Failure horizon must be nonnegative.")
    if type(rows) not in (list, tuple) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("Failure rows must be a bounded nonempty sequence.")
    wrapped, by_pair = [], {}
    for row in rows:
        if type(row) is not tuple or len(row) != 5:
            raise ValueError("Each pair needs an ID and two last-event-free/first-failure time pairs.")
        key = _label(row[0])
        if key in by_pair:
            raise ValueError("Failure pair identifiers must be unique.")
        by_pair[key] = row[1:]
        wrapped.append((key, 0, 0, 0, 0, 0, 0))

    def event_range(last, first):
        last = None if last is None else _number(last)
        first = None if first is None else _number(first)
        if ((last is not None and last < 0) or (first is not None and first < 0)
                or (last is not None and first is not None and first <= last)):
            raise ValueError("Failure observation times must be nonnegative and strictly ordered.")
        if first is not None and first <= h:
            return (1, 1)
        if last is not None and last >= h:
            return (0, 0)
        return (0, 1)

    groups = {}
    paired = list(_paired_bounded_interval_rows(wrapped, expected_roster))
    for row, (group, *_) in zip(wrapped, paired, strict=True):
        a0, a1, b0, b1 = by_pair[row[0]]
        a_lo, a_hi = event_range(a0, a1)
        b_lo, b_hi = event_range(b0, b1)
        count, al, ah, bl, bh = groups.get(group, (0, 0, 0, 0, 0))
        groups[group] = (count + 1, al + a_lo, ah + a_hi, bl + b_lo, bh + b_hi)

    rows_n = len(rows)
    group_n = len(groups)
    all_stats = tuple(groups.values())
    row_values = tuple(Fraction(sum(s[i] for s in all_stats), rows_n) for i in range(1, 5))
    group_values = tuple(
        sum(Fraction(s[i], s[0]) for s in all_stats) / group_n for i in range(1, 5))

    def summarize(vals):
        al, ah, bl, bh = vals
        return {"a_risk_bounds": (al, ah), "b_risk_bounds": (bl, bh),
                "b_minus_a_risk_bounds": (bl - ah, bh - al)}

    return {
        "n_pairs": rows_n, "n_declared_groups": group_n, "horizon_exact": h,
        "a_indeterminate_pairs": sum(s[2] - s[1] for s in all_stats),
        "b_indeterminate_pairs": sum(s[4] - s[3] for s in all_stats),
        "row_weighted": summarize(row_values),
        "equal_group": summarize(group_values),
    }


def bound_paired_onset_brier_paths(rows, *, expected_roster, horizons):
    """Sharp B-minus-A Brier difference over compatible first-event paths.

    Give a complete frozen (episode_id, group) roster and rows (episode_id,
    last_event_free, first_failed, a_probabilities, b_probabilities). A and
    B predict the common cumulative onset target at increasing horizons.
    Times are nonnegative and use last_event_free < T <= first_failed;
    None means that bound was not observed. The event must be first onset,
    not a reversible snapshot. Forecasts must be fixed at a common origin.
    Every episode receives all horizons; sharpness assumes admissible event
    intervals are independently selectable across episodes.
    """
    if type(horizons) not in (list, tuple) or not 1 <= len(horizons) <= 256:
        raise ValueError("Expected a bounded nonempty list of horizons.")
    grid = tuple(_number(h) for h in horizons)
    m = len(grid)
    if grid[0] < 0 or any(a >= b for a, b in zip(grid, grid[1:])):
        raise ValueError("Horizons must be nonnegative and strictly increasing.")
    if (type(rows) not in (list, tuple) or not 1 <= len(rows) <= MAX_ROWS
            or len(rows) * m > 200_000):
        raise ValueError("Episode/horizon population is empty or above the exact scoring limit.")

    wrapped = []
    for row in rows:
        if type(row) is not tuple or len(row) != 5:
            raise ValueError("Each episode must include two event-time bounds and two forecast paths.")
        wrapped.append((row[0], 0, 0, 0, 0, 0, 0))

    def probability_path(values):
        if type(values) not in (list, tuple) or len(values) != m:
            raise ValueError("Both forecast paths must cover every registered horizon.")
        v = tuple(_number(p) for p in values)
        if v[0] < 0 or v[-1] > 1 or any(a > b for a, b in zip(v, v[1:])):
            raise ValueError("Cumulative onset probabilities must be nondecreasing in [0, 1].")
        return v

    groups, n_ambiguous = {}, 0
    for (group, *_), (_, last0, first0, a0, b0) in zip(
            _paired_bounded_interval_rows(wrapped, expected_roster), rows, strict=True):
        last = None if last0 is None else _number(last0)
        first = None if first0 is None else _number(first0)
        if ((last is not None and last < 0) or (first is not None and first < 0)
                or (last is not None and first is not None and first <= last)):
            raise ValueError("Event times must be nonnegative and strictly ordered.")
        k_min = 0 if last is None else bisect_right(grid, last)
        k_max = m if first is None else bisect_left(grid, first)
        a, b = probability_path(a0), probability_path(b0)

        # Path k begins with k no-event labels and ends with m-k event labels.
        # Changing one label from 1 to 0 increases (Brier B − Brier A) by
        # exactly 2*(p_B - p_A). Sweep once, never allow a later 1-to-0 path.
        diff = sum((y - x) * (y + x - 2) for x, y in zip(a, b))
        diff += sum(2 * (b[i] - a[i]) for i in range(k_min))
        lo = hi = diff
        for i in range(k_min, k_max):
            diff += 2 * (b[i] - a[i])
            lo, hi = min(lo, diff), max(hi, diff)
        n_ambiguous += int(k_min < k_max)
        count, lows, highs = groups.get(group, (0, Fraction(), Fraction()))
        groups[group] = count + 1, lows + lo, highs + hi

    totals = list(groups.values())
    n, g = len(rows), len(totals)
    return {
        "n_episodes": n, "n_declared_groups": g, "n_horizons": m,
        "n_episodes_with_ambiguous_path": n_ambiguous,
        "row_weighted_b_minus_a_brier_bounds": tuple(
            sum(s[i] for s in totals) / (n * m) for i in (1, 2)),
        "equal_group_b_minus_a_brier_bounds": tuple(
            sum(s[i] / s[0] for s in totals) / (g * m) for i in (1, 2)),
    }


def bound_paired_interval_coverage(rows, *, expected_roster):
    """Bound B-minus-A empirical coverage under a shared, bounded truth per row.

    Use the same frozen tuple and row/group roster contract as
    bound_paired_interval_scores. Each interval and truth range is closed.
    The caller matches target, units and nominal coverage between arms.
    Marginal truths can vary independently across rows but cannot differ
    between arms on a row. No labels are imputed, no rows are dropped, and
    these are descriptive finite-roster rate bounds, not coverage guarantees.
    """
    def possibly_only(lo, hi, al, ah, bl, bh):
        # Clip A to the admissible range. There is an A-only value exactly
        # when the clip is nonempty and is not contained in B.
        left, right = max(lo, al), min(hi, ah)
        return left <= right and (left < bl or right > bh)

    def necessarily_only(lo, hi, al, ah, bl, bh):
        return al <= lo <= hi <= ah and (hi < bl or lo > bh)

    group_totals = {}
    n_uncertain = 0
    for group, lo, hi, al, ah, bl, bh in _paired_bounded_interval_rows(rows, expected_roster):
        lower = (-1 if possibly_only(lo, hi, al, ah, bl, bh)
                 else int(necessarily_only(lo, hi, bl, bh, al, ah)))
        upper = (1 if possibly_only(lo, hi, bl, bh, al, ah)
                 else -int(necessarily_only(lo, hi, al, ah, bl, bh)))
        n_uncertain += lower != upper
        count, a, b = group_totals.get(group, (0, 0, 0))
        group_totals[group] = count + 1, a + lower, b + upper
    return {
        "n_rows": len(rows),
        "n_declared_groups": len(group_totals),
        "n_rows_with_unresolved_coverage_difference": n_uncertain,
        "row_weighted_coverage_b_minus_a_bounds": tuple(
            Fraction(sum(stats[i] for stats in group_totals.values()), len(rows)) for i in (1, 2)),
        "equal_group_coverage_b_minus_a_bounds": tuple(
            _mean([Fraction(stats[i], stats[0]) for stats in group_totals.values()]) for i in (1, 2)),
    }


def bound_paired_aggregate_bias(rows, *, expected_roster):
    """Bound signed bias and paired absolute aggregate bias with censored truths.

    Rows are (row ID, truth lower, truth upper, point A, point B). Supply an
    independently frozen (row ID, group ID) roster. Adjacent intervals may
    degenerate to an exact truth. Each row has one truth shared by both models;
    distinct row truths vary independently over their closed intervals.
    Absolute aggregate bias means the magnitude of the weighted mean signed
    error, not mean absolute error or mean absolute group bias. Each weighting
    is optimized separately; the results are finite-roster sensitivity bounds.
    """
    if (type(rows) not in (list, tuple) or not 1 <= len(rows) <= MAX_ROWS
            or any(type(row) is not tuple or len(row) != 5 for row in rows)):
        raise ValueError("Each row must supply one ID, a truth interval and two point forecasts.")
    wrapped = [(key, lo, hi, a, a, b, b) for key, lo, hi, a, b in rows]
    totals = {}
    for group, lo, hi, a, _, b, _ in _paired_bounded_interval_rows(wrapped, expected_roster):
        item = totals.setdefault(group, [0, Fraction(), Fraction(), Fraction(), Fraction()])
        item[0] += 1
        for index, value in enumerate((lo, hi, a, b), 1):
            item[index] += value

    def summary(lo, hi, a, b):
        # A positive weighted sum of independent closed intervals fills its
        # whole aggregate interval. The two absolute biases share that value.
        candidates = {lo, hi} | {value for value in (a, b) if lo < value < hi}
        contrasts = [abs(b - y) - abs(a - y) for y in candidates]
        return {
            "mean_truth_bounds": (lo, hi),
            "a_signed_bias_bounds": (a - hi, a - lo),
            "b_signed_bias_bounds": (b - hi, b - lo),
            "signed_bias_b_minus_a": b - a,
            "absolute_bias_b_minus_a_bounds": (min(contrasts), max(contrasts)),
        }

    micro = [sum(group[i] for group in totals.values()) / len(rows) for i in range(1, 5)]
    macro = [_mean([group[i] / group[0] for group in totals.values()]) for i in range(1, 5)]
    return {"n_rows": len(rows), "n_groups": len(totals),
            "row_weighted": summary(*micro), "equal_group": summary(*macro)}


def bound_paired_window_direction_accuracy(rows, *, expected_roster, deadband):
    """Sharp B-minus-A direction accuracy bounds for two censored anchors.

    Rows are (window ID, start truth low/high, end truth low/high, A start,
    A end, B start, B end). The roster freezes (window ID, group ID).
    Changes within the inclusive [-deadband, deadband] interval are flat.
    The two observed anchors can vary independently in their stated limits;
    each resulting direction is shared by both forecasts. Across windows
    endpoint uncertainty must also be uncoupled. Repeated physical anchors
    with a common unknown value are not supported by these sharp bounds.
    No causality, error control or population accuracy is implied.
    """
    deadband = _number(deadband)
    if deadband < 0:
        raise ValueError("The direction deadband must be nonnegative.")
    if (type(rows) not in (list, tuple) or not 1 <= len(rows) <= MAX_ROWS
            or any(type(row) is not tuple or len(row) != 9 for row in rows)):
        raise ValueError("Each row requires one ID, two closed truth intervals and both forecast endpoints.")

    def direction(change):
        return -1 if change < -deadband else 1 if change > deadband else 0

    # Normalize all arithmetic before constructing the derived intervals.
    # Pass original decimal/float endpoints through the shared roster validator,
    # rather than casting exact derived endpoints back to a supported scalar.
    starts = [(key, slo, shi, a0, a0, b0, b0)
              for key, slo, shi, _, _, a0, _, b0, _ in rows]
    ends = [(key, elo, ehi, a1, a1, b1, b1)
            for key, _, _, elo, ehi, _, a1, _, b1 in rows]
    groups = {}
    n_ambiguous = 0
    for start, end in zip(_paired_bounded_interval_rows(starts, expected_roster),
                          _paired_bounded_interval_rows(ends, expected_roster), strict=True):
        group, slo, shi, a0, _, b0, _ = start
        _, elo, ehi, a1, _, b1, _ = end
        possible = range(direction(elo - shi), direction(ehi - slo) + 1)
        n_ambiguous += len(possible) > 1
        a, b = direction(a1 - a0), direction(b1 - b0)
        contrasts = [int(b == y) - int(a == y) for y in possible]
        stats = groups.setdefault(group, [0, 0, 0])
        stats[0] += 1
        stats[1] += min(contrasts)
        stats[2] += max(contrasts)

    return {
        "n_windows": len(rows), "n_groups": len(groups),
        "n_direction_ambiguous_windows": n_ambiguous,
        "row_weighted_accuracy_b_minus_a_bounds": tuple(
            Fraction(sum(stats[i] for stats in groups.values()), len(rows)) for i in (1, 2)),
        "equal_group_accuracy_b_minus_a_bounds": tuple(
            _mean([Fraction(stats[i], stats[0]) for stats in groups.values()]) for i in (1, 2)),
    }


def diagnose_saved_interval_reference(document):
    """Return an opt-in descriptive sidecar, without changing normal evaluation.

    Reuse the entire saved-interval contract before deriving any diagnostics.
    Exact gaps compare observed rates with a declared reference; they are not
    calibration errors, hypothesis tests or targets to optimize to zero. No
    fitting, interval construction, cohort selection or file writing occurs.
    """
    report = evaluate_saved_intervals(document)
    alpha = None if report["alpha_exact"] is None else Fraction(report["alpha_exact"])
    reference = None if alpha is None else {
        "coverage": 1 - alpha, "below_lower": alpha / 2, "above_upper": alpha / 2,
    }
    tails = {row["group"]: row for row in report["tail_misses"]["groups"]}
    group_counts = [{
        "group": row["group"], "n_rows": row["n_rows"], "covered": row["n_covered"],
        "below_lower": tails[row["group"]]["below_lower"],
        "above_upper": tails[row["group"]]["above_upper"],
    } for row in report["groups"]]
    n_rows = report["cohort"]["scored_rows"]
    n_groups = len(group_counts)
    count_keys = {"coverage": "covered", "below_lower": "below_lower", "above_upper": "above_upper"}
    total_counts = {key: sum(row[key] for row in group_counts) for key in count_keys.values()}
    row_rates = {name: Fraction(total_counts[key], n_rows) for name, key in count_keys.items()}
    group_rates = {name: _mean([Fraction(row[key], row["n_rows"]) for row in group_counts])
                   for name, key in count_keys.items()}

    def exact_rates_and_gaps(rates):
        return {
            "observed_rate_exact": {key: str(value) for key, value in rates.items()},
            "observed_minus_declared_exact": None if reference is None else {
                key: str(value - reference[key]) for key, value in rates.items()
            },
        }

    return {
        "schema_id": "mlcalc.saved_interval_reference_diagnostic", "schema_version": "1.0",
        "status": "complete_descriptive_reference_diagnostic", "claim_scope": "descriptive_only",
        "source_evaluation_schema_version": report["schema_version"],
        "metadata": dict(report["metadata"]), "cohort": dict(report["cohort"]),
        "provenance_verified": False, "boundary_rule": report["boundary_rule"],
        "alpha_exact": report["alpha_exact"],
        "reference_origin": "declared_input_alpha_not_verified",
        "declared_reference_exact": None if reference is None else {
            key: str(value) for key, value in reference.items()
        },
        "numeric_semantics": "Rates, references and signed gaps are exact rational strings; no float rendering is used. Input numeric semantics are those of evaluate_saved_intervals.",
        "row_weighted": {"weighting": "equal_rows", "n_rows": n_rows,
                         "counts": total_counts, **exact_rates_and_gaps(row_rates)},
        "equal_group": {"weighting": "equal_declared_groups_then_equal_rows_within_group",
                        "n_groups": n_groups, "denominator_support": "group_counts",
                        **exact_rates_and_gaps(group_rates)},
        "group_counts": group_counts,
        "support": {"n_scored_rows": n_rows, "n_declared_groups": n_groups,
                    "counting_unit": "declared_input_groups_not_independent_sources",
                    "independence": "not_established"},
        "limitations": [
            "Observed-minus-declared differences describe this supplied cohort only; no calibration test, confidence interval or future-coverage guarantee is provided.",
            "Nonzero differences are not calibration failures: finite cohorts, discrete mass and closed endpoints can produce them even with correct predictive quantiles. Zero is not a pass threshold.",
            "Alpha and central equal-tailed interpretation are declarations, not verified probabilities. Null alpha leaves references and differences unavailable; none is inferred.",
            "Equal-group weighting does not establish independence and has no row-like binomial denominator. group_counts retains each declared group denominator.",
            "Do not change intervals, alpha, groups or the frozen cohort to improve these diagnostics after inspecting outcomes.",
            "No source provenance or independent-source count is established. Metadata and group labels are included; review before sharing.",
        ],
    }


def diagnose_saved_interval_whole_group_coverage(document):
    """Describe coverage of every declared row within each supplied group.

    The exact expected row roster and saved-interval rules are checked before
    counting one vote per declared group. The declared row roster and row-to-
    group assignments are not independently verified. Missing groups or rows
    omitted from both inputs are undetectable. The result is not an estimate
    of simultaneous future-group coverage or a calibration diagnostic.
    """
    report = evaluate_saved_intervals(document)
    groups = [
        {"group": row["group"], "n_rows": row["n_rows"], "n_covered": row["n_covered"],
         "all_rows_covered": row["n_rows"] == row["n_covered"]}
        for row in report["groups"]
    ]
    n_covered = sum(row["all_rows_covered"] for row in groups)
    return {
        "schema_id": "mlcalc.saved_interval_whole_group_coverage",
        "schema_version": "1.0",
        "claim_scope": "descriptive_supplied_cohort_only",
        "provenance_verified": False,
        "boundary_rule": report["boundary_rule"],
        "cohort": dict(report["cohort"]),
        "whole_group": {
            "observed_coverage_exact": str(Fraction(n_covered, len(groups))),
            "n_groups": len(groups),
            "n_groups_all_rows_covered": n_covered,
        },
        "groups": groups,
        "limitations": [
            "Each group counts once and is covered only if every declared row is covered.",
            "The supplied group assignments and physical independence are not verified.",
            "Groups and rows absent from both declarations and input are undetectable.",
            "No interval construction, calibration or future-group coverage guarantee.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlcalc evaluate-intervals", description=__doc__)
    parser.add_argument("input", type=Path, help="Regular local saved-interval JSON; report goes to stdout only")
    diagnostics = {
        "reference": diagnose_saved_interval_reference,
        "whole-group": diagnose_saved_interval_whole_group_coverage,
    }
    parser.add_argument("--diagnostic", choices=diagnostics,
                        help="Emit an opt-in diagnostic instead of the default report: "
                             "reference compares observed rates with declared alpha; "
                             "whole-group counts groups whose every supplied row is covered.")
    args = parser.parse_args(argv)
    try:
        evaluator = evaluate_saved_intervals if args.diagnostic is None else diagnostics[args.diagnostic]
        report = evaluator(load_document(_read_input(args.input)))
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, TypeError) as exc:
        print(f"interval evaluation rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
