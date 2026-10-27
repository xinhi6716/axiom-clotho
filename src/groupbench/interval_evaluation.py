"""Descriptive, no-fit evaluation of saved prediction intervals on one cohort."""
from __future__ import annotations

import argparse
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


def _label(value, limit=256):
    if (not isinstance(value, str) or not value or len(value) > limit
            or value != value.strip() or any(ord(c) < 32 for c in value)):
        raise ValueError("Labels must be bounded, nonempty, trimmed text without control characters")
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
        if isinstance(value, str) and len(value) > 1024:
            raise ValueError("Text group identifiers must not exceed 1024 characters")
        bounded.append(value)
    # No NumPy numeric promotion before the shared exact-identity validator.
    return validate_groups(bounded, len(bounded))


def _mean(values):
    return sum(values, Fraction()) / len(values)


def _summary(rows, alpha):
    return {
        "n_rows": len(rows),
        "n_covered": sum(row["covered"] for row in rows),
        "empirical_coverage_approx": _approx(Fraction(sum(row["covered"] for row in rows), len(rows))),
        "mean_width_approx": _approx(_mean([row["width"] for row in rows])),
        "mean_interval_score_approx": None if alpha is None else _approx(_mean([row["score"] for row in rows])),
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
        if row["status"] != "ok":
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
                          "covered": lower <= y <= upper, "width": width, "score": score})
    evaluated.sort(key=lambda row: row["row_id"])
    by_group = {}
    for row in evaluated:
        by_group.setdefault(row["group"], []).append(row)
    group_rows = [{"group": group, **_summary(rows, alpha)} for group, rows in sorted(by_group.items())]
    # Aggregate exact group means, never the rounded per-group output numbers.
    equal = {
        "n_groups": len(by_group),
        "empirical_coverage_approx": _approx(_mean([Fraction(sum(r["covered"] for r in rows), len(rows)) for rows in by_group.values()])),
        "mean_width_approx": _approx(_mean([_mean([r["width"] for r in rows]) for rows in by_group.values()])),
        "mean_interval_score_approx": None if alpha is None else _approx(_mean([_mean([r["score"] for r in rows]) for rows in by_group.values()])),
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
        "row_weighted": _summary(evaluated, alpha), "equal_group": equal,
        "groups": group_rows, "observations": observations, "limitations": list(LIMITATIONS),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlcalc evaluate-intervals", description=__doc__)
    parser.add_argument("input", type=Path, help="Regular local saved-interval JSON; report goes to stdout only")
    args = parser.parse_args(argv)
    try:
        report = evaluate_saved_intervals(load_document(_read_input(args.input)))
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, TypeError) as exc:
        print(f"interval evaluation rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
