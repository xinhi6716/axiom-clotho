"""No-fit, complete-declared-cohort evaluation of saved binary probabilities."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import sys

from ..budget_comparison import _read_input

from ..data import validate_groups
from .data import label_contract, validate_binary_target
from .metrics import _probability_array, grouped_binary_metrics, metric_contract

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 16
MAX_ROWS = 20_000
MAX_INTEGER = 10**100
METADATA_KEYS = {
    "target_definition", "positive_class_definition", "negative_class_definition",
    "prediction_definition", "group_definition", "cohort_definition",
}


def _keys(value, expected, context):
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{context} must be a plain dict with exactly these keys: {sorted(expected)}")


def _label(value, limit):
    if (type(value) is not str or not value or len(value) > limit
            or value != value.strip() or any(ord(char) < 32 for char in value)):
        raise ValueError("Declarations and row IDs must be bounded, nonempty, trimmed text without control characters")
    return value


def _numeric(value):
    # Inspect native scalars before array conversion can discard their types.
    if type(value) not in (int, float):
        raise ValueError("Labels and probabilities must be plain numeric int/float scalars, not Boolean, text, missing or array values")
    if type(value) is int and not -MAX_INTEGER < value < MAX_INTEGER:
        raise ValueError("Numeric integers must contain at most 100 decimal digits")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Labels and probabilities must be finite")
    return value


def _group(value):
    if type(value) is str:
        if len(value) > 1024:
            raise ValueError("Text groups must contain at most 1024 characters")
        return value
    if type(value) not in (int, float):
        raise ValueError("Groups must be plain text or numeric int/float identifiers, not Boolean or containers")
    if type(value) is int and not -MAX_INTEGER < value < MAX_INTEGER:
        raise ValueError("Numeric group integers must contain at most 100 decimal digits")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Numeric groups must be finite")
    return value


def evaluate_saved_binary_probabilities(document):
    """Score all supplied expected rows or reject with ValueError, without fitting.

    A separate plain-Python document API, not an existing bundle reader. Source
    declarations, expected-row completeness, outcome timing and prediction
    freezing cannot be independently established. See SAVED_BINARY_EVALUATION.md.
    """
    _keys(document, {"schema_version", "metadata", "expected_row_ids", "rows"}, "document")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    metadata = document["metadata"]
    _keys(metadata, METADATA_KEYS, "metadata")
    for value in metadata.values():
        _label(value, 2048)
    expected, rows = document["expected_row_ids"], document["rows"]
    for values in (expected, rows):
        if type(values) is not list or not 1 <= len(values) <= MAX_ROWS:
            raise ValueError("expected_row_ids and rows must be nonempty plain lists of at most 20000 entries")
    expected = [_label(value, 256) for value in expected]
    if len(set(expected)) != len(expected):
        raise ValueError("Duplicate expected row ID")
    ids = []
    for row in rows:
        _keys(row, {"row_id", "group", "observed_label", "probability_1", "status"}, "row")
        ids.append(_label(row["row_id"], 256))
        if type(row["status"]) is not str or row["status"] != "ok":
            raise ValueError("Failed/non-ok row: entire cohort rejected; no row is dropped")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate supplied row ID")
    if set(ids) != set(expected):
        raise ValueError("Missing or extra row: supplied rows must exactly match expected_row_ids")

    ordered = sorted(rows, key=lambda row: row["row_id"])
    labels = validate_binary_target([_numeric(row["observed_label"]) for row in ordered],
                                    len(ordered), require_both=False)
    probabilities = _probability_array([_numeric(row["probability_1"]) for row in ordered])
    groups = validate_groups([_group(row["group"]) for row in ordered], len(ordered))
    metrics, per_group = grouped_binary_metrics(labels, probabilities, groups)
    observations = [{
        "row_id": row["row_id"], "group": str(group), "observed_label": int(label),
        "probability_1": float(probability), "predicted_label": int(probability >= 0.5),
    } for row, group, label, probability in zip(ordered, groups, labels, probabilities)]
    return {
        "schema_id": "mlcalc.saved_binary_probability_evaluation", "schema_version": "1.0",
        "status": "complete_descriptive_binary_evaluation", "claim_scope": "descriptive_only",
        "metadata": dict(metadata), "provenance_verified": False,
        "input_schema_version": 1,
        "numeric_semantics": "Plain Python int/float inputs are scored with existing float64 binary metrics. Already lost caller-side precision or source scalar types cannot be recovered; no exact-decimal or source-authenticity claim.",
        "cohort": {"expected_rows": len(expected), "input_rows": len(rows),
                   "scored_rows": len(ordered), "omitted_rows": 0},
        "label_contract": label_contract(), "metric_contract": metric_contract(),
        "metrics": metrics, "groups": per_group, "observations": observations,
        "support": {
            "n_declared_groups": len(per_group), "independence": "not_established",
            "n_groups_with_both_classes": sum(row["n_true_0"] > 0 and row["n_true_1"] > 0 for row in per_group),
            "n_groups_only_class_0": sum(row["n_true_1"] == 0 for row in per_group),
            "n_groups_only_class_1": sum(row["n_true_0"] == 0 for row in per_group),
        },
        "limitations": [
            "No training, model selection, probability calibration, threshold tuning, statistical tests or future-performance guarantee.",
            "Expected IDs, target/class definitions, predictions and cohort completeness are declarations; rows omitted from both manifests cannot be detected.",
            "Prediction freezing, outcome timing, leakage, provenance and source independence are not verified.",
            "Single-class groups and cohorts are retained; undefined threshold metrics remain null, not zero or a reason to drop rows.",
            "Equal-group metrics weight declared groups equally; group counts do not establish independent sources.",
            "The fixed threshold is 0.5; do not optimize it or remove scored observations after inspecting outcomes.",
            "This output contains row/group IDs, observed labels, probabilities and declared metadata; review before sharing.",
        ],
    }


def _check_json_depth(text):
    """Bound containers before recursive JSON parsing, ignoring string contents."""
    depth, in_string, escaped = 0, False, False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError("Saved binary JSON nesting exceeds 16 containers")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Saved binary JSON has invalid container structure")


def _json_decimal_to_native(value, field):
    if type(value) is not Decimal:
        return value
    if field == "observed_label":
        if value not in (0, 1):
            raise ValueError("JSON binary labels must be exactly numeric 0 or 1 before conversion")
        return int(value)
    if field == "group":
        if value == value.to_integral_value():
            result = int(value)
            if not -MAX_INTEGER < result < MAX_INTEGER:
                raise ValueError("JSON numeric group integers exceed 100 digits; use text IDs")
            return result
        result = float(value)
        if not math.isfinite(result) or Decimal.from_float(result) != value:
            raise ValueError("JSON fractional group IDs must be exactly representable in float64; use text IDs to preserve identity")
        return result
    # Probabilities have an exact source-domain check before ordinary rounding.
    if not 0 <= value <= 1:
        raise ValueError("JSON probabilities must be in [0,1] before float64 conversion")
    result = float(value)
    if (not math.isfinite(result) or (value != 0 and result == 0)
            or (value != 1 and result == 1)):
        raise ValueError("JSON probability conversion loses a nonendpoint value at a float64 endpoint; supply supported precision")
    return result


def load_saved_binary_document(text):
    """Parse bounded JSON with explicit binary64 conversion, without file I/O.

    Duplicate keys and lossy group/label identities reject. This loader performs
    structural/numeric conversion; evaluate_saved_binary_probabilities applies
    the complete cohort contract. Numeric metrics are not exact-decimal scores.
    """
    if type(text) is not str or len(text) > MAX_JSON_BYTES:
        raise ValueError("Saved binary JSON must be text of at most 4 MiB in UTF-8")
    try:
        if len(text.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError("Saved binary JSON exceeds the 4 MiB UTF-8 limit")
    except UnicodeError:
        raise ValueError("Saved binary JSON must be valid UTF-8 text") from None
    _check_json_depth(text)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def integer(token):
        if len(token.lstrip("-")) > 100:
            raise ValueError("JSON integers exceed 100 decimal digits")
        return int(token)

    def decimal(token):
        if len(token) > 64:
            raise ValueError("JSON decimal tokens exceed 64 characters")
        try:
            value = Decimal(token)
        except InvalidOperation:
            raise ValueError("Unsupported JSON decimal") from None
        parts = value.as_tuple()
        if not value.is_finite() or len(parts.digits) > 64 or abs(parts.exponent) > 400:
            raise ValueError("JSON decimal precision or exponent exceeds supported limits")
        return value

    def constant(_token):
        raise ValueError("Nonfinite JSON constants are unsupported")

    try:
        document = json.loads(text, object_pairs_hook=pairs, parse_int=integer,
                              parse_float=decimal, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError):
        raise ValueError("Saved binary JSON is malformed or exceeds supported nesting") from None
    _keys(document, {"schema_version", "metadata", "expected_row_ids", "rows"}, "document")
    rows = document["rows"]
    if type(rows) is not list or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("Saved binary rows must contain 1 to 20000 entries")
    for row in rows:
        _keys(row, {"row_id", "group", "observed_label", "probability_1", "status"}, "row")
        for field in ("observed_label", "probability_1", "group"):
            row[field] = _json_decimal_to_native(row[field], field)
    return document


class _SavedBinaryParser(argparse.ArgumentParser):
    def error(self, _message):
        self.print_usage(sys.stderr)
        self.exit(2, "error: invalid saved binary arguments; use --help. Supplied values are not echoed.\n")


def main(argv=None):
    parser = _SavedBinaryParser(prog="python -m groupbench.binary.saved_evaluation",
                               description=__doc__, allow_abbrev=False)
    parser.add_argument("input", type=Path, help="Regular local UTF-8 saved-binary JSON; complete report goes to stdout only")
    args = parser.parse_args(argv)
    try:
        document = load_saved_binary_document(_read_input(args.input))
        report = evaluate_saved_binary_probabilities(document)
        output = json.dumps(report, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=True)
    except (ValueError, TypeError, OSError, OverflowError):
        print("saved binary evaluation rejected: invalid or unavailable local input, unsupported numeric precision, or incomplete cohort; no scores were written.", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
