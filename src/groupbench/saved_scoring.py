"""Score a complete delayed-outcome cohort against saved regression predictions.

Read-only and zero-fit, not a fresh holdout or an identity/provenance guarantee.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
from importlib import import_module as _import_module
import io
import json
import math
import os
from pathlib import Path
import re
import sys

from .prediction_schema import _load_prediction_bundle, MAX_ROWS
from .verification import _signature

# Preserve the former module-level metric attributes and monkeypatch targets,
# without making the bounded raw-byte reader import numerical dependencies.
_METRIC_NAMES = frozenset({"grouped_metrics", "paired_absolute_error_difference", "contrast_float"})


def __getattr__(name):
    if name not in _METRIC_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(".metrics", __package__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _METRIC_NAMES)


def _load_metrics(*names):
    for name in names:
        if name not in globals():
            __getattr__(name)


MAX_OUTCOMES_BYTES = 4 * 1024 * 1024
_POSITION = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
LIMITATIONS = [
    "Descriptive scores for the supplied complete saved cohort only; no fitting, selection, calibration or model loading.",
    "row_position refers to the original new-feature row, not a physical specimen ID. The caller must retain and verify that mapping; no identity is inferred from CSV order.",
    "Scoring groups are supplied literal labels, not verified physical independence or evidence of training/test disjointness.",
    "Outcomes must use the prediction target and units. Their meaning, timing, cohort completeness and origin are caller responsibilities, not verified by matching positions.",
    "Checksums bind scores to the bytes consumed; they are not authenticity, historical blinding, model-execution proof or an atomic filesystem snapshot.",
    "Integer-syntax scoring tokens (observed and saved prediction) must be exactly representable in float64. Other decimal tokens are rounded to float64; decimal measurement precision is not audited. Nonfinite and nonzero-to-zero conversions are rejected.",
    "Metrics use existing checked float64 arithmetic. Constant-target R2 is null. Equal-group metrics do not repair a sampling design or establish independence.",
    "Repeated scoring does not create a fresh holdout; do not tune a model on these outcomes and present the same cohort as independent validation.",
    "Successful output contains supplied group labels, observed values and predictions; review before sharing.",
]


class SavedScoringError(ValueError):
    """Categorical error messages do not expose paths or source values."""


def _score_number(token):
    if len(token) > 64 or _DECIMAL.fullmatch(token) is None:
        raise SavedScoringError("Scoring values must be finite decimal tokens of at most 64 characters, without units or whitespace.")
    if _INTEGER.fullmatch(token):
        # Keep Python integers until the existing metric validator checks exact
        # float64 representability. Inference before this check would lose bits.
        return int(token)
    try:
        value = float(token)
        if not math.isfinite(value) or (value == 0 and Decimal(token) != 0):
            raise SavedScoringError("Scoring values overflow or underflow float64; rescale units explicitly.")
        return value
    except (ValueError, InvalidOperation, OverflowError):
        raise SavedScoringError("Scoring values overflow or underflow float64; rescale units explicitly.") from None


def _outcomes(raw, n_predictions):
    if len(raw) > MAX_OUTCOMES_BYTES:
        raise SavedScoringError("Outcomes exceed the 4 MiB byte limit.")
    try:
        text = raw.decode("utf-8-sig")
        if "\0" in text:
            raise SavedScoringError("Outcomes CSV cannot contain NUL characters.")
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        if next(rows, None) != ["row_position", "group", "observed"]:
            raise SavedScoringError("Outcomes header must be exactly row_position,group,observed.")
        by_position = {}
        for row in rows:
            if len(by_position) >= min(n_predictions, MAX_ROWS) or len(row) != 3:
                raise SavedScoringError("Outcomes must have exactly one three-field row for every prediction position; no rows are dropped.")
            position, group, token = row
            if len(position) > 5 or _POSITION.fullmatch(position) is None or int(position) >= n_predictions:
                raise SavedScoringError("row_position must be canonical zero-based integer text within the saved prediction roster.")
            key = int(position)
            if key in by_position:
                raise SavedScoringError("Duplicate outcome position; every saved prediction must be matched exactly once.")
            if not group.strip() or len(group) > 256:
                raise SavedScoringError("Scoring groups must be nonblank literal text of at most 256 characters; they are not normalized.")
            by_position[key] = (group, _score_number(token))
        if len(by_position) != n_predictions:
            raise SavedScoringError("Incomplete outcome roster; every saved prediction must have exactly one observed outcome.")
        return [by_position[position] for position in range(n_predictions)]
    except (UnicodeError, csv.Error):
        raise SavedScoringError("Outcomes must be bounded valid UTF-8 CSV with consistent quoting and fields.") from None


def _read_outcomes(path):
    # Reuse the CLI's bounded regular-file opening contract, including its
    # documented allowance for local symlinks; no second pandas/type inference.
    from .cli import _open_csv_source
    path = Path(path)
    try:
        with _open_csv_source(path) as source:
            before = os.fstat(source.fileno())
            if before.st_size > MAX_OUTCOMES_BYTES:
                raise SavedScoringError("Outcomes exceed the 4 MiB byte limit.")
            chunks, size = [], 0
            while True:
                chunk = source.read(min(65536, MAX_OUTCOMES_BYTES - size + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_OUTCOMES_BYTES:
                    raise SavedScoringError("Outcomes exceed the 4 MiB byte limit.")
            if (size != before.st_size or _signature(os.fstat(source.fileno())) != _signature(before)
                    or _signature(path.stat()) != _signature(before)):
                raise SavedScoringError("Outcomes changed during reading; retry with an idle local file.")
    except SavedScoringError:
        raise
    except (OSError, ValueError):
        raise SavedScoringError("Outcomes must be an available bounded regular local file.") from None
    raw = b"".join(chunks)
    return raw, hashlib.sha256(raw).hexdigest()


def score_saved_predictions(run_dir, outcomes_csv):
    """Return complete descriptive JSON-ready scores, or reject the whole cohort.

    No writes or fits. Exact bundle bytes returned by the existing pinned-reader
    validator are scored; the prediction file is not reopened after validation.
    """
    try:
        schema, _manifest, prediction_raw, hashes = _load_prediction_bundle(run_dir)
    except (OSError, ValueError):
        raise SavedScoringError("Prediction bundle failed bounded byte/schema verification; inspect it with validate-prediction-bundle. No scores were produced.") from None
    outcome_raw, outcomes_hash = _read_outcomes(outcomes_csv)
    outcomes = _outcomes(outcome_raw, schema.n_predictions)
    return _score_loaded_bundle((schema, _manifest, prediction_raw, hashes), outcomes, outcomes_hash)


def _score_loaded_bundle(loaded, outcomes, outcomes_hash):
    """Score retained verified bytes and one canonical parsed outcome roster."""
    _load_metrics("grouped_metrics")

    schema, _manifest, prediction_raw, hashes = loaded
    # The shared validator checked structure on these exact bytes. Scoring
    # additionally rejects underflow and integer precision loss on both sides;
    # this narrower scoring subset does not change public bundle validation.
    prediction_rows = csv.reader(io.StringIO(prediction_raw.decode("utf-8"), newline=""), strict=True)
    next(prediction_rows)
    predictions = [_score_number(row[1]) for row in prediction_rows]
    groups = [row[0] for row in outcomes]
    observed = [row[1] for row in outcomes]
    try:
        overall, per_group = grouped_metrics(observed, predictions, groups)
    except (ValueError, FloatingPointError, OverflowError):
        raise SavedScoringError("Metric inputs or float64 arithmetic are unsupported; use exactly representable integers and rescale outcomes and predictions to practical units before scoring.") from None
    return {
        "schema_version": "tabular-saved-score-v1",
        "mode": "score_saved_regression_predictions", "fits_performed": 0,
        "n_rows": schema.n_predictions, "n_groups": overall["n_groups"],
        "prediction_bundle": {"schema_version": schema.schema_version, "files_sha256": dict(hashes)},
        "outcomes_sha256": outcomes_hash,
        "overall": overall, "per_group": per_group,
        "scored_rows": [{"row_position": position, "group": group,
                         "observed": value, "prediction": predictions[position]}
                        for position, (group, value) in enumerate(outcomes)],
        "limitations": list(LIMITATIONS),
    }


_COMPARISON_CONTRACT = frozenset({
    "method_a", "method_b", "target_column", "target_definition", "target_unit",
    "cohort_definition", "row_mapping_definition", "group_definition",
})


def _validate_comparison_contract(contract):
    if type(contract) is not dict or set(contract) != _COMPARISON_CONTRACT:
        raise SavedScoringError("Comparison requires the exact common semantic declaration fields.")
    for key, value in contract.items():
        limit = 256 if key in ("method_a", "method_b") else 2048
        if type(value) is not str or not value.strip() or len(value) > limit or "\0" in value:
            raise SavedScoringError("Comparison declarations must be bounded nonblank text without NUL.")
    if contract["method_a"] == contract["method_b"]:
        raise SavedScoringError("Comparison method labels must be distinct.")


def compare_saved_predictions(run_a, run_b, query_csv, outcomes_csv, contract):
    """Descriptively compare two saved bundles linked to identical query bytes.

    A shared source hash binds declared bytes, not physical identity, producer
    behavior, units, authenticity or independence. No model is loaded or fitted.
    Requires explicit semantic declarations; no source-hash override is offered.
    """
    _validate_comparison_contract(contract)
    try:
        loaded_a = _load_prediction_bundle(run_a)
        loaded_b = _load_prediction_bundle(run_b)
    except (OSError, ValueError):
        raise SavedScoringError("Both comparison bundles must pass bounded byte/schema verification.") from None
    n = loaded_a[0].n_predictions
    if n != loaded_b[0].n_predictions:
        raise SavedScoringError("Comparison bundles must contain the same complete query row count.")
    # Use the same one-read CSV parser/hash semantics as prediction production.
    from .cli import _read_csv_with_hash
    try:
        query, query_hash = _read_csv_with_hash(Path(query_csv))
    except (OSError, ValueError, TypeError):
        raise SavedScoringError("Comparison query must be a bounded valid local CSV.") from None
    if len(query) != n:
        raise SavedScoringError("Original query rows must match both complete prediction rosters.")
    for loaded in (loaded_a, loaded_b):
        manifest = loaded[1]
        claimed_hash = manifest.get("prediction_source_file_sha256")
        if (type(claimed_hash) is not str
                or re.fullmatch(r"[0-9a-f]{64}", claimed_hash) is None
                or claimed_hash != query_hash):
            raise SavedScoringError("Exact query bytes must match both manifest source-hash declarations.")
        config = manifest["config"]
        if config.get("target") != contract["target_column"]:
            raise SavedScoringError("Both target columns must match the explicit common target declaration.")
        if any(name not in query.columns for name in config["features"]):
            raise SavedScoringError("Original query is missing a declared model feature column.")
    outcome_raw, outcome_hash = _read_outcomes(outcomes_csv)
    outcomes = _outcomes(outcome_raw, n)
    report_a = _score_loaded_bundle(loaded_a, outcomes, outcome_hash)
    report_b = _score_loaded_bundle(loaded_b, outcomes, outcome_hash)
    _load_metrics("paired_absolute_error_difference", "contrast_float")

    # grouped_metrics above validates integer precision and finite arithmetic
    # before any float conversion in the shared exact contrast primitive.
    contrasts, grouped, rows = [], {}, []
    try:
        for a, b in zip(report_a["scored_rows"], report_b["scored_rows"]):
            difference = paired_absolute_error_difference(a["prediction"], b["prediction"], a["observed"])
            contrasts.append(difference)
            grouped.setdefault(a["group"], []).append(difference)
            rows.append({"row_position": a["row_position"], "group": a["group"],
                         "observed": a["observed"], "prediction_a": a["prediction"],
                         "prediction_b": b["prediction"],
                         "absolute_error_b_minus_a": contrast_float(difference)})
        group_means = {group: sum(values, Fraction()) / len(values)
                       for group, values in grouped.items()}
        row_mean = contrast_float(sum(contrasts, Fraction()) / n)
        equal_group_mean = contrast_float(sum(group_means.values(), Fraction()) / len(group_means))
        per_group = [{"group": group, "n_rows": len(grouped[group]),
                      "mean_absolute_error_b_minus_a": contrast_float(value)}
                     for group, value in sorted(group_means.items())]
    except (ValueError, OverflowError):
        raise SavedScoringError("Paired contrast arithmetic exceeds supported numeric range.") from None
    return {
        "schema_version": "tabular-saved-paired-comparison-v1",
        "mode": "compare_saved_regression_predictions", "fits_performed": 0,
        "contract": dict(contract), "n_rows": n, "n_groups": len(group_means),
        "omitted_rows": 0, "query_sha256": query_hash, "outcomes_sha256": outcome_hash,
        "query_bytes_match_both_manifest_declarations": True,
        "physical_identity_verified": False, "provenance_verified": False,
        "identical_bundle_hashes": loaded_a[3] == loaded_b[3],
        "metric": "absolute error in declared target units; no normalization",
        "contrast_direction": "B minus A; negative is lower B absolute error on this supplied cohort only",
        "row_weighting": "equal supplied rows; not independent-unit inference",
        "group_weighting": "equal supplied groups after averaging their rows; not independence certification",
        "row_mean_absolute_error_b_minus_a": row_mean,
        "equal_group_mean_absolute_error_b_minus_a": equal_group_mean,
        "per_group": per_group, "paired_rows": rows,
        "model_a": report_a, "model_b": report_b,
        "limitations": list(LIMITATIONS) + [
            "Common query bytes match both producer declarations, not authenticated historical mapping or model execution.",
            "Target definitions, units and group/cohort/mapping meanings are caller declarations, not verified facts.",
            "Different training sources, feature subsets and protocols may be compared; this is not a controlled or causal contrast.",
            "Paired contrasts are exact on accepted binary64 inputs before display rounding; they may differ from subtraction of displayed MAEs.",
            "No hypothesis tests, intervals, automatic winner, independence or future-performance inference are supplied.",
            "Inputs are consumed sequentially, not as a joint atomic snapshot. Output includes supplied declarations and row values.",
        ],
    }


MAX_COMPARISON_CONTRACT_BYTES = 64 * 1024
MAX_COMPARISON_OUTPUT_BYTES = 64 * 1024 * 1024


def _load_comparison_contract(path):
    # Existing reader: 4 MiB bound plus one detection byte; 64 KiB accepted here.
    # Reuse its descriptor/signature checks, not a second file-opening policy.
    raw, _digest = _read_outcomes(path)
    if len(raw) > MAX_COMPARISON_CONTRACT_BYTES:
        raise SavedScoringError("Comparison contract exceeds the 64 KiB accepted-byte limit.")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate contract key")
            result[key] = value
        return result
    def number(_token):
        raise ValueError("Contract values must be strings")
    try:
        contract = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                              parse_int=number, parse_float=number, parse_constant=number)
        _validate_comparison_contract(contract)
    except (ValueError, TypeError, RecursionError):
        raise SavedScoringError("Comparison contract must be strict UTF-8 JSON with exactly the required string declarations.") from None
    return contract


class _ComparisonParser(argparse.ArgumentParser):
    def error(self, _message):
        self.print_usage(sys.stderr)
        self.exit(2, "error: invalid comparison arguments; use mlcalc compare-predictions --help. Supplied values are not echoed.\n")


def compare_main(argv=None):
    """Stdout-only entry point for the existing same-query comparison API."""
    parser = _ComparisonParser(prog="mlcalc compare-predictions", allow_abbrev=False,
                               description="Compare two complete saved regression bundles on identical query bytes; no fitting.",
                               epilog="Stdout JSON only; no output-path, overwrite or identity-bypass options. Exit 0 compared, 2 rejected.")
    parser.add_argument("--run-a", type=Path, required=True, help="Reference regression prediction bundle")
    parser.add_argument("--run-b", type=Path, required=True, help="Candidate regression prediction bundle")
    parser.add_argument("--query", type=Path, required=True, help="Exact original query CSV shared by both bundles")
    parser.add_argument("--outcomes", type=Path, required=True, help="Complete shared row_position,group,observed CSV")
    parser.add_argument("--contract", type=Path, required=True, help="Strict UTF-8 JSON semantic declarations (accepted <=64 KiB)")
    args = parser.parse_args(argv)
    try:
        contract = _load_comparison_contract(args.contract)
        report = compare_saved_predictions(args.run_a, args.run_b, args.query, args.outcomes, contract)
        # Complete and bound serialization before any stdout, never partial JSON.
        output = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        if len(output.encode("utf-8")) + 1 > MAX_COMPARISON_OUTPUT_BYTES:
            raise ValueError("Complete comparison output exceeds supported size")
    except (SavedScoringError, OSError, ValueError, TypeError, OverflowError):
        print("saved comparison rejected: invalid or unavailable local inputs, unsupported numeric precision, mismatched query or incomplete cohort; no comparison was written.", file=sys.stderr)
        return 2
    print(output)
    return 0


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        self.print_usage(sys.stderr)
        self.exit(2, "error: invalid saved-scoring arguments; use mlcalc score-predictions --help. Supplied values are not echoed.\n")


def main(argv=None):
    parser = _Parser(prog="mlcalc score-predictions", description=__doc__, allow_abbrev=False,
                     epilog="Stdout JSON only; no files written. Requires the existing ML dependencies and POSIX safe bundle access. Exit 0 scored, 2 rejected.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Complete supported regression prediction bundle")
    parser.add_argument("--outcomes", type=Path, required=True, help="UTF-8 CSV: row_position,group,observed; full saved cohort")
    args = parser.parse_args(argv)
    try:
        # Serialize completely before writing stdout; no partial JSON on failure.
        result = json.dumps(score_saved_predictions(args.run_dir, args.outcomes),
                            indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    except SavedScoringError as error:
        print("saved prediction scoring rejected: " + str(error), file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError):
        print("saved prediction scoring rejected: local inputs or complete JSON output are unavailable.", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
