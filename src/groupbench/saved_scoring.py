"""Score a complete delayed-outcome cohort against saved regression predictions.

Read-only and zero-fit, not a fresh holdout or an identity/provenance guarantee.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

from .metrics import grouped_metrics
from .prediction_schema import _load_prediction_bundle, MAX_ROWS
from .verification import _signature

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
