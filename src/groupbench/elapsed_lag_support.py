"""Compare exact within-group elapsed-lag multisets of two declared schedules."""

import argparse
from collections import Counter, defaultdict
from decimal import InvalidOperation
from itertools import combinations
import json
from pathlib import Path
import sys
import unicodedata

from .budget_comparison import _read_input, load_document
from .quantity_contract import _number

MAX_ASSOCIATIONS = 20_000
MAX_PAIRS_PER_SCHEDULE = 100_000


def _label(value):
    if (type(value) is not str or not 1 <= len(value) <= 256 or value != value.strip()
            or any(unicodedata.category(c) in ("Cc", "Cs") for c in value)):
        raise ValueError("Invalid declared identifier")
    return value


def _document(doc, keys, rows_key):
    if (type(doc) is not dict or set(doc) != keys | {"schema_version"}
            or type(doc["schema_version"]) is not int or doc["schema_version"] != 1):
        raise ValueError("Invalid elapsed-lag document")
    rows = doc[rows_key]
    if type(rows) is not list or not 1 <= len(rows) <= MAX_ASSOCIATIONS:
        raise ValueError("Invalid elapsed-lag row population")
    return rows


def _roster(doc):
    expected = {}
    for row in _document(doc, {"associations"}, "associations"):
        if type(row) is not list or len(row) != 2:
            raise ValueError("Expected one row identifier and group")
        ident, group = map(_label, row)
        if ident in expected:
            raise ValueError("Duplicate frozen row identifier")
        expected[ident] = group
    counts = Counter(expected.values())
    pairs = sum(n * (n - 1) // 2 for n in counts.values())
    if not 1 <= pairs <= MAX_PAIRS_PER_SCHEDULE:
        raise ValueError("Within-group pair population is empty or above the fixed limit")
    return expected, pairs


def _schedule(doc, expected):
    seen, grouped = set(), defaultdict(list)
    for row in _document(doc, {"time_unit", "rows"}, "rows"):
        if type(row) is not list or len(row) != 3:
            raise ValueError("Expected a row identifier, group and timestamp")
        ident, group = map(_label, row[:2])
        if ident in seen or expected.get(ident) != group:
            raise ValueError("Schedule contradicts the frozen association roster")
        seen.add(ident)
        grouped[group].append(_number(row[2], "Timestamp"))
    if len(seen) != len(expected):
        raise ValueError("Schedule omits declared associations")
    return _label(doc["time_unit"]), grouped


def compare_elapsed_lag_support(reference, candidate, roster):
    """Compare absolute timestamp differences between each unordered same-group pair.

    Association and unit declarations are checked, not independently verified.
    Distinct row IDs at equal timestamps contribute zero-lag occurrences.
    Pair multiplicities matter; no timestamp-to-row permutation is inferred.
    """
    expected, pairs = _roster(roster)
    ref_unit, ref = _schedule(reference, expected)
    cand_unit, cand = _schedule(candidate, expected)
    if ref_unit != cand_unit:
        raise ValueError("Time-unit declarations must match exactly")

    matched = groups_with_pairs = groups_with_mismatch = 0
    for group, timestamps in ref.items():
        if len(timestamps) < 2:
            continue
        groups_with_pairs += 1

        def histogram(values):
            return Counter(right - left for left, right in combinations(sorted(values), 2))

        ref_lags, cand_lags = histogram(timestamps), histogram(cand[group])
        groups_with_mismatch += ref_lags != cand_lags
        matched += sum((ref_lags & cand_lags).values())

    return {
        "schema_id": "mlcalc.elapsed_lag_support_comparison", "schema_version": 1,
        "claim_scope": "descriptive_declared_schedules_only",
        "n_associations_per_schedule": len(expected), "n_groups": len(ref),
        "n_groups_with_pairs": groups_with_pairs,
        "n_groups_with_mismatched_distributions": groups_with_mismatch,
        "n_pairs_per_schedule": pairs,
        "n_matched_lag_occurrences_per_schedule": matched,
        "n_unmatched_lag_occurrences_per_schedule": pairs - matched,
        "same_within_group_lag_distributions": groups_with_mismatch == 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlcalc compare-elapsed-lag-support", allow_abbrev=False,
        description="Compare exact within-group elapsed-lag multiplicities; aggregate JSON, no outcomes.")
    parser.add_argument("reference", type=Path, help="Reference v1 JSON timestamp schedule")
    parser.add_argument("candidate", type=Path, help="Candidate v1 JSON timestamp schedule")
    parser.add_argument("roster", type=Path, help="Separate frozen v1 JSON row-to-group association roster")
    args = parser.parse_args(argv)
    try:
        report = compare_elapsed_lag_support(*(load_document(_read_input(path))
                                               for path in (args.reference, args.candidate, args.roster)))
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError, OSError, InvalidOperation):
        print("elapsed lag support rejected: invalid, unavailable, excessive or mismatched inputs",
              file=sys.stderr)
        return 2
    return 0
