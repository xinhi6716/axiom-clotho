"""Compare declared schedule support and control availability without outcomes."""

import argparse
from collections import Counter, defaultdict
from decimal import InvalidOperation
from itertools import combinations, pairwise
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


def _associations(doc):
    expected = {}
    for row in _document(doc, {"associations"}, "associations"):
        if type(row) is not list or len(row) != 2:
            raise ValueError("Expected one row identifier and group")
        ident, group = map(_label, row)
        if ident in expected:
            raise ValueError("Duplicate frozen row identifier")
        expected[ident] = group
    return expected


def _roster(doc):
    expected = _associations(doc)
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


def compare_control_availability(schedule, roster):
    """Compare two information cutoffs for each frozen forecast association.

    A control is counted only when its declared release precedes the cutoff
    strictly. Equal timestamps establish no event order. This does not read
    control values or prove the release times, sensor clock, or effect.
    """
    expected = _associations(roster)
    rows = _document(schedule, {"time_unit", "rows"}, "rows")
    _label(schedule["time_unit"])
    seen, grouped = set(), defaultdict(list)
    both = update_only = 0
    for row in rows:
        if type(row) is not list or len(row) != 6:
            raise ValueError("Expected an association plus two cutoffs, issue and control release")
        ident, group = map(_label, row[:2])
        if ident in seen or expected.get(ident) != group:
            raise ValueError("Schedule contradicts frozen associations")
        seen.add(ident)
        fixed, update, issue, release = (_number(value, "Timestamp") for value in row[2:])
        if not fixed <= update <= issue:
            raise ValueError("Cutoffs must not exceed issue time or shrink the fixed window")
        grouped[group].append((issue, fixed, update))
        if release < fixed:
            both += 1
        elif release < update:
            update_only += 1
    if len(seen) != len(expected):
        raise ValueError("Schedule omits declared associations")

    for entries in grouped.values():
        entries.sort()
        prev_issue, group_origin, prev_update = entries[0]
        for issue, fixed, update in entries[1:]:
            if fixed != group_origin or update < prev_update or (issue == prev_issue and update != prev_update):
                raise ValueError("Fixed origins or ordered update cutoffs disagree")
            prev_issue, prev_update = issue, update

    return {
        "schema_id": "mlcalc.control_availability_comparison",
        "schema_version": 1,
        "claim_scope": "descriptive_declared_cutoffs_only",
        "release_rule": "strictly_before_cutoff",
        "n_rows": len(expected),
        "n_groups": len(grouped),
        "n_available_to_both": both,
        "n_available_only_to_update": update_only,
        "n_available_to_neither": len(expected) - both - update_only,
    }


def control_availability_main(argv=None):
    parser = argparse.ArgumentParser(prog="mlcalc compare-control-availability", allow_abbrev=False,
        description="Compare fixed versus updated declared control availability; no outcomes read.")
    parser.add_argument("schedule", type=Path, help="Local v1 paired control/cutoff schedule JSON")
    parser.add_argument("roster", type=Path, help="Separate frozen v1 row-to-group association roster JSON")
    args = parser.parse_args(argv)
    try:
        report = compare_control_availability(load_document(_read_input(args.schedule)),
                                              load_document(_read_input(args.roster)))
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, TypeError, InvalidOperation):
        print("control availability comparison rejected: invalid, unavailable or mismatched inputs",
              file=sys.stderr)
        return 2
    return 0


def audit_schedule_gaps(schedule, contract):
    """Count adjacent elapsed gaps above a frozen bound, without interpolation."""
    associations = _document(contract, {"associations", "time_unit", "max_gap"}, "associations")
    expected = _associations({"schema_version": 1, "associations": associations})
    expected_unit = _label(contract["time_unit"])
    maximum = _number(contract["max_gap"], "Maximum gap")
    if maximum < 0:
        raise ValueError("Maximum gap must be nonnegative")

    schedule_unit, grouped = _schedule(schedule, expected)
    if schedule_unit != expected_unit:
        raise ValueError("Time-unit declarations must match exactly")

    above = tied = zero = singleton = affected = 0
    for timestamps in grouped.values():
        singleton += len(timestamps) == 1
        exceeds = False
        for left, right in pairwise(sorted(timestamps)):
            gap = right - left
            zero += gap == 0
            tied += gap == maximum
            if gap > maximum:
                above += 1
                exceeds = True
        affected += exceeds

    return {
        "schema_id": "mlcalc.schedule_gap_audit", "schema_version": 1,
        "claim_scope": "descriptive_declared_schedule_only",
        "n_rows": len(expected), "n_groups": len(grouped),
        "n_singleton_groups": singleton,
        "n_adjacent_gaps": len(expected) - len(grouped),
        "n_zero_gaps": zero, "n_gaps_equal_maximum": tied,
        "n_gaps_above_maximum": above, "n_groups_with_above_maximum_gaps": affected,
    }


def schedule_gap_main(argv=None):
    parser = argparse.ArgumentParser(prog="mlcalc audit-schedule-gaps", allow_abbrev=False,
        description="Audit adjacent within-group schedule gaps; aggregate counts, no outcomes.")
    parser.add_argument("schedule", type=Path, help="Local v1 timestamp schedule JSON")
    parser.add_argument("contract", type=Path, help="Separate frozen v1 associations, time unit, and maximum gap")
    args = parser.parse_args(argv)
    try:
        report = audit_schedule_gaps(load_document(_read_input(args.schedule)),
                                     load_document(_read_input(args.contract)))
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError, OSError, InvalidOperation):
        print("schedule gap audit rejected: invalid, unavailable or mismatched inputs", file=sys.stderr)
        return 2
    return 0
