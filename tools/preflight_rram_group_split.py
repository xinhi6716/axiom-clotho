"""Audit declared RRAM train/evaluation splits at a chosen physical grouping level."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from preflight_rram_split_read_pairing import roster

UNITS = {"chip": 1, "address": 2, "cell": 3, "episode": 4}


def support(fit, evaluation, unit):
    counts = {}
    for label, depth in UNITS.items():
        train_units = {key[:depth] for key in fit}
        eval_units = {key[:depth] for key in evaluation}
        shared = train_units & eval_units
        counts[label] = {
            "fit_distinct": len(train_units),
            "evaluation_distinct": len(eval_units),
            "shared_declared_units": len(shared),
            "evaluation_rows_on_fit_unit": sum(key[:depth] in shared for key in evaluation),
        }
    return {
        "schema": "clotho.rram_group_split.v1",
        "required_disjoint_unit": unit,
        "support_by_unit": counts,
        "declared_split_disjoint_at_required_unit": not counts[unit]["shared_declared_units"],
        "physical_genealogy_verified": False,
    }


def condition_support(fit, evaluation, unit, minimum):
    depth = UNITS[unit]
    fit_units = defaultdict(set)
    for key, condition in fit.items():
        fit_units[condition].add(key[:depth])
    eval_rows = Counter(evaluation.values())
    histogram = Counter()
    for condition in eval_rows:
        histogram[len(fit_units.get(condition, ()))] += 1
    below = {condition for condition in eval_rows if len(fit_units.get(condition, ())) < minimum}
    return {
        "minimum_distinct_fit_units_per_evaluation_condition": minimum,
        "evaluation_declared_conditions": len(eval_rows),
        "evaluation_conditions_by_distinct_fit_unit_count": dict(sorted(histogram.items())),
        "evaluation_conditions_below_minimum": len(below),
        "evaluation_rows_in_conditions_below_minimum": sum(eval_rows[c] for c in below),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Roster format: chip,address,lane,episode,condition. A declared cell is chip/address/lane. "
        "Conditions cannot make a repeated physical grouping independent. "
        "Labels must come from independent acquisition metadata; no genealogy is inferred."
    ))
    parser.add_argument("--fit-roster", type=Path, required=True)
    parser.add_argument("--evaluation-roster", type=Path, required=True)
    parser.add_argument("--disjoint-at", choices=UNITS, required=True)
    parser.add_argument("--minimum-fit-units-per-condition", type=int,
                        help="require this many distinct fit units at the chosen level for each evaluation condition")
    args = parser.parse_args()
    minimum = args.minimum_fit_units_per_condition
    if minimum is not None and minimum < 1:
        parser.error("minimum fit units per condition must be positive")
    try:
        _, fit = roster(args.fit_roster)
        _, evaluation = roster(args.evaluation_roster)
        report = support(fit, evaluation, args.disjoint_at)
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: could not read UTF-8 roster CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    failed_support = False
    if minimum is not None:
        fit_support = condition_support(fit, evaluation, args.disjoint_at, minimum)
        report["declared_fit_condition_support"] = fit_support
        failed_support = bool(fit_support["evaluation_conditions_below_minimum"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(not report["declared_split_disjoint_at_required_unit"] or failed_support)


if __name__ == "__main__":
    raise SystemExit(main())
