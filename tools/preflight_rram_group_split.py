"""Audit declared RRAM train/evaluation splits at a chosen physical grouping level."""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Roster format: chip,address,lane,episode,condition. A declared cell is chip/address/lane. "
        "Conditions cannot make a repeated physical grouping independent. "
        "Labels must come from independent acquisition metadata; no genealogy is inferred."
    ))
    parser.add_argument("--fit-roster", type=Path, required=True)
    parser.add_argument("--evaluation-roster", type=Path, required=True)
    parser.add_argument("--disjoint-at", choices=UNITS, required=True)
    args = parser.parse_args()
    try:
        _, fit = roster(args.fit_roster)
        _, evaluation = roster(args.evaluation_roster)
        report = support(fit, evaluation, args.disjoint_at)
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: could not read UTF-8 roster CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(not report["declared_split_disjoint_at_required_unit"])


if __name__ == "__main__":
    raise SystemExit(main())
