"""Exact missing-disposition bounds for first observed threshold failure."""
import argparse
import json


def identified_bounds(attempted, ramp_failures, ramp_unknown, later_failures, later_unknown):
    """Return sharp rational interval endpoints over all admissible stage completions."""
    counts = attempted, ramp_failures, ramp_unknown, later_failures, later_unknown
    if any(type(value) is not int or value < 0 for value in counts) or not attempted:
        raise ValueError("counts must be nonnegative integers and attempted must be positive")
    survivors = attempted - ramp_failures - ramp_unknown
    if survivors < 0:
        raise ValueError("ramp failures and unknowns exceed the attempted cohort")
    if later_failures + later_unknown > survivors:
        raise ValueError("later failures and unknowns exceed documented ramp survivors")

    def interval(lo, hi, denominator):
        return {"lower_numerator": lo, "upper_numerator": hi, "denominator": denominator}

    return {
        "schema": "clotho.rram_retention_stage_bounds.v1",
        "estimand": "first observed threshold failure across disjoint stages",
        "attempted": attempted,
        "ramp_failures": ramp_failures,
        "ramp_unknown": ramp_unknown,
        "documented_ramp_survivors": survivors,
        "later_failures_among_documented_ramp_survivors": later_failures,
        "later_unknown_among_documented_ramp_survivors": later_unknown,
        "attempted_cohort_bounds": interval(
            ramp_failures + later_failures, ramp_failures + later_failures + later_unknown + ramp_unknown,
            attempted,
        ),
        "conditional_on_actual_ramp_survival_bounds": (
            interval(later_failures, later_failures + later_unknown + ramp_unknown, survivors + ramp_unknown)
            if survivors else None
        ),
        "conditional_status": ("defined_with_documented_ramp_survivors" if survivors
                               else "nonempty_survivor_cohort_not_established"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Use a fixed attempted roster. Later counts apply only to documented ramp survivors. "
        "Intervals are partial identification bounds, not confidence intervals or evidence about latent crossing."
    ))
    for option in ("attempted", "ramp-failures", "ramp-unknown", "later-failures", "later-unknown"):
        parser.add_argument(f"--{option}", type=int, required=True)
    args = parser.parse_args()
    try:
        report = identified_bounds(**vars(args))
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
