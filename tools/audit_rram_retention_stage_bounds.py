"""Exact missing-disposition bounds for first observed threshold failure."""
import argparse
import json
from math import lcm


def identified_bounds(attempted, ramp_failures, ramp_unknown, later_failures, later_unknown,
                      later_failures_with_unknown_ramp=0):
    """Return sharp rational interval endpoints over all admissible stage completions."""
    counts = attempted, ramp_failures, ramp_unknown, later_failures, later_unknown, later_failures_with_unknown_ramp
    if any(type(value) is not int or value < 0 for value in counts) or not attempted:
        raise ValueError("counts must be nonnegative integers and attempted must be positive")
    survivors = attempted - ramp_failures - ramp_unknown
    if survivors < 0:
        raise ValueError("ramp failures and unknowns exceed the attempted cohort")
    if later_failures + later_unknown > survivors:
        raise ValueError("later failures and unknowns exceed documented ramp survivors")

    if later_failures_with_unknown_ramp > ramp_unknown:
        raise ValueError("later failures with unknown ramp exceed unknown ramp dispositions")

    def interval(lo, hi, denominator):
        return {"lower_numerator": lo, "upper_numerator": hi, "denominator": denominator}

    conditional = None
    if survivors:
        # Lower witnessed by assigning later-observed failures with unknown
        # ramp to ramp failures, and other ramp unknowns to nonfailing survivors.
        # Including one such failure as a survivor adds one to numerator and
        # denominator, which cannot lower a probability. For the upper bound,
        # assign every ramp unknown and later unknown to survival then failure.
        min_risk_survivors = survivors + ramp_unknown - later_failures_with_unknown_ramp
        max_risk_survivors = survivors + ramp_unknown
        denominator = lcm(min_risk_survivors, max_risk_survivors)
        conditional = interval(
            later_failures * (denominator // min_risk_survivors),
            (later_failures + later_unknown + ramp_unknown) * (denominator // max_risk_survivors),
            denominator,
        )
    report = {
        "schema": "clotho.rram_retention_stage_bounds.v1",
        "estimand": "first observed threshold failure across disjoint stages",
        "attempted": attempted,
        "ramp_failures": ramp_failures,
        "ramp_unknown": ramp_unknown,
        "documented_ramp_survivors": survivors,
        "later_failures_among_documented_ramp_survivors": later_failures,
        "later_unknown_among_documented_ramp_survivors": later_unknown,
        "attempted_cohort_bounds": interval(
            ramp_failures + later_failures + later_failures_with_unknown_ramp,
            ramp_failures + later_failures + later_unknown + ramp_unknown,
            attempted,
        ),
        "conditional_on_actual_ramp_survival_bounds": conditional,
        "conditional_status": ("defined_with_documented_ramp_survivors" if survivors
                               else "nonempty_survivor_cohort_not_established"),
    }
    if later_failures_with_unknown_ramp:
        report["later_failures_with_unknown_ramp"] = later_failures_with_unknown_ramp
        if survivors:
            report["conditional_bound_extremizing_survivor_counts"] = {
                "at_lower_bound_witness": min_risk_survivors,
                "at_upper_bound_witness": max_risk_survivors,
            }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Use a fixed attempted roster. Later counts apply only to documented ramp survivors. "
        "Intervals are partial identification bounds, not confidence intervals or evidence about latent crossing."
    ))
    for option in ("attempted", "ramp-failures", "ramp-unknown", "later-failures", "later-unknown"):
        parser.add_argument(f"--{option}", type=int, required=True)
    parser.add_argument("--later-failures-with-unknown-ramp", type=int, default=0,
                        help="subset of ramp unknowns with a documented later failure, without known first stage")
    args = parser.parse_args()
    try:
        report = identified_bounds(**vars(args))
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
