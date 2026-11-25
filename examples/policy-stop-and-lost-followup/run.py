"""Exact fixed-horizon arithmetic for a toy initiated-attempt roster."""

from fractions import Fraction
from itertools import groupby

HORIZON = 5
COMPLETE = ((1, "policy_stop"), (2, "success"), (3, "success"), (5, "event_free_to_horizon"))
INCOMPLETE = ((1, "lost"), *COMPLETE[1:])


def evaluate(attempts):
    """Known absorbing policy stop is different from incomplete follow-up."""
    if not attempts:
        raise ValueError("An initiated-attempt roster is required.")
    successes = missing = 0
    alive = len(attempts)
    survival_if_stops_erased = Fraction(1)
    for tick, rows in groupby(sorted(attempts), key=lambda row: row[0]):
        counts = {"success": 0, "lost": 0, "policy_stop": 0, "event_free_to_horizon": 0}
        for _, status in rows:
            if (type(tick) is not int or not 0 <= tick <= HORIZON or status not in counts
                    or (status == "event_free_to_horizon" and tick != HORIZON)
                    or (status == "lost" and tick == HORIZON)):
                raise ValueError("Outcome is inconsistent with the fixed horizon.")
            counts[status] += 1
        d = counts["success"]
        successes += d
        missing += counts["lost"]
        # Anti-pattern for the demonstration: treat known absorbing stops as
        # censoring; it is not the all-initiated policy's success incidence.
        survival_if_stops_erased *= Fraction(alive - d, alive)
        alive -= sum(counts.values())
    n = len(attempts)
    return (Fraction(successes, n), Fraction(successes + missing, n),
            1 - survival_if_stops_erased)


def main():
    lo, hi, wrong = evaluate(COMPLETE)
    assert lo == hi
    print(f"known policy stop: deployed-policy success={lo}; wrongly censoring it gives {wrong}")
    lo, hi, _ = evaluate(INCOMPLETE)
    print(f"lost follow-up: success fraction can range from {lo} to {hi}")


if __name__ == "__main__":
    main()
