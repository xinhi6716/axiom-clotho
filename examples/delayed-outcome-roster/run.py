"""Read-only toy: frozen predictions with labels released at separate ticks."""

from decimal import Decimal
from fractions import Fraction
import json

from groupbench.interval_evaluation import compare_saved_interval_scores
from groupbench.metrics import paired_mae_missing_bounds


ROSTER = ("e0", "e1", "e2", "e3", "l0", "l1")
FORECASTS = {row: (0, 3) for row in ROSTER}
ARRIVAL_TICK = {row: (1 if row.startswith("e") else 6) for row in ROSTER}
# Invented endpoint labels; the cutoff rule is a workflow illustration, not ACL.
LABELS = {row: (1 if row.startswith("e") else 3) for row in ROSTER}
META = {"target_definition": "invented endpoint", "value_unit": "toy unit",
        "group_definition": "arrival batch (not physical independence)",
        "cohort_definition": "same six predeclared rows"}


def snapshot(cutoff):
    """A label at exactly cutoff is not yet included in this strict as-of rule."""
    available = {row: LABELS[row] for row in ROSTER if ARRIVAL_TICK[row] < cutoff}
    bounds = paired_mae_missing_bounds(
        [(row, available.get(row), *FORECASTS[row]) for row in ROSTER],
        expected_roster=ROSTER,
    )
    groups = {"e": "early arrival", "l": "late arrival"}

    def interval_document(arm):
        return {
            "schema_version": 1, "metadata": META, "expected_row_ids": list(ROSTER),
            "alpha": Decimal("0.2"),
            "rows": [{"row_id": row, "group": groups[row[0]], "observed": target,
                      "lower": FORECASTS[row][arm] - 1, "upper": FORECASTS[row][arm] + 1,
                      "status": "ok"} for row, target in available.items()],
        }

    a, b = interval_document(0), interval_document(1)
    try:
        score = compare_saved_interval_scores(a, b)["row_weighted"]["b_minus_a"]["interval_score_exact"]
    except ValueError as exc:
        expected_rejection = ("Missing or extra row:" if available
                              else "expected_row_ids and rows must be nonempty")
        if len(available) == len(ROSTER) or not str(exc).startswith(expected_rejection):
            raise
        interval = {"status": "incomplete_roster_not_scored", "reason": str(exc)}
    else:
        interval = {"status": "complete_roster_scored", "interval_score_b_minus_a_exact": score}
    return {
        "scoring_cutoff": cutoff,
        "pairs": {key: str(value) if isinstance(value, Fraction) else value
                  for key, value in bounds.items()},
        "paired_intervals": interval,
    }


if __name__ == "__main__":
    print(json.dumps([snapshot(cutoff) for cutoff in (2, 6, 7)], indent=2))
