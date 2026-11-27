"""Unit provenance can reverse a saved model comparison."""
from fractions import Fraction as Q
import json

TO_MS = {"ms": 1, "s": 1000}


def score_saved(rows, certified_unit, origin_ms):
    scale = TO_MS.get(certified_unit)
    inferred = []
    for stamp, candidate, baseline, outcome in rows:
        available = scale is not None and stamp * scale <= origin_ms
        predicted = candidate if available else Q(1, 2)
        inferred.append((available, (predicted - outcome) ** 2, (baseline - outcome) ** 2))
    n = len(inferred)
    a = sum(x[1] for x in inferred) / n
    b = sum(x[2] for x in inferred) / n
    return {"available": sum(x[0] for x in inferred),
            "roster_size": n, "brier_a": str(a), "brier_b": str(b),
            "brier_a_minus_b": str(a - b)}


def fixture():
    # Same raw ready stamps, feature-derived candidates, independently ready
    # comparators and target labels in every scenario.
    rows = ((2, Q(0), Q(1, 4), 0), (4, Q(1), Q(3, 4), 1))
    return {
        "hypothetical_certified_ms": score_saved(rows, "ms", 10),
        "hypothetical_certified_s": score_saved(rows, "s", 10),
        "conflicting_unresolved_metadata": score_saved(rows, None, 10),
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
