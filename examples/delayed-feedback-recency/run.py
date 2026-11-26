"""Exact replay of two trailing-one update rules on invented delayed feedback."""
import json
from fractions import Fraction

HISTORY = ((1, 5, 0), (2, 3, 1))  # event tick, arrival tick, outcome


def forecast(history, cutoff, key):
    ready = [row for row in history if row[1] < cutoff]
    return Fraction(max(ready, key=lambda row: row[key])[2]) if ready else Fraction(1, 2)


def demonstrate():
    cutoffs = (4, 6)
    labels = (1, 1)
    policies = {
        "last_arrival": [forecast(HISTORY, cutoff, 1) for cutoff in cutoffs],
        "last_event_ready": [forecast(HISTORY, cutoff, 0) for cutoff in cutoffs],
        "frozen_half": [Fraction(1, 2)] * len(cutoffs),
    }
    return {name: {
        "forecasts": list(map(str, predictions)),
        "brier": str(sum((y - f)**2 for y, f in zip(labels, predictions)) / len(labels)),
    } for name, predictions in policies.items()}


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
