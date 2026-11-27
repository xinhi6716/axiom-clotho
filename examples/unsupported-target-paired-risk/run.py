"""Sharp paired binary-score bounds on an incomplete target roster."""
from fractions import Fraction as Q
from itertools import product
import json


def gap(a, b, y):
    return (a - y) ** 2 - (b - y) ** 2


def bounds(values):
    return [str(min(values)), str(max(values))]


def fixture():
    observed = ((Q(0), Q(1, 2), 0),) * 3
    unsupported = ((Q(1, 4), Q(3, 4)),) * 2
    known = sum(gap(a, b, y) for a, b, y in observed)
    n = len(observed) + len(unsupported)
    completions = [
        (known + sum(gap(a, b, y) for (a, b), y in zip(unsupported, labels))) / n
        for labels in product((0, 1), repeat=len(unsupported))
    ]
    generic = [(known + sign * len(unsupported)) / n for sign in (-1, 1)]
    return {
        "target_size": n,
        "labeled_overlap": len(observed),
        "overlap_brier_gap_a_minus_b": str(known / len(observed)),
        "generic_bounds": bounds(generic),
        "paired_bounds": bounds(completions),
        "paired_completions": [str(value) for value in completions],
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
