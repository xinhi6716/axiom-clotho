"""Same-unit MAE and optimal marginal 1-Wasserstein distance answer different questions."""
from fractions import Fraction
from itertools import permutations
import json


def distance(left, right):
    return Fraction(sum(abs(x - y) for x, y in zip(left, right)), len(left))


def comparison(target, prediction):
    if not target or target.keys() != prediction.keys():
        raise ValueError("different or empty paired rosters")
    y = list(target.values())
    p = [prediction[key] for key in target]
    unconstrained = [distance(y, perm) for perm in permutations(p)]
    return {
        "paired_mae": str(distance(y, p)),
        "sorted_distance": str(distance(sorted(y), sorted(p))),
        "exhaustive_minimum": str(min(unconstrained)),
        "optimal_index_permutations": unconstrained.count(min(unconstrained)),
    }


def fixture():
    y = {"u": 0, "v": 0, "w": 2}
    return {
        "local_error": comparison(y, {"u": 0, "v": 0, "w": 3}),
        "swapped_units": comparison(y, {"u": 2, "v": 0, "w": 0}),
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
