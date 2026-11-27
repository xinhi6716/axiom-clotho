"""Enclose a frozen nearest-neighbor rank over a certified operand box."""
from fractions import Fraction as Q
import json


def score_bounds(references, box, k):
    if not box or any(lo > hi for lo, hi in box):
        raise ValueError("invalid operand box")
    if not 1 <= k <= len(references) or any(len(row) != len(box) for row in references):
        raise ValueError("invalid neighbors or dimensions")

    distances = []
    for row in references:
        lower = upper = 0
        for value, (lo, hi) in zip(row, box):
            lower += max(lo - value, value - hi, 0) ** 2
            upper += max(abs(value - lo), abs(value - hi)) ** 2
        distances.append((lower, upper))
    return tuple(sorted(values)[k - 1] for values in zip(*distances))


def rank_enclosure(calibration, score_interval, alpha):
    lo, hi = score_interval
    def p(score):
        return Q(1 + sum(value >= score for value in calibration), len(calibration) + 1)
    least, greatest = p(hi), p(lo)
    status = ("alert" if greatest <= alpha else
              "no_alert" if least > alpha else "unresolved")
    return {"score": [str(lo), str(hi)], "p": [str(least), str(greatest)], "status": status}


def fixture():
    refs = [(-1, 0), (1, 0)]
    def report(box, k):
        return rank_enclosure([Q(1, 2)], score_bounds(refs, box, k), Q(1, 2))

    box = [(-1, 1), (0, 1)]
    return {
        "coarse_box_k2": report(box, 2),
        "geometrically_tightened_k2": rank_enclosure([Q(1, 2)], (1, 5), Q(1, 2)),
        "coarse_box_k1": report(box, 1),
        "k1_feasible_no_alert": report([(-1, -1), (0, 0)], 1),
        "k1_feasible_alert": report([(0, 0), (1, 1)], 1),
        "score_tie_no_alert": rank_enclosure([Q(1, 2)], (Q(1, 2), Q(1, 2)), Q(1, 2)),
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
