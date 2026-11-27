"""Test a fixed max-score rank under two different calibrating dependencies."""
from fractions import Fraction
import json


def rank(calibration, query):
    return Fraction(1 + sum(value >= query for value in calibration), len(calibration) + 1)


def report(rows, query, alpha):
    marginal = [rank(values, score) for values, score in zip(zip(*rows), query)]
    scores = [max(row) for row in rows]
    score = max(query)
    p = rank(scores, score)
    pooled = scores + [score]
    reject_count = sum(
        rank(pooled[:j] + pooled[j + 1:], value) <= alpha
        for j, value in enumerate(pooled)
    )
    return {
        "component_ranks": list(map(str, marginal)),
        "max_rank": str(p),
        "flag_at_alpha": p <= alpha,
        "uniform_query_rejection_count_of_20": reject_count,
    }


def fixture():
    alpha = Fraction(1, 10)
    aligned = [(1, 1)] + [(0, 0)] * 18
    separated = [(1, 0), (0, 1)] + [(0, 0)] * 17
    query = (1, 1)
    return {
        "aligned": report(aligned, query, alpha),
        "separated": report(separated, query, alpha),
        "intact_row_order_control": report(aligned[::-1], query, alpha),
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
