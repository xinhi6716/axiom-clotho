"""Enumerate the sampling law of leave-one-parent-out squared loss."""
import json
from fractions import Fraction
from itertools import product


def avg(values):
    values = tuple(values)
    return Fraction(sum(values), len(values))


def variance(values):
    values = tuple(values)
    center = avg(values)
    return avg((v - center)**2 for v in values)


def loo_losses(data):
    return tuple((y - avg(data[:i] + data[i+1:]))**2 for i, y in enumerate(data))


def summarize(n):
    worlds = tuple(product((-1, 1), repeat=n))
    scores = tuple(map(loo_losses, worlds))
    actual = variance(map(avg, scores))
    independent = avg(variance(row) for row in zip(*scores)) / n
    cov = avg(v[0] * v[1] for v in scores) - avg(v[0] for v in scores) * avg(v[1] for v in scores)
    assert actual == independent + Fraction(n - 1, n) * cov
    return (actual, independent, cov)


if __name__ == "__main__":
    results = {n: summarize(n) for n in (2, 3, 4)}
    assert results[2] == (4, 2, 4)
    assert results[3][2] == 0
    assert results[4] == (Fraction(8, 27), Fraction(10, 27), Fraction(-8, 81))
    print(json.dumps({n: dict(zip(("actual_variance", "independence_shortcut",
                                  "pairwise_fold_loss_covariance"), map(str, v)))
                      for n, v in results.items()}, indent=2))
