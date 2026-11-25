"""Separate pooled and within-stratum pair order on a frozen toy roster."""

from fractions import Fraction
from itertools import combinations


def report(rows):
    """Each row is (origin stratum, outcome, frozen score)."""
    # Each bucket has concordant, discordant, prediction tie, outcome tie.
    tally = {scope: [0, 0, 0, 0] for scope in ("pooled", "within", "between")}
    for (g, y, s), (other, target, score) in combinations(rows, 2):
        scope = "within" if g == other else "between"
        if y == target:
            category = 3
        elif s == score:
            category = 2
        else:
            category = 0 if (y < target) == (s < score) else 1
        tally[scope][category] += 1
        tally["pooled"][category] += 1
    result = {}
    for scope, (c, d, tie, outcome_tie) in tally.items():
        n = c + d + tie
        result[scope] = ((c, d, tie, outcome_tie), None if n == 0 else Fraction(2*c + tie, 2*n))
    return result


def fixture(scores):
    return tuple(zip((0, 0, 0, 1, 1, 1), (1, 2, 3, 10, 11, 12), scores))


def main():
    for name, scores in (
        ("candidate", (3, 2, 1, 12, 11, 10)),
        ("constant", (0,)*6),
        ("origin-stratum baseline", (2, 2, 2, 11, 11, 11)),
    ):
        r = report(fixture(scores))
        print(name + ": " + ", ".join(f"{scope}={data[1]}" for scope, data in r.items()))


if __name__ == "__main__":
    main()
