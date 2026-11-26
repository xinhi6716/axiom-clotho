"""Exact sharp-null examples for two distinct physical assignment mechanisms."""
import json
from fractions import Fraction
from itertools import combinations

UNITS = range(6)
COVARIATE = (0, 0, 0, 1, 1, 2)


def tails(outcomes, observed):
    complete = tuple(combinations(UNITS, 3))
    accepted = tuple(s for s in complete if sum(COVARIATE[i] for i in s) == 2)
    if observed not in accepted:
        raise ValueError("Observed assignment violated the preregistered balance rule")

    def statistic(s):
        return abs(2 * sum(outcomes[i] for i in s) - sum(outcomes))

    def tail(support):
        return Fraction(sum(statistic(s) >= statistic(observed) for s in support), len(support))

    return len(accepted), len(complete), tail(accepted), tail(complete)


def main():
    examples = (((0, 1, 2, 6, 7, 12), (0, 1, 5)),
                ((1, 10, 4, 8, 9, 5), (0, 2, 5)))
    for outcomes, observed in examples:
        k, n, conditional, unconditional = tails(outcomes, observed)
        print(json.dumps(dict(accepted=k, complete=n, balanced_tail=str(conditional),
                              unrestricted_tail=str(unconditional))))


if __name__ == "__main__":
    main()
