"""Identical pairwise endpoint tables can hide different device paths."""

from collections import Counter
from fractions import Fraction
from itertools import combinations, product


def roster(parity):
    return tuple(path for path in product((0, 1), repeat=3) if sum(path) % 2 == parity)


def pair_tables(rows):
    return tuple(Counter((r[i], r[j]) for r in rows)
                 for i, j in combinations(range(3), 2))


def recovery(rows):
    """Share back in state 0 among paths that started there and reached 1."""
    eligible = [r for r in rows if r[:2] == (0, 1)]
    return Fraction(sum(r[2] == 0 for r in eligible), len(eligible)) if eligible else None


if __name__ == "__main__":
    even, odd = roster(0), roster(1)
    print(f"all three pair tables agree: {pair_tables(even) == pair_tables(odd)}")
    print(f"conditional recovery, even vs odd: {recovery(even)} vs {recovery(odd)}")
