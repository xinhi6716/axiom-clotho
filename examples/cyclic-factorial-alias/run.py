"""Exact three-factor example: pairwise balance is not conditional support."""

from collections import Counter
from itertools import product


FACTORS = range(3)
DESIGN = tuple((x, y, (x + y) % 3) for x, y in product(FACTORS, repeat=2))


def powder_world(x, y, z):
    return z


def interaction_world(x, y, z):
    return (x + y) % 3


def conditional_support(rows):
    # A conditional contrast only exists when one fixed (x,y) cell has
    # multiple z levels, regardless of how balanced the pair margins look.
    by_cell = {}
    for x, y, z in rows:
        by_cell.setdefault((x, y), set()).add(z)
    return tuple(cell for cell, levels in sorted(by_cell.items()) if len(levels) >= 2)


def audit():
    return {
        "pair_margins": tuple(Counter((r[i], r[j]) for r in DESIGN)
                              for i, j in ((0, 1), (0, 2), (1, 2))),
        "observed_worlds_equal": all(powder_world(*r) == interaction_world(*r)
                                     for r in DESIGN),
        "in_design_contrasts": conditional_support(DESIGN),
        "augmented_contrasts": conditional_support(DESIGN + ((0, 0, 2),)),
        "conditional_deltas": tuple(f(0, 0, 2) - f(0, 0, 0)
                                     for f in (powder_world, interaction_world)),
    }


if __name__ == "__main__":
    result = audit()
    print("pair margins balanced:", all(c == Counter({(i, j): 1 for i, j in
                                        product(FACTORS, repeat=2)}) for c in result["pair_margins"]))
    print("matching observed worlds:", result["observed_worlds_equal"])
    print("independently varied third-factor cells:", len(result["in_design_contrasts"]))
    print("new matched contrast responses:", result["conditional_deltas"])
