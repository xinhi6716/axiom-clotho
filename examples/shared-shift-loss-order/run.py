"""Exact paired ranking under two losses and one shared nuisance shift."""
import json
from fractions import Fraction

from groupbench.metrics import paired_mae_shared_shift_bounds


def contrast(a, b, limits):
    roster = tuple(str(i) for i in range(len(a)))
    rows = tuple((i, 0, x, y, 1) for i, x, y in zip(roster, a, b))
    mae = paired_mae_shared_shift_bounds(rows, expected_roster=roster, shift_bounds=limits)
    squared = sorted(
        sum((Fraction(y) - c)**2 - (Fraction(x) - c)**2 for x, y in zip(a, b)) / len(a)
        for c in map(Fraction, limits)
    )
    return {
        "mae_range": [str(mae[k]) for k in ("min_mae_b_minus_a", "max_mae_b_minus_a")],
        "mae_endpoints": [str(mae[k]) for k in ("at_lower_shift_mae_b_minus_a", "at_upper_shift_mae_b_minus_a")],
        "mse_range": list(map(str, squared)),
    }


def demonstrate():
    return {
        "opposite_robust_rankings": contrast((-4, 0, 4), (-3, -3, 3), (-0.25, 0.25)),
        "interior_mae_extremum": contrast((0, 0), (-0.5, 0.5), (-1, 1)),
    }


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
