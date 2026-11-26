"""Compare two declared calibration-dependence designs on invented data."""
from fractions import Fraction
import json

from groupbench.metrics import paired_mae_shared_shift_bounds


ROWS = [
    ("a1", 0, -1, 1, 1), ("a2", 0, -1, 1, 1),
    ("b1", 0, 1, -1, 1), ("b2", 0, 1, -1, 1),
]
ROSTERS = {"path-a": ("a1", "a2"), "path-b": ("b1", "b2")}
BOUNDS = (-1, 1)


def demonstrate():
    both = tuple(key for roster in ROSTERS.values() for key in roster)
    common = paired_mae_shared_shift_bounds(ROWS, expected_roster=both, shift_bounds=BOUNDS)
    separate = []
    for roster in ROSTERS.values():
        group = [row for row in ROWS if row[0] in roster]
        separate.append(paired_mae_shared_shift_bounds(
            group, expected_roster=roster, shift_bounds=BOUNDS,
        ))

    # Mean of per-path contrasts: this assumes the stated independent axes
    # and fixed equal-path weighting. All rows happen to have equal weights.
    lower, upper = "min_mae_b_minus_a", "max_mae_b_minus_a"
    pooled = tuple(common[k] for k in (lower, upper))
    independent = tuple(sum(x[k] for x in separate) / len(separate) for k in (lower, upper))
    if pooled != (Fraction(0), Fraction(0)) or independent != (Fraction(-2), Fraction(2)):
        raise RuntimeError("Toy calibration contrast did not match the exact reference.")
    return {"common_path_range": list(map(str, pooled)),
            "independent_paths_range": list(map(str, independent))}


if __name__ == "__main__":
    print(json.dumps(demonstrate(), sort_keys=True))
