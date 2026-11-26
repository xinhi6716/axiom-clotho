"""Exact sign bounds on an ideal two-terminal series path."""
import json
from fractions import Fraction


def cell_change_bounds(early, late, min_path_change, max_path_change):
    early, late, low, high = map(Fraction, (early, late, min_path_change, max_path_change))
    if min(early, late) < 0 or low > high:
        raise ValueError("Require nonnegative observed resistances and ordered path bounds")
    # For nonnegative cell and path components, early path lies in [0,early],
    # late path in [0,late]. Their difference spans [-early,late].
    low, high = max(low, -early), min(high, late)
    if low > high:
        raise ValueError("Path bounds conflict with the observed series totals")
    return late - early - high, late - early - low


def demonstrate():
    return {
        "raw_terminal_change": "2",
        "path_change_plus_or_minus_three": list(map(str, cell_change_bounds(10, 12, -3, 3))),
        "path_change_plus_or_minus_one": list(map(str, cell_change_bounds(10, 12, -1, 1))),
    }


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
