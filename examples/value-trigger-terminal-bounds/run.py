"""Exact finite-roster witness for a value-triggered sampling stop."""
import json
from fractions import Fraction


def fixture():
    # Planned samples at ticks 1 and 2. A value >= 5 at tick 1 stops sampling.
    # Roster rows are (first observed sample, recorded terminal sample).
    a = ((2, 1), (6, None))
    b = ((4, 4), (4, 4))
    assert all((first >= 5) == (end is None) for first, end in a + b)
    assert len(a) == len(b) == 2

    def contrast(unknown):
        return Fraction(a[0][1] + unknown, len(a)) - Fraction(
            sum(end for _, end in b), len(b)
        )

    # Integer/clip support is a property of this synthetic endpoint, not the
    # instrument. A separate illustrative one-tick slew certificate is tighter.
    broad = tuple(range(9))
    slew = tuple(x for x in broad if abs(x - a[1][0]) <= 1)

    def sharp(support):
        scores = [contrast(x) for x in support]
        return dict(
            feasible_terminal=list(support),
            a_minus_b=[str(min(scores)), str(max(scores))],
        )

    return {
        "attempted": [len(a), len(b)],
        "terminal_observed": [1, 2],
        "complete_case_difference": str(Fraction(a[0][1]) - 4),
        "invalid_carryforward_difference": str(contrast(a[1][0])),
        "worlds_same_observed_log": {str(x): str(contrast(x)) for x in (0, 8)},
        "range_only": sharp(broad),
        "range_and_slew": sharp(slew),
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
