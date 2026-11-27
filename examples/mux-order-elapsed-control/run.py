"""Synthetic reciprocal queue diagnostic with hard elapsed error bounds."""
from collections import Counter
import json


def elapsed(program, read, program_error, read_error):
    if any(lo > hi for lo, hi in (program_error, read_error)):
        raise ValueError("reversed bounds")
    # The identical offset on both timestamps cancels before bounding.
    return (read - program + read_error[0] - program_error[1],
            read - program + read_error[1] - program_error[0])


def state(limits):
    lo, hi = limits
    if hi <= 0 or lo > 1020:
        return "outside"
    return "inside" if 0 < lo <= hi <= 1020 else "unresolved"


def capture(order, common_offset):
    return [
        (cell, pos, state(elapsed(common_offset, common_offset + center,
                                 (-3, 3), (-2, 2))))
        for pos, (cell, center) in enumerate(zip(order, (1000, 1018, 1030)), 1)
    ]


def transitions(forward, reverse, by):
    first = {row[by]: row[2] for row in forward}
    second = {row[by]: row[2] for row in reverse}
    assert first.keys() == second.keys()
    return dict(sorted(Counter(f"{first[key]}->{second[key]}" for key in first).items()))


def fixture():
    forward = capture(("A", "B", "C"), 19000)
    reverse = capture(("C", "B", "A"), -8000)
    return {"forward": forward, "reverse": reverse,
            "by_cell": transitions(forward, reverse, 0),
            "by_position": transitions(forward, reverse, 1)}


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
