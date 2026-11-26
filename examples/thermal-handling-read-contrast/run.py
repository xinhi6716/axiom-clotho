"""Exact synthetic example of matched thermal handling in a read study."""

from fractions import Fraction


def tray_mean_bounds(trays, missing_bounds):
    """Give each assigned tray equal weight and retain missing cells."""
    lo, hi = missing_bounds
    if not trays or any(not tray for tray in trays) or lo > hi:
        raise ValueError("nonempty assigned trays and ordered bounds required")
    return tuple(
        sum(
            sum(Fraction(bound if value is None else value) for value in tray)
            / len(tray)
            for tray in trays
        )
        / len(trays)
        for bound in (lo, hi)
    )


def difference(a, b):
    return (a[0] - b[1], a[1] - b[0])


def display(bounds):
    return f"[{bounds[0]}, {bounds[1]}]"


def main():
    missing_bounds = (-4, 4)
    a = ((1, 1), (1, 1, 1, 1))
    b = ((-3, -3), (-1, -1, -1, -1))
    c = ((-1, -1), (0, 0, 0, None))
    a, b, c = (tray_mean_bounds(arm, missing_bounds) for arm in (a, b, c))
    handling = difference(b, a)
    read_under_handling = difference(c, b)
    combined = difference(c, a)
    if (handling, read_under_handling, combined) != ((-3, -3), (1, 2), (-2, -1)):
        raise AssertionError("unexpected contrast bounds")
    print(f"Handling only (B-A): {display(handling)}")
    print(f"Added reads with handling (C-B): {display(read_under_handling)}")
    print(f"Combined read procedure (C-A): {display(combined)}")


if __name__ == "__main__":
    main()
