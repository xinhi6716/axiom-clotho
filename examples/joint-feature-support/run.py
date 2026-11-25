"""Exact joint-support witness on one invented two-variable design."""

from fractions import Fraction as F

TRAIN = ((0, 0), (1, 0), (0, 1))
OUTSIDE = (F(3, 4), F(3, 4))
INSIDE = (F(1, 4), F(1, 4))


def coordinate_checks(point):
    """Classify this triangle only; this is not a generic hull algorithm."""
    x, y = point
    return 0 <= x <= 1 and 0 <= y <= 1, min(1 - x - y, x, y) >= 0


def worlds(point):
    """Two possible response surfaces that agree at every training vertex."""
    x, y = point
    return 0, 4 * max(0, x + y - 1) + 16 * x * y


if __name__ == "__main__":
    for name, point in (("outside", OUTSIDE), ("inside", INSIDE)):
        ranges, hull = coordinate_checks(point)
        left, right = worlds(point)
        print(f"{name}: ranges={ranges}, hull={hull}, world gap={right - left}")
