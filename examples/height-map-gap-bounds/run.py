"""Exact bounds for an invented missing optical profile, not an imputer."""

from fractions import Fraction as F


def envelope(coordinates, readings, max_slope):
    """Sharp simultaneous pointwise bounds on a one-dimensional fixed grid."""
    if len(coordinates) != len(readings) or max_slope < 0:
        raise ValueError("misaligned inputs or invalid slope")
    known = [(x, y) for x, y in zip(coordinates, readings) if y is not None]
    if not known:
        raise ValueError("no observed profile anchors")
    low = [max(y - max_slope * abs(x - u) for u, y in known) for x in coordinates]
    high = [min(y + max_slope * abs(x - u) for u, y in known) for x in coordinates]
    if any(a > b for a, b in zip(low, high)):
        raise ValueError("observed anchors violate the declared slope")
    return low, high


def main():
    low, high = envelope([0, 1, 2], [2, None, 2], F(1))
    reference = F(21, 10)
    mean = [sum(path) / len(path) for path in (low, high)]
    peak = [max(path) for path in (low, high)]
    delta_mean = [v - reference for v in mean]
    delta_peak = [v - reference for v in peak]
    if (delta_mean, delta_peak) != ([F(-13, 30), F(7, 30)], [F(-1, 10), F(9, 10)]):
        raise AssertionError("missing-cell extrema")
    try:
        envelope([0, 1, 2], [0, None, 3], F(1))
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible observed anchors")
    print(f"Mean difference: [{delta_mean[0]}, {delta_mean[1]}]")
    print(f"Peak difference: [{delta_peak[0]}, {delta_peak[1]}]")


if __name__ == "__main__":
    main()
