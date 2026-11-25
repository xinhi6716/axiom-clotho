"""Exact aggregation-order witness on two invented registered profiles."""

from fractions import Fraction as F

A = ((0, 4, 0, 0), (0, 0, 4, 0))
B = ((0, 3, 1, 0), (0, 3, 1, 0))


def summaries(profiles):
    n = len(profiles)
    average = tuple(F(sum(level), n) for level in zip(*profiles))
    mean_peak = F(sum(max(profile) for profile in profiles), n)
    unit_grid_area = sum((l + r) / 2 for l, r in zip(average, average[1:]))
    return max(average), mean_peak, unit_grid_area


if __name__ == "__main__":
    for name, profiles in (("A", A), ("B", B)):
        peak, mean, area = summaries(profiles)
        print(f"{name}: peak-of-mean={peak}, mean-of-peaks={mean}, area={area}")
