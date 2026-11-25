"""Identical accelerated survival curves can imply different use-level risks."""

from fractions import Fraction as F


def hazards(stress):
    """Return one exponential hazard and a sum of two independent hazards."""
    single = F(25, 3) * F(6, 5)**stress
    competing = 8 + F(2)**stress
    return single, competing


if __name__ == "__main__":
    for label, stress in (("accelerated 1", 1), ("accelerated 2", 2), ("use", 0)):
        single, competing = hazards(stress)
        print(f"{label}: single={single}, competing={competing}")
