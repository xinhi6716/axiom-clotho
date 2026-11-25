"""Exact synthetic example of selecting an interval using calibration outcomes."""

from dataclasses import dataclass
from fractions import Fraction

SCORE = "max-absolute-residual"


@dataclass(frozen=True)
class Band:
    center: int
    score: str
    half_width: Fraction

    @property
    def bounds(self):
        return self.center - self.half_width, self.center + self.half_width


def calibrate(center, calibration):
    """Calibrate only a fixed center at the maximum-residual rank."""
    if center not in (0, 1) or not calibration:
        raise ValueError("Expected a fixed center and calibration devices.")
    return Band(center, SCORE, max(abs(y - center) for y in calibration))


def width_selected(calibration):
    """Deliberate anti-pattern: choose using the final calibration outcomes."""
    return min((calibrate(0, calibration), calibrate(1, calibration)),
               key=lambda band: (band.half_width, band.center))


def covered(band, truth):
    lo, hi = band.bounds
    return lo <= truth <= hi


def exact_coverages(n):
    """For IID Uniform(0,1) devices and calibration by the maximum."""
    if type(n) is not int or n < 1:
        raise ValueError("Expected a positive device count.")
    tail_mean = Fraction(1, n + 1)
    # Integral_0^(1/2) (1-2t)^n dt for min(left gap, right gap).
    smaller_gap_mean = tail_mean / 2
    return 1 - tail_mean, 1 - (2 * tail_mean - smaller_gap_mean)


def fixture():
    return tuple(Fraction(k, 100) for k in range(31, 50))


if __name__ == "__main__":
    fixed, selected = exact_coverages(19)
    print(f"19 calibration devices: fixed={fixed}, width-selected={selected}")
    roster = fixture()
    narrow, frozen = width_selected(roster), calibrate(1, roster)
    truth = Fraction(3, 5)
    print(f"exact example: selected center={narrow.center}, half-width={narrow.half_width}, "
          f"covered={covered(narrow, truth)}")
    print(f"prespecified center={frozen.center}, half-width={frozen.half_width}, "
          f"covered={covered(frozen, truth)}")
