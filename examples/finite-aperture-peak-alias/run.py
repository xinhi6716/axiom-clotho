"""An exact synthetic example for ideal rectangular read apertures."""
import json
from fractions import Fraction


def summary(trace):
    return {
        "full_aperture_mean": str(Fraction(sum(trace), 4)),
        "half_aperture_means": [str(Fraction(sum(trace[i:i+2]), 2)) for i in (0, 2)],
        "peak": max(trace),
        "fraction_above_two": str(Fraction(sum(v > 2 for v in trace), 4)),
    }


def case():
    return {name: summary(trace) for name, trace in {
        "constant": (1, 1, 1, 1),
        "sharp": (0, 4, 0, 0),
        "broad": (2, 2, 0, 0),
    }.items()}


if __name__ == "__main__":
    print(json.dumps(case(), indent=2))
