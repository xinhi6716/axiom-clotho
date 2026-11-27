"""Exact synthetic energy for synchronized piecewise-constant channels."""
import json
from fractions import Fraction


def energy(voltage, current):
    if not voltage or len(voltage) != len(current):
        raise ValueError("Require a nonempty common bin grid")
    # One time unit per bin, equal across all cases.
    paired = sum(v * i for v, i in zip(voltage, current))
    marginal = Fraction(sum(voltage) * sum(current), len(voltage))
    return paired, marginal


def demonstrate():
    reference = energy((1, 1), (3, 4))[0]
    return {
        name: {
            "paired_energy": paired, "product_of_channel_means_times_duration": str(marginal),
            "contrast_to_fixed_reference": paired - reference,
        }
        for name, (paired, marginal) in {
            "aligned_high_with_high": energy((1, 3), (1, 3)),
            "aligned_high_with_low": energy((1, 3), (3, 1)),
        }.items()
    }


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
