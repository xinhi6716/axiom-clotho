"""Exact invented fixed-bin forecast under a common conductance gain."""

from fractions import Fraction


def log2_power(value):
    """Integer base-2 log for positive integer powers of two in this fixture."""
    if type(value) is not int or value <= 0 or value & (value - 1):
        raise ValueError("This fixture uses positive integer powers of two.")
    return value.bit_length() - 1


def fixture(gain=1):
    """Apply an episode-wide gain at all reads; preserve the frozen cut."""
    log2_power(gain)
    early = (1, 2, 8)
    late = (2, 4, 4)
    errors = []
    for start, finish in zip(early, late):
        anchor = log2_power(start * gain)
        observed = log2_power(finish * gain)
        # This correction and the absolute conductance cut did not see test data.
        correction = 1 if start * gain < 4 else -1
        errors.append((abs(observed - anchor), abs(observed - anchor - correction)))
    pooled, binned = zip(*errors)
    return {
        "pooled_errors": pooled, "binned_errors": binned,
        "pooled_mae": Fraction(sum(pooled), len(pooled)),
        "binned_mae": Fraction(sum(binned), len(binned)),
    }


if __name__ == "__main__":
    for gain in (1, 8):
        result = fixture(gain)
        print(f"gain {gain}: pooled MAE={result['pooled_mae']}, "
              f"fixed-bin MAE={result['binned_mae']} (log2 units)")
