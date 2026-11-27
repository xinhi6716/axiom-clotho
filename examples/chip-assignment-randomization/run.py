"""Repeat reads do not increase the support of chip-level randomization."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
import json


def reference(values):
    n = len(values) // 2
    total = sum(values)
    return Counter(
        abs(2 * sum(values[i] for i in assignment) - total)
        for assignment in combinations(range(len(values)), n)
    )


def tail(reference_counts, observed):
    return Fraction(
        sum(n for statistic, n in reference_counts.items() if statistic >= observed),
        sum(reference_counts.values()),
    )


def fixture():
    chip_values = (0, 0, 0, 1, 1, 1)
    reads = tuple(v for v in chip_values for _ in range(2))
    chip_reference, row_reference = reference(chip_values), reference(reads)
    alpha = Fraction(1, 20)
    allocation_count = sum(chip_reference.values())
    # On clustered assignments, doubling every read preserves the original
    # mean difference, but doubles the unnormalized statistic.
    wrong_rejection = Fraction(
        sum(n for d, n in chip_reference.items() if tail(row_reference, 2 * d) <= alpha),
        allocation_count,
    )
    correct_rejection = Fraction(
        sum(n for d, n in chip_reference.items() if tail(chip_reference, d) <= alpha),
        allocation_count,
    )
    return {
        "observed_absolute_mean_difference": "1",
        "chip_allocations": allocation_count,
        "wrong_row_allocations": sum(row_reference.values()),
        "chip_p_value": str(tail(chip_reference, 3)),
        "wrong_row_p_value": str(tail(row_reference, 6)),
        "alpha": str(alpha),
        "chip_sharp_null_rejection": str(correct_rejection),
        "wrong_sharp_null_rejection_under_chip_design": str(wrong_rejection),
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
