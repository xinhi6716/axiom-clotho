"""Exhaustive synthetic example of selection using a previously scored roster."""

from fractions import Fraction
from itertools import product

LABELS = tuple(product((0, 1), repeat=4))
CONSTANTS = (LABELS[0], LABELS[-1])


def loss(predictions, labels):
    """Brier loss for this toy's hard 0/1 probabilities."""
    return Fraction(sum((p - y)**2 for p, y in zip(predictions, labels)), 4)


def select(family, previous_labels):
    """Tie-break using the candidate vector, not the fresh roster."""
    return min(family, key=lambda p: (loss(p, previous_labels), p))


def expected_losses(family):
    """Average reused and genuinely fresh scores over all 256 label worlds."""
    old_total = fresh_total = Fraction(0)
    for previous in LABELS:
        chosen = select(family, previous)
        for fresh in LABELS:
            old_total += loss(chosen, previous)
            fresh_total += loss(chosen, fresh)
    return old_total / 256, fresh_total / 256


if __name__ == "__main__":
    for name, family in (
        ("fixed zero", CONSTANTS[:1]),
        ("two constants", CONSTANTS),
        ("all 16 vectors", LABELS),
    ):
        reused, fresh = expected_losses(family)
        print(f"{name}: reused={reused}, fresh={fresh}")
