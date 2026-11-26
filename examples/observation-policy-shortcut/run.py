"""Invented pre-origin schedule features under fixed, inert device dynamics."""
import json
from fractions import Fraction
from itertools import product


def brier(rows, predict):
    return Fraction(sum((predict(read) - target)**2 for read, target in rows), len(rows))


def demonstrate():
    # An unrecorded pre-origin flag predicts the eventual endpoint in this toy.
    # The additional measurement itself has no effect and its value is unused.
    dev = [(flag, flag) for flag in (0, 1)]
    frozen_schedule_head = {read: Fraction(y) for read, y in dev}
    frozen_pooled = Fraction(sum(y for _, y in dev), len(dev))
    return {
        label: {
            "schedule_head_brier": str(brier(rows, frozen_schedule_head.__getitem__)),
            "pooled_brier": str(brier(rows, lambda _: frozen_pooled)),
        }
        for label, rows in {
            "untouched_same_policy": [(flag, flag) for flag in (0, 1)],
            "untouched_independent_policy": list(product((0, 1), repeat=2)),
        }.items()
    }


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
