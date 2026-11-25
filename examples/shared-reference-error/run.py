"""Exact audit of a synthetic endpoint with shared measurement error."""

from fractions import Fraction
from itertools import product


def scores():
    shared, separate, latent, duplicate = [], [], [], []
    for signal, common_error, separate_error in product((-1, 1), repeat=3):
        physical = 4 + signal
        # Both rules may use the same origin tuple (signal, common_error).
        # The first keeps the acquisition-path offset; the second ignores it.
        keep_offset, ignore_offset = physical + common_error, physical
        reference = physical + common_error
        repeated_reference = physical + common_error
        alternative_reference = physical + separate_error
        for values, target in ((shared, reference), (separate, alternative_reference),
                               (latent, physical)):
            values.append(((keep_offset - target)**2, (ignore_offset - target)**2))
        duplicate.append((reference - repeated_reference)**2)

    def pair_mean(rows):
        return tuple(Fraction(sum(column), len(rows)) for column in zip(*rows))

    return {
        "shared reference": pair_mean(shared),
        "separate reference": pair_mean(separate),
        "physical target": pair_mean(latent),
        "shared-reference duplicate disagreement": (Fraction(sum(duplicate), len(duplicate)),),
    }


if __name__ == "__main__":
    for scope, values in scores().items():
        print(f"{scope}: {', '.join(str(value) for value in values)}")
