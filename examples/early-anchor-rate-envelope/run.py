"""Tight pre-first-sample bounds with an independently certified rate."""
from fractions import Fraction as Q
import json


def anchor_envelope(anchor, samples, min_drop, max_drop):
    if not 0 <= min_drop <= max_drop or not samples or anchor >= samples[0][0]:
        raise ValueError("invalid pre-first-sample envelope")
    for (left, a), (right, b) in zip(samples, samples[1:]):
        if not right > left or not min_drop <= (a - b) / (right - left) <= max_drop:
            raise ValueError("logged samples contradict certificate")
    # Enforce every endpoint cone. A straight segment to the first sample
    # and between each observed pair attain the extremes under this model.
    return (
        max(y + min_drop * (t - anchor) for t, y in samples),
        min(y + max_drop * (t - anchor) for t, y in samples),
    )


def fixture():
    sparse = ((10, Q(10)), (20, Q(8)))
    target = 1
    broad = anchor_envelope(target, sparse, Q(0), Q(1))
    certified = anchor_envelope(target, sparse, Q(1, 5), Q(1, 4))
    extrapolation = sparse[0][1] + Q(1, 5) * (sparse[0][0] - target)
    contradiction = False
    try:
        anchor_envelope(target, sparse, Q(1, 3), Q(1, 2))
    except ValueError:
        contradiction = True
    return {
        "native_early_read_available": False,
        "anchor": target,
        "linear_back_extrapolation": str(extrapolation),
        "monotone_plus_broad_rate": list(map(str, broad)),
        "independent_tight_rate": list(map(str, certified)),
        "inconsistent_rate_rejected": contradiction,
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
