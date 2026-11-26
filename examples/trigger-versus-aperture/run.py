"""Timing-interval examples with invented bounds in integer microseconds."""


def physical_elapsed(trigger_delta, program_end_lag, aperture_lag):
    """Bound physical aperture onset minus delivered program end."""
    p0, p1 = program_end_lag
    a0, a1 = aperture_lag
    if not (p0 <= p1 and a0 <= a1):
        raise ValueError("lag bounds must be ordered")
    return (trigger_delta + a0 - p1, trigger_delta + a1 - p0)


def verdict(interval, upper=1_020_000):
    lo, hi = interval
    if hi <= 0 or lo > upper:
        return "outside"
    if lo <= 0 or hi > upper:
        return "ambiguous"
    return "inside"


def main():
    examples = (
        ("trigger point passes but opening is late",
         (1_005_000, (0, 0), (25_000, 25_000)), (1_030_000, 1_030_000), "outside"),
        ("corrected center passes but bounds cross",
         (1_018_000, (2_000, 7_000), (2_000, 7_000)), (1_013_000, 1_023_000), "ambiguous"),
        ("upper endpoint is included",
         (1_015_000, (2_000, 2_000), (5_000, 7_000)), (1_018_000, 1_020_000), "inside"),
        ("zero could be included and is not admissible",
         (500, (1_000, 1_000), (500, 600)), (0, 100), "ambiguous"),
    )
    for name, inputs, expected, expected_status in examples:
        observed = physical_elapsed(*inputs)
        status = verdict(observed)
        if (observed, status) != (expected, expected_status):
            raise AssertionError(name)
        print(f"{name}: {observed} us [{status}]")


if __name__ == "__main__":
    main()
