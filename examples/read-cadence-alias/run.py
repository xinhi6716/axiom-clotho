"""Separate elapsed-time and prior-read effects in an invented retention design."""


def losses(age, prior_reads):
    """One time-only world and one prior-read-only world."""
    return age, prior_reads


def determinant(first, second):
    """Rank witness for a two-column, no-intercept contrast design."""
    a, r = first
    b, s = second
    return a * s - b * r


if __name__ == "__main__":
    dense = [(age, age) for age in range(1, 5)]
    for name, setting in (("held chip, same cadence", dense[-1]),
                          ("same age, no intermediate reads", (4, 0))):
        print(f"{name}: time-only/read-only loss={losses(*setting)}")
    print(f"same-cadence determinant={determinant(dense[0], dense[-1])}")
    print(f"matched-age contrast determinant={determinant(dense[-1], (4, 0))}")
