"""Keep penetration observations and adhesion adjudications independent."""

from fractions import Fraction as F

ROWS = (
    ("a", 0, "adhered"),
    ("b", 4, "failed"),
    ("c", 2, "unknown"),
    ("d", None, "adhered"),
)


def report(rows):
    positive = zero = unavailable = confirmed = failures = unresolved = 0
    seen = set()
    for key, depth, status in rows:
        if key in seen:
            raise ValueError("Duplicate attempt.")
        seen.add(key)
        if depth is None:
            unavailable += 1
        elif type(depth) is not int or depth < 0:
            raise ValueError("Expected nonnegative integer depth or unavailable.")
        elif depth == 0:
            zero += 1
        else:
            positive += 1
        if status == "adhered":
            confirmed += 1
        elif status == "failed":
            failures += 1
        elif status == "unknown":
            unresolved += 1
        else:
            raise ValueError("Expected an explicit adhesion status.")
    return (positive, zero, unavailable), (confirmed, failures, unresolved)


if __name__ == "__main__":
    depth, adhesion = report(ROWS)
    known, failed, unknown = adhesion
    print(f"depth positive/zero/unavailable: {depth}")
    print(f"adhesion confirmed/failed/unknown: {adhesion}")
    print(f"confirmed adhesion per attempted track: {F(known, known + failed + unknown)}")
