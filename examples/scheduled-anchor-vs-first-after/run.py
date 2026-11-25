"""Exact nominal acquisition anchors are not the first read after a clock cut."""

ANCHORS = (0, 1, 100_000)
ROWS = (
    (0, 0, 8),
    (1, 1, 7),
    (90_000, 100_001, 10),  # a delayed older nominal read
    (100_000, 100_011, 3),
)


def scheduled(rows, anchors=ANCHORS):
    """Return one row per exact nominal anchor, or identify an incomplete roster."""
    selected = []
    for anchor in anchors:
        matches = [row for row in rows if row[0] == anchor]
        if len(matches) > 1:
            raise ValueError(f"Duplicate nominal anchor: {anchor}")
        if not matches:
            return None
        selected.append(matches[0])
    if any(a[1] >= b[1] for a, b in zip(selected, selected[1:])):
        raise ValueError("Actual anchor timestamps are not strictly increasing.")
    return tuple(selected)


def first_after(rows, cutoff):
    """Separate clock-based statistic; these toy rows have no actual-time ties."""
    return min((row for row in rows if row[1] > cutoff),
               key=lambda row: row[1], default=None)


if __name__ == "__main__":
    exact = scheduled(ROWS)
    clock = first_after(ROWS, ANCHORS[-1])
    print(f"exact nominal terminal: actual={exact[-1][1]} value={exact[-1][2]}")
    print(f"first after clock cutoff: nominal={clock[0]} actual={clock[1]} value={clock[2]}")
    print(f"missing nominal terminal: exact={scheduled(ROWS[:-1])}")
