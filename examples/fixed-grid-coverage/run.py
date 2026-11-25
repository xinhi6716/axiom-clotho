"""Enumerate exchangeable held-device positions on one fixed toy read grid."""

from fractions import Fraction

GRID = (10, 20, 30, 40)
ALPHA = Fraction(1, 20)


def audit(scores):
    """Compare pointwise, outcome-selected and simultaneous score events."""
    if len(scores) != 20 or any(set(row) != set(GRID) for row in scores):
        raise ValueError("The example requires 20 complete devices on the fixed read grid.")
    pointwise = dict.fromkeys(GRID, 0)
    selected, whole = 0, 0
    # n=19: ceil((n+1)(1-alpha)) = 19 is exactly the calibration maximum.
    assert len(scores) * (1 - ALPHA) == len(scores) - 1
    for held, query in enumerate(scores):
        calibration = [row for i, row in enumerate(scores) if i != held]
        thresholds = {t: max(row[t] for row in calibration) for t in GRID}
        # Deliberately outcome-selected, not a deployable forecasting rule.
        chosen = max(GRID, key=query.__getitem__)
        for t in GRID:
            pointwise[t] += query[t] <= thresholds[t]
        selected += query[chosen] <= thresholds[chosen]
        whole += max(query.values()) <= max(max(row.values()) for row in calibration)
    return tuple(pointwise.values()), selected, whole


def fixture():
    return [{t: j + 1 if i == j else 0 for j, t in enumerate(GRID)} for i in range(20)]


def main():
    point, selected, whole = audit(fixture())
    print("four fixed times:", " ".join(f"{n}/20" for n in point))
    print(f"outcome-selected time with pointwise intervals: {selected}/20")
    print(f"simultaneous fixed grid: {whole}/20")


if __name__ == "__main__":
    main()
