"""Exact paired Brier bounds for a fixed binary horizon and invented rows."""
import json
from fractions import Fraction


def bounds(rows):
    """Each row is (A probability, B probability, known horizon label or None)."""
    ranges = []
    for a, b, label in rows:
        a, b = Fraction(a), Fraction(b)
        if not (0 <= a <= 1 and 0 <= b <= 1) or label not in (0, 1, None):
            raise ValueError("Invalid forecast or horizon label")
        choices = (0, 1) if label is None else (label,)
        gaps = [(b - y)**2 - (a - y)**2 for y in choices]
        ranges.append((min(gaps), max(gaps)))
    if not ranges:
        raise ValueError("Require the full declared roster")
    return tuple(sum(col) / len(rows) for col in zip(*ranges))


def demonstrate():
    rows = ((Fraction(1, 2), Fraction(3, 4), 1),
            (Fraction(1, 2), Fraction(3, 4), None))
    return {
        "resolved_only_gap": str(bounds(rows[:1])[0]),
        "full_roster_gap_range": list(map(str, bounds(rows))),
    }


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
