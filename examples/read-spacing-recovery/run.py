"""Invented dynamics, not fitted or calibrated to a physical device."""
import json
from fractions import Fraction

SCHEDULES = {"none": (), "early": (1, 2), "late": (4, 5), "dense": (1, 2, 3, 4, 5)}


def summarize(intermediate, terminal=6):
    if tuple(sorted(set(intermediate))) != intermediate or any(not 0 < t < terminal for t in intermediate):
        raise ValueError("Require unique ordered intermediate ticks strictly before endpoint")
    state = Fraction(0)
    for tick in range(1, terminal + 1):
        state = state / 2 + (tick in intermediate)
    return {
        "commands_before_terminal": len(intermediate),
        "elapsed_only": str(Fraction(0)),
        "count_only": str(len(intermediate)),
        "spacing_sensitive": str(state),
    }


def demonstrate():
    return {name: summarize(times) for name, times in SCHEDULES.items()}


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
