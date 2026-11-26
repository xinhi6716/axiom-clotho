"""Invented reads: a plausible timer subtraction is not proof of elapsed time."""
from dataclasses import dataclass

LOWER, UPPER = 80_000, 120_000


@dataclass(frozen=True)
class Timestamp:
    seconds: int
    clock: str
    epoch: str
    error_seconds: int


def classify(origin, terminal):
    """Assumes a verified common timeline, or leaves elapsed time unscored."""
    if origin is None:
        return "unscored_missing_origin", None
    if (not origin.clock or not origin.epoch
            or (origin.clock, origin.epoch) != (terminal.clock, terminal.epoch)):
        return "unscored_unlinked_clock", None
    delta = terminal.seconds - origin.seconds
    error = origin.error_seconds + terminal.error_seconds
    bounds = delta - error, delta + error
    if bounds[0] >= LOWER and bounds[1] <= UPPER:
        return "eligible", bounds
    if bounds[1] < LOWER or bounds[0] > UPPER:
        return "outside_window", bounds
    return "unscored_boundary_uncertain", bounds


ORIGIN = Timestamp(400, "timer-A", "epoch-1", 1)
CASES = (
    ("linked", ORIGIN, Timestamp(100_400, "timer-A", "epoch-1", 1)),
    ("reset", ORIGIN, Timestamp(100_400, "timer-A", "epoch-2", 1)),
    ("clock-change", ORIGIN, Timestamp(100_400, "timer-B", "epoch-1", 1)),
    ("missing-origin", None, Timestamp(100_400, "timer-A", "epoch-1", 1)),
    ("uncertain-edge", ORIGIN, Timestamp(80_400, "timer-A", "epoch-1", 3)),
    ("outside", ORIGIN, Timestamp(80_390, "timer-A", "epoch-1", 3)),
)


if __name__ == "__main__":
    for name, origin, terminal in CASES:
        # Demonstrate, rather than endorse, the naive fallback of treating a
        # missing origin as the timer's zero.
        difference = terminal.seconds - (0 if origin is None else origin.seconds)
        verdict, bounds = classify(origin, terminal)
        print(name, f"naive={LOWER <= difference <= UPPER}", verdict, bounds)
