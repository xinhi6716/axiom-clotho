"""Exact bounded state relaxation under piecewise-constant drive (synthetic)."""

from fractions import Fraction
import json


def unit(value):
    if type(value) not in (int, Fraction) or not 0 <= value <= 1:
        raise ValueError("Expected an exact rational in [0,1].")
    return Fraction(value)


def step(state, equilibrium, ticks, decay_per_tick=Fraction(1, 2)):
    """The closed-form step, with 'equilibrium' supplied by the drive map."""
    x, u = unit(state), unit(equilibrium)
    q = unit(decay_per_tick)
    if not 0 < q < 1:
        raise ValueError("Expected a strict contraction.")
    if type(ticks) is not int or not 0 <= ticks <= 256:
        raise ValueError("Expected 0 through 256 elapsed integer ticks.")
    # Convex combination of x and u; exact rational positive invariance and
    # fixed-drive subdivision, without Euler step-size restrictions.
    a = q ** ticks
    return a * x + (1 - a) * u


def current(state, voltage):
    """Dimensionless conduction-only, no capacitance or compliance model."""
    if type(voltage) not in (int, Fraction):
        raise ValueError("Expected an exact rational voltage.")
    return Fraction(voltage) * (1 + unit(state))


def witness():
    outputs = []
    # Equal duration, duty count, and final drive; only pulse order differs.
    histories = ((1, 0, 1), (0, 1, 1))
    for history in histories:
        x = Fraction(0)
        for u in history:
            x = step(x, u, 1)
        outputs.append(current(x, 1))
    baseline_mae_floor = abs(outputs[0] - outputs[1]) / 2
    return {
        "terminal_currents": list(map(str, outputs)),
        "order_insensitive_baseline_mae_floor": str(baseline_mae_floor),
        "same_law_constructed_world_mae": "0",
    }


if __name__ == "__main__":
    print(json.dumps(witness(), sort_keys=True))
