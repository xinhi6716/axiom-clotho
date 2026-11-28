"""Exact positive control for commanded versus measured terminal voltage."""

from fractions import Fraction as F
import json


def rational(value):
    if type(value) not in (int, F):
        raise ValueError("Use exact rational synthetic inputs.")
    return F(value)


def ideal_sample(conductance, command_voltage, current_cap):
    """Static memoryless fixture; not an SMU or switching simulator."""
    g, command, cap = map(rational, (conductance, command_voltage, current_cap))
    if g < 0 or cap <= 0:
        raise ValueError("Expected nonnegative conductance and positive current limit.")
    free = g * command
    if abs(free) <= cap:
        return command, free
    limited = cap if free > 0 else -cap
    return limited / g, limited


def infer_conductance(terminal_voltage, current):
    """Use a contemporaneous, authenticated device-terminal voltage only."""
    v, i = map(rational, (terminal_voltage, current))
    if not v or v * i < 0:
        raise ValueError("Need a nonzero terminal voltage and passive signed current.")
    return i / v


def witness():
    drive, cap, held = F(1), F(1), F(1, 8)
    truth, naive, terminal_corrected, actual_v = [], [], [], []
    for g in (2, 4):
        v, i = ideal_sample(g, drive, cap)
        actual_v.append(v)
        truth.append(ideal_sample(g, held, cap)[1])
        naive.append(infer_conductance(drive, i) * held)
        terminal_corrected.append(infer_conductance(v, i) * held)
    n = len(truth)
    mae = lambda pred: sum(abs(a - b) for a, b in zip(pred, truth)) / n
    return {
        "shared_command_current_record": [str(drive), str(cap)],
        "actual_device_voltages": list(map(str, actual_v)),
        "held_sublimit_currents": list(map(str, truth)),
        "command_as_terminal_mae": str(mae(naive)),
        "any_shared_prediction_mae_floor": str(abs(truth[0] - truth[1]) / n),
        "synchronized_terminal_voltage_mae": str(mae(terminal_corrected)),
    }


if __name__ == "__main__":
    print(json.dumps(witness(), sort_keys=True))
