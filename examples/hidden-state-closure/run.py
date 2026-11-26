"""An exact counterexample to closing a device state by its present read."""
from fractions import Fraction
import json


def demonstrate():
    # Two hidden states share the observed reading, and both decay to
    # the same terminal class. Their single-tick survival probabilities
    # represent different fixed transition rates under the same protocol.
    survival = (Fraction(1, 2), Fraction(1, 4))
    mixture_one = sum(survival) / 2
    mixture_two = sum(p**2 for p in survival) / 2
    start_gap = max(survival) - min(survival)
    results = {
        "hidden_one_tick_survival": list(survival),
        "one_forecast_minimax_absolute_error": start_gap / 2,
        "half_each_survival_at_one_tick": mixture_one,
        "half_each_survival_at_two_ticks": mixture_two,
        "homogeneous_one_rate_survival_at_two_ticks": mixture_one**2,
        "two_tick_error_of_one_rate": mixture_two - mixture_one**2,
    }
    if results["one_forecast_minimax_absolute_error"] != Fraction(1, 8):
        raise AssertionError("Incorrect one-tick bound")
    if results["two_tick_error_of_one_rate"] != Fraction(1, 64):
        raise AssertionError("Incorrect two-tick mixture difference")
    return results


if __name__ == "__main__":
    print(json.dumps(demonstrate(), default=str, sort_keys=True))
