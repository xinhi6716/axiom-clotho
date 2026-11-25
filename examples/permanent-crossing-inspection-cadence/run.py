"""Exact finite-roster contrast under unequal inspection cadence."""

from fractions import Fraction

HORIZON = 10
# (last qualified stable read, first qualified failed read; None if stable
# through HORIZON). Reads bracket a permanent, not recoverable, transition.
ARM_A = ((0, 8), (10, None))
ARM_B = ((4, 6), (10, None))


def restricted_rows(roster, horizon):
    """Return supports of min(T, horizon); nondegenerate supports are (a,b]."""
    if type(horizon) is not int or horizon < 1 or not roster:
        raise ValueError("Require a positive integer horizon and complete roster")
    result = []
    for stable, failed in roster:
        if type(stable) is not int or not 0 <= stable <= horizon:
            raise ValueError("Require a qualified last stable tick on the horizon")
        if failed is None and stable == horizon:
            result.append((stable, stable))
        elif type(failed) is int and stable < failed <= horizon:
            result.append((stable, failed))
        else:
            raise ValueError("Require a failure bracket or stability through the horizon")
    return result


def summarize(arm_a, arm_b, horizon=HORIZON):
    """Describe A minus B on this finite roster, not a sampling interval."""
    a, b = restricted_rows(arm_a, horizon), restricted_rows(arm_b, horizon)
    al, au = (Fraction(sum(row[i] for row in a), len(a)) for i in (0, 1))
    bl, bu = (Fraction(sum(row[i] for row in b), len(b)) for i in (0, 1))
    return {
        "first_detection_difference": au - bu,
        "midpoint_difference": ((al + au) - (bl + bu)) / 2,
        "lower": al - bu,
        "upper": au - bl,
        "lower_included": all(lo == hi for lo, hi in a),
        "upper_included": all(lo == hi for lo, hi in b),
    }


if __name__ == "__main__":
    result = summarize(ARM_A, ARM_B)
    print("A minus B: first-detection =", result["first_detection_difference"],
          "; bracket midpoint =", result["midpoint_difference"])
    left = "[" if result["lower_included"] else "("
    right = "]" if result["upper_included"] else ")"
    print(f"Latent restricted time contrast: {left}{result['lower']}, {result['upper']}{right}")
