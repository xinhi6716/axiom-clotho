"""Invented paired deposition measurements under fixed or balanced order."""
from fractions import Fraction
import json


def difference(order, *, level_effect, period_effect, carry_from_b):
    values = {}
    for position, recipe in enumerate(order):
        direct = level_effect if recipe == "B" else 0
        carry = carry_from_b if position and order[position - 1] == "B" else 0
        values[recipe] = direct + position * period_effect + carry
    return values["B"] - values["A"]


def demonstrate():
    assumptions = {"level_effect": 2, "period_effect": 4}
    out = {}
    for key, carry in (("no_carryover", 0), ("B_changes_next_measurement", 6)):
        ab, ba = (difference(order, **assumptions, carry_from_b=carry)
                  for order in ("AB", "BA"))
        out[key] = {"always_AB": ab, "always_BA": ba,
                    "balance_AB_BA": str(Fraction(ab + ba, 2))}
    assert out["no_carryover"]["balance_AB_BA"] == "2"
    assert out["B_changes_next_measurement"]["balance_AB_BA"] == "-1"
    return out


if __name__ == "__main__":
    print(json.dumps(demonstrate(), sort_keys=True))
