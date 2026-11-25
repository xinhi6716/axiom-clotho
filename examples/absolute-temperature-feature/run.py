"""Compare an inverse absolute-temperature feature across Celsius and kelvin."""

from fractions import Fraction as F

CELSIUS_OFFSET = F(27315, 100)


def inverse_temperature(value, unit, quantity):
    """For this toy, convert an absolute temperature before reciprocating."""
    if quantity != "absolute_temperature":
        raise ValueError("An absolute temperature, not an increment, is required.")
    if unit not in ("degC", "K"):
        raise ValueError("Expected a declared temperature unit.")
    kelvin = value + CELSIUS_OFFSET if unit == "degC" else F(value)
    if kelvin <= 0:
        raise ValueError("The reciprocal requires temperature above absolute zero.")
    return 1 / kelvin


if __name__ == "__main__":
    def contrast(lo, hi, unit):
        return (inverse_temperature(lo, unit, "absolute_temperature")
                - inverse_temperature(hi, unit, "absolute_temperature"))

    print("converted Celsius:", contrast(-25, 25, "degC"))
    print("declared kelvin:", contrast(F("248.15"), F("298.15"), "K"))
    print("raw Celsius anti-pattern:", F(1, -25) - F(1, 25))
