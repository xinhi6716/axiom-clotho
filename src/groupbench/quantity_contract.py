"""Opt-in, no-fit electrical quantity helpers; NOT an ML admission gate.

Declarations are caller assertions, not verified physical facts. A per-column
unit cannot detect secretly mixed-unit rows. No raw-value provenance is retained.
"""
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
import re
import unicodedata

REGISTRY_VERSION = "electrical-decimal-v2"
MAX_VALUES = 20_000
# dimension, canonical unit, exact base-ten exponent
_UNITS = {
    "A": ("electric_current", "A", 0),
    "mA": ("electric_current", "A", -3),
    "uA": ("electric_current", "A", -6),
    "nA": ("electric_current", "A", -9),
    "pA": ("electric_current", "A", -12),
    "V": ("electric_potential", "V", 0),
    "mV": ("electric_potential", "V", -3),
    "uV": ("electric_potential", "V", -6),
    "ohm": ("resistance", "ohm", 0),
    "kohm": ("resistance", "ohm", 3),
    "Mohm": ("resistance", "ohm", 6),
    "S": ("electric_conductance", "S", 0),
    "mS": ("electric_conductance", "S", -3),
    "uS": ("electric_conductance", "S", -6),
    "nS": ("electric_conductance", "S", -9),
    "pS": ("electric_conductance", "S", -12),
    "s": ("elapsed_time", "s", 0),
    "ms": ("elapsed_time", "s", -3),
    "us": ("elapsed_time", "s", -6),
    "ns": ("elapsed_time", "s", -9),
    "ks": ("elapsed_time", "s", 3),
    "1": ("dimensionless", "1", 0),
}
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE]([+-]?[0-9]+))?\Z")


@dataclass(frozen=True)
class QuantityDeclaration:
    column: str
    quantity_id: str
    statistic_id: str
    condition_id: str
    dimension: str
    unit: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (type(value) is not str or not value or len(value) > 256
                    or value != value.strip() or any(unicodedata.category(c) == "Cc" for c in value)):
                raise ValueError("Quantity declarations require bounded unpadded text without control characters.")
        if self.unit not in _UNITS or _UNITS[self.unit][0] != self.dimension:
            raise ValueError("Unsupported unit or inconsistent declared dimension.")


def _declaration(value):
    if type(value) is not QuantityDeclaration:
        raise ValueError("An explicit QuantityDeclaration is required.")
    # Recheck even if a caller circumvented frozen dataclass assignment.
    value.__post_init__()
    return value


def compare_declarations(expected, supplied, *, mode="validate"):
    """Reject different meanings; differing supported units require normalize mode.

    Return a canonical declaration in normalize mode, otherwise the expectation.
    No synonym inference or physical comparability certification is performed.
    """
    expected, supplied = _declaration(expected), _declaration(supplied)
    if type(mode) is not str or mode not in ("validate", "normalize"):
        raise ValueError("mode must be validate or normalize.")
    fields = ("column", "quantity_id", "statistic_id", "condition_id", "dimension")
    if any(getattr(expected, f) != getattr(supplied, f) for f in fields):
        raise ValueError("Quantity, statistic, condition, column or dimension declarations differ.")
    if mode == "validate":
        if expected.unit != supplied.unit:
            raise ValueError("Different units require explicit normalize mode.")
        return expected
    return QuantityDeclaration(*(getattr(expected, f) for f in fields), _UNITS[expected.unit][1])


def _values(values):
    if type(values) not in (list, tuple) or not 1 <= len(values) <= MAX_VALUES:
        raise ValueError("Values must be a nonempty bounded list or tuple.")


def normalize_decimal_tokens(tokens, declaration):
    """Explicitly convert bounded decimal text to exact canonical-unit Decimals.

    Accept only strings, at most 256 characters and 64 coefficient digits, with
    explicit and effective Decimal exponent magnitudes at most 200. Missing
    values are unsupported. Never parse a previously rounded float as raw text.
    """
    declaration = _declaration(declaration)
    _values(tokens)
    shift = _UNITS[declaration.unit][2]
    output = []
    for token in tokens:
        if type(token) is not str or len(token) > 256:
            raise ValueError("Expected bounded decimal text, not typed numeric values.")
        match = _NUMBER.fullmatch(token)
        if match is None or (match.group(1) is not None and abs(int(match.group(1))) > 200):
            raise ValueError("Invalid decimal token or exponent outside bounds.")
        coefficient = re.split("[eE]", token)[0]
        if sum(c.isdigit() for c in coefficient) > 64:
            raise ValueError("Decimal coefficient exceeds 64 digits.")
        value = Decimal(token)
        if abs(value.as_tuple().exponent) > 200:
            raise ValueError("Effective decimal exponent outside bounds.")
        # Tuple construction shifts the exponent without context rounding.
        sign, digits, exponent = value.as_tuple()
        output.append(Decimal((sign, digits, exponent + shift)))
    return tuple(output)


def decimal_to_float64(values):
    """Explicit approximate Python binary64 boundary, with conservative guards.

    Integral values must be exactly representable. Fractional values can round;
    distinct supplied decimals may not collapse within this batch. No claim is
    made about cross-batch collisions or decimal precision already lost upstream.
    """
    _values(values)
    output, seen = [], {}
    for value in values:
        if type(value) is not Decimal or not value.is_finite():
            raise ValueError("Expected finite Decimal values.")
        result = float(value)
        if not math.isfinite(result) or (value != 0 and result == 0):
            raise ValueError("Float64 overflow or nonzero underflow.")
        if value == value.to_integral_value() and Decimal.from_float(result) != value:
            raise ValueError("Integral values must be exactly representable in float64.")
        if result in seen and seen[result] != value:
            raise ValueError("Distinct decimals collapse at the float64 boundary.")
        seen[result] = value
        output.append(result)
    return tuple(output)


def normalize_comparable_pair_to_float64(expected, expected_tokens, supplied, supplied_tokens):
    """Normalize a declared pair and guard the shared binary64 boundary.

    Both batches must be nonempty and together contain at most MAX_VALUES
    entries. Validate matching measurement meaning before parsing either
    batch. Reject collisions across as well as within batches. Return the
    canonical declaration and two immutable tuples in input order.
    """
    canonical = compare_declarations(expected, supplied, mode="normalize")
    _values(expected_tokens)
    _values(supplied_tokens)
    n = len(expected_tokens)
    if n + len(supplied_tokens) > MAX_VALUES:
        raise ValueError("Combined values exceed the float64 comparison limit.")
    a = normalize_decimal_tokens(expected_tokens, expected)
    b = normalize_decimal_tokens(supplied_tokens, supplied)
    converted = decimal_to_float64(a + b)
    return canonical, converted[:n], converted[n:]


def _number(value, role):
    if type(value) is int and value.bit_length() <= 512:
        return Fraction(value)
    if type(value) is float and math.isfinite(value):
        return Fraction(value)
    if type(value) is Decimal and value.is_finite():
        _, digits, exponent = value.as_tuple()
        if len(digits) <= 64 and -200 <= exponent <= 200:
            return Fraction(value)
    raise ValueError(f"{role} needs a bounded finite real number; booleans are not measurements.")


def _three(values, role):
    if type(values) not in (list, tuple) or len(values) != 3:
        raise ValueError(f"{role} must contain low, interior and high values, in that order.")
    return tuple(_number(v, role) for v in values)


@dataclass(frozen=True)
class AffineReadoutComparison:
    """Descriptive checks in readout units, not a calibration certificate."""

    reference_interior_error: Fraction
    comparison_interior_error: Fraction
    low_path_difference: Fraction
    high_path_difference: Fraction
    max_affine_path_difference: Fraction
    interior_pass: bool
    shared_affine_pass: bool


def compare_affine_readouts(standards, reference, comparison, *,
                            max_interior_error, max_path_difference):
    """Fit two endpoint lines, each with a reserved interior check.

    Each arm must measure the same three known, ordered standards in the same
    units. Caller establishes physical path matching, reference accuracy,
    repeats, drift and tolerances. Thresholds are nonnegative, inclusive,
    fixed before comparison, and expressed in response units. A bounded
    model-vs-model difference does not exclude unsampled nonlinearity.
    """
    x0, x1, x2 = _three(standards, "standards")
    r0, r1, r2 = _three(reference, "reference")
    c0, c1, c2 = _three(comparison, "comparison")
    e = _number(max_interior_error, "max_interior_error")
    d = _number(max_path_difference, "max_path_difference")
    if not x0 < x1 < x2 or e < 0 or d < 0:
        raise ValueError("Standards must strictly increase and tolerances must be nonnegative.")
    if not r0 < r2 or not c0 < c2:
        raise ValueError("Each path needs a positive endpoint response slope.")
    t = (x1 - x0) / (x2 - x0)
    er = r1 - (r0 + t * (r2 - r0))
    ec = c1 - (c0 + t * (c2 - c0))
    low, high = c0 - r0, c2 - r2
    maximum = max(abs(low), abs(high))
    interior_pass = abs(er) <= e and abs(ec) <= e
    return AffineReadoutComparison(er, ec, low, high, maximum, interior_pass,
                                  interior_pass and maximum <= d)
