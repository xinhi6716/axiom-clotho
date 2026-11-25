"""Invented decimal arithmetic only; no model construction or fitting."""
from dataclasses import replace
from decimal import Decimal, localcontext
import unittest
from groupbench.quantity_contract import (QuantityDeclaration as Q,
    compare_declarations, normalize_decimal_tokens, decimal_to_float64,
    normalize_comparable_pair_to_float64)


def declaration(unit="A", dimension="electric_current"):
    return Q("x", "terminal_current", "median", "declared_protocol_v1", dimension, unit)


class QuantityContractTests(unittest.TestCase):
    def test_exact_normalization_train_query(self):
        train, query = declaration(), declaration("nA")
        self.assertEqual(compare_declarations(train, query, mode="normalize"), train)
        a = normalize_decimal_tokens(["1e-9", "-0", "-2e-9"], train)
        b = normalize_decimal_tokens(["1", "-0", "-2"], query)
        self.assertEqual(a, b)
        self.assertEqual(decimal_to_float64(a), decimal_to_float64(b))
        self.assertEqual(a[1].as_tuple().sign, 1)
        with self.assertRaises(ValueError): compare_declarations(train, query)

    def test_registry_and_exact_scales(self):
        groups = [("electric_current", {"A": 0,"mA": -3,"uA": -6,"nA": -9,"pA": -12}),
                  ("electric_potential", {"V": 0,"mV": -3,"uV": -6}),
                  ("resistance", {"ohm": 0,"kohm": 3,"Mohm": 6}),
                  ("dimensionless", {"1": 0})]
        for dimension, units in groups:
            for unit, power in units.items():
                q = declaration(unit, dimension)
                self.assertEqual(normalize_decimal_tokens(["1.25"], q), (Decimal("1.25") * Decimal(10) ** power,))
                self.assertEqual(compare_declarations(q, q), q)

    def test_read_clock_and_conductance_units_are_exact_and_distinct(self):
        for dimension, unit, scaled, raw, canonical in (
            ("electric_conductance", "S", "uS", "12.5", "0.0000125"),
            ("electric_conductance", "S", "pS", "12.5", "0.0000000000125"),
            ("elapsed_time", "s", "ms", "1020", "1.02"),
            ("elapsed_time", "s", "ns", "20", "0.00000002"),
            ("elapsed_time", "s", "ks", "100", "100000"),
        ):
            with self.subTest(unit=scaled), localcontext() as ctx:
                ctx.prec = 1
                left = Q("x", dimension, "instantaneous", "same_protocol", dimension, unit)
                right = replace(left, unit=scaled)
                self.assertEqual(normalize_decimal_tokens([raw], right), (Decimal(canonical),))
                self.assertEqual(normalize_comparable_pair_to_float64(
                    left, [canonical], right, [raw]), (left, (float(canonical),), (float(canonical),)))
                with self.assertRaises(ValueError):
                    compare_declarations(left, replace(right, condition_id="different_clock_or_bias"),
                                         mode="normalize")
        with self.assertRaises(ValueError):
            compare_declarations(declaration("ohm", "resistance"),
                                 declaration("S", "electric_conductance"), mode="normalize")
        for unit in ("m", "µS", "Hz", "hours", "S/cm"):
            with self.subTest(unit=unit), self.assertRaises(ValueError):
                declaration(unit, "electric_conductance")

    def test_semantic_fields_are_not_units(self):
        q = declaration()
        for field, value in [("column", "y"), ("quantity_id", "current_density"),
                             ("statistic_id", "mean"), ("condition_id", "other_protocol")]:
            for mode in ("validate", "normalize"):
                with self.assertRaises(ValueError): compare_declarations(q, replace(q, **{field:value}), mode=mode)
        with self.assertRaises(ValueError): compare_declarations(q, declaration("V", "electric_potential"), mode="normalize")

    def test_unknown_mixed_alias_and_dimension_rejected(self):
        for unit in ("unknown", "A or nA", "", "arbitrary", "µA", "UA", " A", "A ", "dimensionless", "decades/V", "V/decade", "K", "degC"):
            with self.subTest(unit=unit), self.assertRaises(ValueError): declaration(unit)
        with self.assertRaises(ValueError): declaration("V")

    def test_declaration_bounds_and_types(self):
        for value in (None, True, 1, [], {}, "", " padded", "padded ", "x\x00y", "x\ny", "x"*257):
            with self.assertRaises(ValueError): replace(declaration(), statistic_id=value)
        for value in (None, {}, "A"):
            with self.assertRaises(ValueError): compare_declarations(declaration(), value)
        for mode in (None, True, [], "automatic"):
            with self.assertRaises(ValueError): compare_declarations(declaration(), declaration(), mode=mode)

    def test_unicode_control_categories_rejected_and_labels_preserved(self):
        for codepoint in (*range(0x20), *range(0x7f, 0xa0)):
            with self.subTest(codepoint=codepoint), self.assertRaises(ValueError):
                replace(declaration(), quantity_id="a" + chr(codepoint) + "b")
        for field in ("column", "quantity_id", "statistic_id", "condition_id"):
            value = "電流_中位數_測定é"
            declared = replace(declaration(), **{field: value})
            self.assertEqual(getattr(declared, field), value)
            self.assertEqual(compare_declarations(declared, declared), declared)

    def test_invalid_tokens_rejected_without_mutation(self):
        for value in (None, True, 1, 1.0, Decimal("1"), complex(1), "", " 1", "1 ", "NaN", "Infinity", "1A", "1_0", "1,0", "１", "1e201", "1e-201", ".1e-200", "1"*65, "1"*257):
            values = ["1", value]
            before = values.copy()
            with self.assertRaises(ValueError): normalize_decimal_tokens(values, declaration())
            self.assertEqual(values, before)
        for values in ([], (), "1", iter(["1"]), ["1"]*20001):
            with self.assertRaises(ValueError): normalize_decimal_tokens(values, declaration())

    def test_decimal_context_cannot_round_normalization(self):
        token = "1234567890123456789012345678901234567890123456789012345678901234"
        with localcontext() as ctx:
            ctx.prec = 2
            ctx.Emax, ctx.Emin = 9, -9
            got = normalize_decimal_tokens([token, "1e200", "1e-200"], declaration("pA"))
        self.assertEqual(got[0].as_tuple().digits, Decimal(token).as_tuple().digits)
        self.assertEqual(got[0].as_tuple().exponent, -12)
        self.assertEqual(got[1:], (Decimal("1e188"), Decimal("1e-212")))

    def test_float_boundary_integral_precision_overflow_underflow(self):
        for value in (Decimal(2**53+1), Decimal("1e309"), Decimal("1e-400"), Decimal("NaN"), Decimal("Infinity"), 1, True, "1"):
            with self.assertRaises(ValueError): decimal_to_float64([value])
        for value in (2**53, 2**53+2, 2**60):
            self.assertEqual(decimal_to_float64([Decimal(value)]), (float(value),))
        self.assertEqual(decimal_to_float64([Decimal("0.1")]), (0.1,))

    def test_float_boundary_collision_and_no_mutation(self):
        values = [Decimal("0.1000000000000000001"), Decimal("0.1000000000000000002")]
        before = values.copy()
        with self.assertRaisesRegex(ValueError, "collapse"): decimal_to_float64(values)
        self.assertEqual(values, before)
        self.assertEqual(decimal_to_float64([Decimal("0.1"), Decimal("0.10")]), (0.1, 0.1))

    def test_paired_boundary_rejects_cross_batch_collision_after_unit_conversion(self):
        left, right = declaration(), declaration("nA")
        a, b = ["0.1000000000000000001"], ["100000000.0000000002"]
        self.assertEqual(decimal_to_float64(normalize_decimal_tokens(a, left)), (0.1,))
        self.assertEqual(decimal_to_float64(normalize_decimal_tokens(b, right)), (0.1,))
        with self.assertRaisesRegex(ValueError, "collapse"):
            normalize_comparable_pair_to_float64(left, a, right, b)
        self.assertEqual(normalize_comparable_pair_to_float64(left, ["1e-9", "-0"], right, ["1"]),
                         (left, (1e-9, -0.0), (1e-9,)))
        incompatible = replace(right, condition_id="different")
        with self.assertRaisesRegex(ValueError, "declarations differ"):
            normalize_comparable_pair_to_float64(left, [None], incompatible, [None])
        with self.assertRaisesRegex(ValueError, "Combined values"):
            normalize_comparable_pair_to_float64(left, ["0"]*10000, right, ["0"]*10001)

    def test_stateless_and_no_default_provenance(self):
        q = declaration("nA")
        alone = normalize_decimal_tokens(["1"], q)
        combined = normalize_decimal_tokens(["1", "1e100", "1e-100"], q)
        self.assertEqual(alone[0], combined[0])
        self.assertIsInstance(alone, tuple)
        self.assertFalse(hasattr(alone, "provenance"))

if __name__ == "__main__": unittest.main()
