from decimal import Decimal
from fractions import Fraction as F
import unittest

from groupbench.input_support import quantize_positive_operand as quantize


class QuantizedSupportInputsTest(unittest.TestCase):
    def test_decimal_halfway_and_zero_separate(self):
        step = F(Decimal("0.02"))
        self.assertIsNone(quantize(F("0.009999999999999999999999999999"), step))
        self.assertEqual(quantize(F("0.01"), step), F("0.02"))
        self.assertEqual(quantize(F("0.03"), step), F("0.04"))
        self.assertEqual(quantize(F("0.029999999999999999999999999999"), step), F("0.02"))

    def test_bounded_exhaustive_integer_grid_matches_nearest_distance(self):
        for denom in (3, 7, 21):
            for num in range(1, 7):
                step = F(num, denom)
                orig = [step * F(i, 14) for i in range(1, 171)]
                snapped = []
                for value in orig:
                    oracle = min((k * step for k in range(14)), key=lambda x: (abs(x - value), -x))
                    out = quantize(value, step)
                    self.assertEqual(out, oracle or None)
                    snapped.append(out if out is not None else F(0))
                self.assertEqual(snapped, sorted(snapped))

    def test_rescaling_and_domain(self):
        factor = F(10) ** 80
        for i in range(1, 30):
            x, step = F(i, 5), F(2)
            out = quantize(x, step)
            self.assertEqual(quantize(factor * x, factor * step), None if out is None else factor * out)
        for bad in (True, 1, 0.5, float("inf"), "0.1", Decimal("0.1")):
            with self.assertRaises(TypeError):
                quantize(bad, F(1))
            with self.assertRaises(TypeError):
                quantize(F(1), bad)
        for bad in (F(0), F(-1)):
            with self.assertRaises(ValueError):
                quantize(bad, F(1))
            with self.assertRaises(ValueError):
                quantize(F(1), bad)
