from fractions import Fraction
from pathlib import Path
import runpy
import unittest


run = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                         "examples/fixed-log-bins-under-gain/run.py"))


class FixedLogBinsGainTest(unittest.TestCase):
    def test_pooled_constant_and_fixed_bin_order_reversal(self):
        clean, shifted = [run["fixture"](gain) for gain in (1, 8)]
        self.assertEqual(clean["pooled_errors"], (1, 1, 1))
        self.assertEqual(shifted["pooled_errors"], clean["pooled_errors"])
        self.assertEqual(clean["binned_errors"], (0, 0, 0))
        self.assertEqual(shifted["binned_errors"], (2, 2, 0))
        self.assertLess(clean["binned_mae"], clean["pooled_mae"])
        self.assertEqual(shifted["binned_mae"], Fraction(4, 3))
        self.assertGreater(shifted["binned_mae"], shifted["pooled_mae"])

    def test_invented_oracle_does_not_claim_general_float_unit_support(self):
        for gain in (0, -1, 3, True, 2.0):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                run["fixture"](gain)
