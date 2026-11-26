from fractions import Fraction
from pathlib import Path
from runpy import run_path
from unittest import TestCase

tails = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                     / "prespecified-balance-rerandomization" / "run.py"))["tails"]


class PrespecifiedBalanceRerandomizationTests(TestCase):
    def test_both_directions_and_inclusive_ties(self):
        self.assertEqual(tails((0, 1, 2, 6, 7, 12), (0, 1, 5)),
                         (6, 20, Fraction(2, 3), Fraction(9, 10)))
        self.assertEqual(tails((1, 10, 4, 8, 9, 5), (0, 2, 5)),
                         (6, 20, Fraction(1, 3), Fraction(1, 10)))
        self.assertEqual(tails((0,) * 6, (0, 1, 5)), (6, 20, 1, 1))

    def test_invalid_observed_allocation_rejected(self):
        with self.assertRaisesRegex(ValueError, "balance rule"):
            tails((0,) * 6, (0, 1, 2))
