"""Check exact design expectations and refusal of unidentified targets."""

from fractions import Fraction as F
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/unequal-block-sampling/run.py"


class UnequalBlockSamplingExampleTests(unittest.TestCase):
    def test_exact_lottery_and_clipping(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True).splitlines()
        self.assertIn("naive_expectation=1/4, ht_expectation=1/2", out[0])
        self.assertIn("ht_expectation=3/4, ht_oracle_variance=27/16", out[1])
        self.assertTrue(out[1].endswith("clipped_at_two_expectation=3/8"))

    def test_no_support_and_missing_outcomes_are_not_silently_ignored(self):
        replay = runpy.run_path(str(EXAMPLE))["replay"]
        with self.assertRaisesRegex(ValueError, "positive inclusion"):
            replay((("A", F(1, 2), F(1), F(0)), ("B", F(1, 2), F(0), F(1))))
        with self.assertRaisesRegex(ValueError, "known outcomes"):
            replay((("A", F(1), F(1), None),))
        with self.assertRaisesRegex(ValueError, "one entry"):
            replay((("A", F(1, 2), F(1, 2), F(0)),)*2)

    def test_listing_order_does_not_change_exact_receipt(self):
        replay = runpy.run_path(str(EXAMPLE))["replay"]
        rows = (("A", F(1, 4), F(3, 4), F(0)), ("B", F(3, 4), F(1, 4), F(1)))
        self.assertEqual(replay(rows), replay(rows[::-1]))
