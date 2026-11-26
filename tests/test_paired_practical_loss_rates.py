"""Freeze the paired roster and margin before describing comparative losses."""

from fractions import Fraction
from itertools import product
import sys
import unittest

from groupbench.metrics import paired_practical_loss_rates


A = "a_lower_loss_beyond_margin"
B = "b_lower_loss_beyond_margin"
MID = "within_margin"


class PairedPracticalLossRateTests(unittest.TestCase):
    def test_large_mean_win_does_not_imply_most_queries_benefit(self):
        rows = [("a", 0, 0, 2), ("b", 0, 0, 2), ("c", 0, 0, 2), ("d", 0, 100, 0)]
        roster = [(row[0], "g" if row[0] != "d" else "h") for row in rows]
        a_mae = Fraction(sum(abs(x[1] - x[2]) for x in rows), len(rows))
        b_mae = Fraction(sum(abs(x[1] - x[3]) for x in rows), len(rows))
        self.assertLess(b_mae, a_mae)
        result = paired_practical_loss_rates(rows, expected_roster=roster, margin=1)
        self.assertEqual(result["row_weighted"], {A: Fraction(3, 4), MID: 0, B: Fraction(1, 4)})
        self.assertEqual(result["equal_group"], {A: Fraction(1, 2), MID: 0, B: Fraction(1, 2)})
        reversed_arms = [row[:2] + row[-2:][::-1] for row in rows[::-1]]
        swapped = paired_practical_loss_rates(reversed_arms, expected_roster=roster[::-1], margin=1)
        self.assertEqual(swapped["row_weighted"], {B: result["row_weighted"][A],
                                                   A: result["row_weighted"][B], MID: 0})

    def test_exact_inclusive_thresholds_and_wide_loss_differences(self):
        tiny = float.fromhex("0x0.0000000000001p-1022")
        huge = sys.float_info.max
        test_cases = [
            ((0, 0, tiny), tiny, MID), ((0, tiny, 0), tiny, MID),
            ((0, 0, tiny), 0, A), ((0, tiny, 0), 0, B),
            ((huge, huge, -huge), huge, A), ((-huge, huge, -huge), huge, B),
        ]
        for values, margin, which in test_cases:
            got = paired_practical_loss_rates([("r", *values)], expected_roster=[("r", "g")],
                                               margin=margin)
            self.assertEqual(got["row_weighted"][which], Fraction(1))
            self.assertEqual(sum(got["row_weighted"].values()), Fraction(1))
            self.assertEqual(got["margin_exact"], Fraction(margin))

    def test_all_small_pairs_and_margins_match_direct_oracle(self):
        values = (-2., -.5, 0., .25, 2.)
        for target, a, b, margin in product(values, values, values, (0., .25, 1.)):
            yy, aa, bb = map(Fraction, (target, a, b))
            gap = abs(bb - yy) - abs(aa - yy)
            which = A if gap > Fraction(margin) else B if gap < -Fraction(margin) else MID
            got = paired_practical_loss_rates([("r", target, a, b)], expected_roster=[("r", "g")],
                                               margin=margin)
            self.assertEqual(got["equal_group"][which], Fraction(1))

    def test_all_rows_and_literal_margins_required(self):
        row = ("a", 0, 0, 0)
        good = [("a", "g")]
        for bad in (True, -1, float("nan"), "0", None):
            with self.assertRaises(ValueError):
                paired_practical_loss_rates([row], expected_roster=good, margin=bad)
        for given, roster in (([row], [*good, ("b", "g")]), ([row, row], good),
                              ([row], [*good, ("a", "h")]), ([("a", None, 1, 1)], good),
                              ([row], [("a", False)])):
            with self.assertRaises(ValueError):
                paired_practical_loss_rates(given, expected_roster=roster, margin=1)
