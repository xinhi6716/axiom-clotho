"""Exact, fixed-cohort sensitivity to absolute vs squared loss."""
from fractions import Fraction
from itertools import product
import sys
import unittest

from groupbench.metrics import paired_absolute_squared_loss_tradeoff as compare


ABS = "normalized_absolute_b_minus_a"
SQ = "normalized_squared_b_minus_a"
WEIGHT = "tie_squared_weight"


class PairedLossTradeoffTests(unittest.TestCase):
    def test_mean_and_tail_sensitive_rankings_disagree(self):
        rows = [("early", 0, 3, 0), ("late", 0, 3, 5)]
        roster = [("early", "site1"), ("late", "site2")]
        result = compare(rows, expected_roster=roster, error_scale=1)
        for key in ("row_weighted", "equal_group"):
            got = result[key]
            self.assertEqual((got[ABS], got[SQ], got[WEIGHT]),
                             (Fraction(-1, 2), Fraction(7, 2), Fraction(1, 8)))
            self.assertTrue(got["strict_interior_rank_reversal"])
        scaled = [(row[0], *(16 * value for value in row[1:])) for row in rows[::-1]]
        self.assertEqual(compare(scaled, expected_roster=roster[::-1], error_scale=16)["row_weighted"],
                         result["row_weighted"])

    def test_group_weighting_changes_tie_without_changing_roster(self):
        rows = [("a", 0, 3, 0), ("b", 0, 3, 5), ("b2", 0, 3, 5)]
        roster = [("a", "site1"), ("b", "site2"), ("b2", "site2")]
        result = compare(rows, expected_roster=roster, error_scale=1)
        self.assertEqual((result["row_weighted"][ABS], result["row_weighted"][SQ]),
                         (Fraction(1, 3), Fraction(23, 3)))
        self.assertIsNone(result["row_weighted"][WEIGHT])
        self.assertEqual(result["equal_group"][WEIGHT], Fraction(1, 8))

    def test_ties_phase_and_extreme_represented_values(self):
        rows = [("a", 0, 3, 0), ("b", 0, 3, 6)]
        roster = [("a", "group"), ("b", "group")]
        got = compare(rows, expected_roster=roster, error_scale=1)["row_weighted"]
        self.assertEqual(got[WEIGHT], 0)
        self.assertFalse(got["strict_interior_rank_reversal"])
        tie = compare([("a", 0, -3, 3)], expected_roster=roster[:1], error_scale=1)["row_weighted"]
        self.assertTrue(tie["all_weights_tied"])
        self.assertIsNone(tie[WEIGHT])
        huge = sys.float_info.max
        giant = compare([("a", huge, huge, -huge)], expected_roster=roster[:1],
                        error_scale=huge)["row_weighted"]
        self.assertEqual((giant[ABS], giant[SQ]), (2, 4))

    def test_exact_input_validation_and_grid_matches_direct_fraction(self):
        roster = [("x", "group"), ("y", "group")]
        for u, v, x, y in product(range(4), repeat=4):
            rows = [("x", 0, u, v), ("y", 0, x, y)]
            got = compare(rows, expected_roster=roster, error_scale=.5)["row_weighted"]
            a, b = Fraction(v + y - u - x), Fraction(2 * (v*v + y*y - u*u - x*x))
            self.assertEqual((got[ABS], got[SQ]), (a, b))
            if got[WEIGHT] is not None:
                self.assertEqual((1 - got[WEIGHT]) * a + got[WEIGHT] * b, 0)
        for invalid in (0, -1, True, None, float("inf")):
            with self.assertRaises(ValueError):
                compare([("x", 0, 0, 0)], expected_roster=roster[:1], error_scale=invalid)
        with self.assertRaisesRegex(ValueError, "match the frozen roster"):
            compare([("x", 0, 0, 0)], expected_roster=roster, error_scale=1)
