from fractions import Fraction
from itertools import permutations
import unittest

from groupbench.metrics import paired_group_bias_cancellation_audit


class PairedGroupBiasCancellationTests(unittest.TestCase):
    def test_equal_signed_pooled_bias_hides_distinct_group_errors(self):
        rows = [("p", 0, 0, 6), ("q", 0, 0, -2), ("r", 0, 0, -2), ("s", 0, 0, -2)]
        roster = [("p", "small"), ("q", "large"), ("r", "large"), ("s", "large")]
        out = paired_group_bias_cancellation_audit(rows, expected_roster=roster)
        self.assertEqual(out["row_mass"]["a"]["signed_bias"], 0)
        self.assertEqual(out["row_mass"]["b"], {
            "signed_bias": 0, "mean_absolute_group_bias": 3, "cancellation_gap": 3})
        self.assertEqual(out["equal_group"]["b_minus_a"], {
            "signed_bias": 2, "mean_absolute_group_bias": 4, "cancellation_gap": 2})
        self.assertEqual(out["per_group"]["small"]["b"], 6)
        self.assertEqual(out["per_group"]["large"]["b"], -2)

    def test_exact_group_sums_under_cancellation_and_reordering(self):
        rows = [("x", 1, 1e16, 1e16), ("y", 0, -1e16, -1e16), ("z", 0, 0, 2)]
        roster = [("z", "g"), ("y", "g"), ("x", "g")]
        for order in permutations(rows):
            out = paired_group_bias_cancellation_audit(order, expected_roster=roster)
            self.assertEqual(out["row_mass"]["a"]["signed_bias"], Fraction(-1, 3))
            self.assertEqual(out["row_mass"]["b"]["signed_bias"], Fraction(1, 3))
            self.assertEqual(out["equal_group"]["b_minus_a"]["cancellation_gap"], 0)

    def test_required_complete_roster_and_valid_residuals(self):
        rows = [("x", 1, 2, 3), ("y", 0, 0, 0)]
        roster = [("x", "a"), ("y", "b")]
        for bad in (rows[:1], rows[:1] * 2, rows[:1] + [("y", 0, 0, True)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired_group_bias_cancellation_audit(bad, expected_roster=roster)
