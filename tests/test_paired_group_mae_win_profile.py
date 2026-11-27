from fractions import Fraction
from math import nextafter
import unittest

from groupbench.metrics import paired_group_mae_win_profile


class PairedGroupMaeWinProfileTests(unittest.TestCase):
    def test_group_prevalence_is_not_row_win_rate_or_mean_ranking(self):
        rows = [("b2", 0, 0, 20), ("a", 0, 1, 0), ("b1", 0, 0, 20),
                ("c", 0, 2, 0), ("d2", 0, 1, 0), ("d1", 0, 0, 1)]
        roster = [("c", "c"), ("a", "a"), ("d1", "d"), ("b1", "b"),
                  ("d2", "d"), ("b2", "b")]
        result = paired_group_mae_win_profile(rows, expected_roster=roster)
        self.assertEqual(result["group_counts"], {"a_wins": 1, "ties": 1, "b_wins": 2})
        self.assertEqual(result["group_fractions"]["b_wins"], Fraction(1, 2))
        self.assertEqual(result["micro_mae_b_minus_a"], Fraction(37, 6))
        self.assertEqual(result["equal_group_mae_b_minus_a"], Fraction(17, 4))
        self.assertEqual(result["per_group"][-1], {"group": "d", "n_rows": 2,
                                                 "mae_b_minus_a": 0, "classification": "ties"})
        self.assertEqual(result, paired_group_mae_win_profile(rows[::-1], expected_roster=roster[::-1]))

    def test_margin_includes_exact_boundary_and_excludes_next_float(self):
        d = nextafter(1, 2)
        rows = [("x", 0, 0, 1), ("y", 0, d, 0), ("z", 0, 0, d)]
        roster = [("x", "x"), ("y", "y"), ("z", "z")]
        result = paired_group_mae_win_profile(rows, expected_roster=roster, margin=1)
        self.assertEqual([item["classification"] for item in result["per_group"]],
                         ["ties", "b_wins", "a_wins"])

    def test_rejects_incomplete_duplicate_or_invalid_pairs(self):
        rows = [("x", 0, 1, 0), ("y", 0, 0, 1)]
        roster = [("x", "h1"), ("y", "h2")]
        for invalid, margin in ((rows[:1], 0), (rows[:1] * 2, 0), (rows, -1), (rows, True),
                                (rows[:1] + [("y", 0, 0, float("nan"))], 0)):
            with self.subTest(rows=invalid, margin=margin), self.assertRaises(ValueError):
                paired_group_mae_win_profile(invalid, expected_roster=roster, margin=margin)
