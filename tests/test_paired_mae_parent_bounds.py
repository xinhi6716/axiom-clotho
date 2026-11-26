"""Invented exact paired sensitivity with independently declared parent membership."""
from fractions import Fraction
from itertools import product
import unittest

from groupbench.metrics import paired_mae_missing_bounds, paired_mae_parent_bounds


class PairedParentBoundsTests(unittest.TestCase):
    def test_unequal_parent_sizes_reverse_ranking_without_changing_pairs(self):
        rows = [("a1", 1, 0, 1), ("a2", 1, 0, 1), ("a3", 1, 0, 1), ("b1", 0, 0, 2)]
        roster = [("a1", "A"), ("a2", "A"), ("a3", "A"), ("b1", "B")]
        row = paired_mae_missing_bounds(rows, expected_roster=[key for key, _ in roster])
        parent = paired_mae_parent_bounds(rows[::-1], expected_roster=roster[::-1])
        self.assertEqual(row["full_mae_b_minus_a_lower"], Fraction(-1, 4))
        self.assertEqual(parent["available_equal_parent_mae_b_minus_a"], Fraction(1, 2))
        self.assertEqual(parent["full_equal_parent_mae_b_minus_a_lower"], Fraction(1, 2))
        self.assertEqual(parent["full_equal_parent_mae_b_minus_a_upper"], Fraction(1, 2))
        self.assertEqual([r["parent"] for r in parent["per_parent"]], ["A", "B"])
        self.assertEqual(parent, paired_mae_parent_bounds(rows, expected_roster=roster))

    def test_parent_without_observations_is_not_dropped(self):
        rows = [("a", 1, 0, 1), ("b", None, 0, 2)]
        roster = [("a", "A"), ("b", "B")]
        report = paired_mae_parent_bounds(rows, expected_roster=roster, truth_bounds={"b": (0, 1)})
        self.assertEqual((report["n_missing"], report["n_parents_with_observations"]), (1, 1))
        self.assertIsNone(report["available_equal_parent_mae_b_minus_a"])
        self.assertIsNone(report["per_parent"][1]["available_mae_b_minus_a"])
        self.assertEqual(report["full_equal_parent_mae_b_minus_a_lower"], Fraction(-1, 2))
        self.assertEqual(report["full_equal_parent_mae_b_minus_a_upper"], Fraction(1, 2))

    def test_bounded_mixed_groups_match_independent_exact_cartesian_oracle(self):
        rows = [("a", 1, 0, 1), ("b", None, 3, 0), ("c", None, 1, 2),
                ("d", None, 2, 0), ("e", 0, 0, 1), ("f", 0, 0, 1)]
        roster = [("a", "A"), ("b", "A"), ("c", "A"), ("d", "B"), ("e", "B"), ("f", "C")]
        bounds = {"b": (None, 1), "d": (1, 2)}
        grid = [Fraction(i, 2) for i in range(-8, 9)]
        possibilities = []
        for b_y, c_y, d_y in product([v for v in grid if v <= 1], grid, [v for v in grid if 1 <= v <= 2]):
            actuals = [1, b_y, c_y, d_y, 0, 0]
            d = [abs(b - y) - abs(a - y) for (_, _, a, b), y in zip(rows, actuals)]
            possibilities.append((sum(map(Fraction, d[:3])) / 3
                                  + sum(map(Fraction, d[3:5])) / 2 + d[5]) / 3)
        report = paired_mae_parent_bounds(rows, expected_roster=roster, truth_bounds=bounds)
        self.assertEqual(report["full_equal_parent_mae_b_minus_a_lower"], min(possibilities))
        self.assertEqual(report["full_equal_parent_mae_b_minus_a_upper"], max(possibilities))
        self.assertEqual(report["available_equal_parent_mae_b_minus_a"], Fraction(1, 3))

    def test_invalid_rosters_rows_and_bounds_fail_closed(self):
        rows = [("a", 0, 0, 1), ("b", None, 1, 0)]
        roster = [("a", "A"), ("b", "B")]
        cases = [
            (rows, [("a", "A"), ("a", "B")], None),
            (rows[:1], roster, None), (rows + rows[:1], roster, None),
            (rows, [("a", "A"), ("c", "B")], None),
            (rows, [("a", "A"), ("b", " ")], None),
            (rows, [("a", "A"), ("b", "\ud800")], None),
            (rows, [("a", "A"), ("b", 1)], None),
            (rows, [("a", "A"), ["b", "B"]], None),
            (rows, roster, {"c": (0, 1)}),
            (rows, roster, {"a": (0, 1)}),
            (rows, roster, {"b": (1, 0)}),
            (rows, roster, {"b": (None, float("inf"))}),
            (rows, roster, {"b": (None, 2**53 + 1)}),
            (rows, roster, [("b", (0, 1))]),
            ([rows[0], ("b", None, None, 0)], roster, None),
        ]
        for entries, expected, bounds in cases:
            with self.subTest(entries=entries, expected=expected, bounds=bounds), self.assertRaises(ValueError):
                paired_mae_parent_bounds(entries, expected_roster=expected, truth_bounds=bounds)


if __name__ == "__main__":
    unittest.main()
