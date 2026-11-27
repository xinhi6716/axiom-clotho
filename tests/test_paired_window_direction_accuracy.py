from fractions import Fraction
from itertools import product
import unittest

from groupbench.interval_evaluation import bound_paired_window_direction_accuracy


class PairedWindowDirectionTests(unittest.TestCase):
    def test_equal_forecast_classes_cancel_even_with_ambiguous_truth(self):
        rows = [("x", 0, 2, 0, 2, 3, 4, 8, 9)]
        out = bound_paired_window_direction_accuracy(rows, expected_roster=[("x", "g")], deadband=1)
        self.assertEqual(out["n_direction_ambiguous_windows"], 1)
        self.assertEqual(out["row_weighted_accuracy_b_minus_a_bounds"], (0, 0))

    def test_inclusive_deadband_and_unequal_group_weights(self):
        rows = [("x", 0, 0, 1, 1, 0, 2, 0, 1), ("y", 0, 0, -2, -2, 0, 0, 0, -2),
                ("z", 0, 0, 1, 2, 0, 1, 0, 2)]
        roster = [("x", "big"), ("y", "big"), ("z", "small")]
        out = bound_paired_window_direction_accuracy(rows, expected_roster=roster, deadband=1)
        self.assertEqual(out["row_weighted_accuracy_b_minus_a_bounds"], (Fraction(1, 3), 1))
        self.assertEqual(out["equal_group_accuracy_b_minus_a_bounds"], (0, 1))
        self.assertEqual(out["n_direction_ambiguous_windows"], 1)

    def test_exhaustive_anchor_truths_and_fail_closed(self):
        roster = [("x", "g")]
        def direction(x):
            return (x > 1) - (x < -1)
        for slo, shi, elo, ehi in product(range(3), repeat=4):
            if slo > shi or elo > ehi:
                continue
            for a, b in product(range(-2, 3), repeat=2):
                row = ("x", slo, shi, elo, ehi, 0, a, 0, b)
                all_truth = [direction(y - x) for x in range(slo, shi + 1) for y in range(elo, ehi + 1)]
                expected = [int(y == direction(b)) - int(y == direction(a)) for y in all_truth]
                out = bound_paired_window_direction_accuracy([row], expected_roster=roster, deadband=1)
                self.assertEqual(out["equal_group_accuracy_b_minus_a_bounds"],
                                 (min(expected), max(expected)))
        row = ("x", 0, 0, 0, 0, 0, 0, 0, 0)
        for rows, expected, band in (([row], roster, -1), ([row, row], roster, 0),
                                     ([row], roster + [("y", "g")], 0),
                                     ([("x", 1, 0, *row[3:])], roster, 0)):
            with self.subTest(rows=rows, expected=expected, band=band), self.assertRaises(ValueError):
                bound_paired_window_direction_accuracy(rows, expected_roster=expected, deadband=band)
