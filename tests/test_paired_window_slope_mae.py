from fractions import Fraction
from itertools import product
import unittest

from groupbench.metrics import paired_window_slope_mae


class PairedWindowSlopeTests(unittest.TestCase):
    def test_equal_endpoint_absolute_error_can_hide_drift_and_unequal_steps(self):
        rows = [("p", 0, 2, 0, 2, -1, 1, 1, 1), ("q", 3, 7, 3, 5, 5, 7, 4, 4),
                ("r", -2, -1, 0, 1, 0, 0, 8, 9)]
        roster = [("r", "small"), ("p", "big"), ("q", "big")]
        result = paired_window_slope_mae(rows, expected_roster=roster)
        self.assertEqual(result["micro"], {
            "a_slope_mae": Fraction(1, 3), "b_slope_mae": Fraction(1, 2),
            "slope_mae_b_minus_a": Fraction(1, 6)})
        self.assertEqual(result["equal_group"]["slope_mae_b_minus_a"], Fraction(-1, 8))
        self.assertEqual(result, paired_window_slope_mae(rows[::-1], expected_roster=roster[::-1]))

    def test_direct_exact_subtract_divided_slopes_oracle(self):
        for a, b, y in product((-2., 0., 2.), repeat=3):
            rows = [("w", .125, .75, 1., y, .5, a, 0., b)]
            result = paired_window_slope_mae(rows, expected_roster=[("w", "g")])["micro"]
            _, t0, t1, y0, y1, a0, a1, b0, b1 = rows[0]
            dt = Fraction(t1) - Fraction(t0)
            y_slope = Fraction(y1) / dt - Fraction(y0) / dt
            for key, lo, hi in (("a_slope_mae", a0, a1), ("b_slope_mae", b0, b1)):
                self.assertEqual(result[key], abs(Fraction(hi) / dt - Fraction(lo) / dt - y_slope))

    def test_rejects_roster_and_time_defects(self):
        row = ("w", 0, 1, 0, 1, 0, 0, 1, 1)
        roster = [("w", "g")]
        for bad, expected in (
            ([row], roster + [("missing", "g")]), ([row, row], roster),
            ([("other", *row[1:])], roster), ([("w", 2, 1, *row[3:])], roster),
            ([("w", 1, 1, *row[3:])], roster), ([("w", True, 2, *row[3:])], roster),
        ):
            with self.subTest(bad=bad, expected=expected), self.assertRaises(ValueError):
                paired_window_slope_mae(bad, expected_roster=expected)
