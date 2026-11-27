from fractions import Fraction
from math import nextafter
import unittest

from groupbench.metrics import three_level_endpoint_contrast


class ThreeLevelEndpointContrastTests(unittest.TestCase):
    def test_unequal_steps_reversal_and_zero_curvature_on_linear_stratum(self):
        rows = [("peak", 5, 1), ("line", 2, 8), ("peak", 0, 1), ("line", 5, 17),
                ("peak", 2, 5), ("line", 0, 2)]
        actual = three_level_endpoint_contrast(rows, expected_strata=("peak", "line"), levels=(0, 2, 5))
        self.assertEqual(actual["per_stratum"]["peak"], {
            "left_slope": 2, "right_slope": Fraction(-4, 3),
            "right_minus_left_slope": Fraction(-10, 3), "second_divided_difference": Fraction(-2, 3),
            "strict_reversal": "peak", "has_flat_step": False})
        self.assertEqual(actual["per_stratum"]["line"]["right_minus_left_slope"], 0)
        self.assertEqual(actual["equal_stratum_mean"]["second_divided_difference"], Fraction(-1, 3))
        self.assertEqual((actual["n_peaks"], actual["n_valleys"]), (1, 0))
        self.assertEqual(actual, three_level_endpoint_contrast(rows[::-1],
                                                               expected_strata=("line", "peak"),
                                                               levels=(0, 2, 5)))

    def test_opposite_slopes_can_cancel_at_panel_mean_and_flat_never_reverses(self):
        levels = (0, 1, 4)
        rows = [(group, level, endpoint) for group, endpoints in
                (("p", (0, 3, 0)), ("v", (0, -3, 0)), ("flat", (0, 0, 0)))
                for level, endpoint in zip(levels, endpoints)]
        out = three_level_endpoint_contrast(rows, expected_strata=("p", "v", "flat"), levels=levels)
        self.assertEqual(out["equal_stratum_mean"]["right_minus_left_slope"], 0)
        self.assertEqual((out["n_peaks"], out["n_valleys"], out["n_strata_with_flat_step"]), (1, 1, 1))

    def test_strict_cell_roster_and_exact_numeric_levels(self):
        rows = [("p", 0, 0), ("p", 1, 2), ("p", 4, 4)]
        for bad, strata, levels in (
            (rows[:2], ("p",), (0, 1, 4)),
            (rows + rows[:1], ("p",), (0, 1, 4)),
            (rows, ("p", "p"), (0, 1, 4)),
            (rows, ("p", "unobserved"), (0, 1, 4)),
            (rows, ("p",), (0, 4, 1)),
            (rows, ("p",), (0, nextafter(1, 2), 4)),
            (rows, ("p",), (False, 1, 4)),
            (rows + [("unexpected", 0, 1)], ("p",), (0, 1, 4)),
        ):
            with self.subTest(bad=bad, strata=strata, levels=levels), self.assertRaises(ValueError):
                three_level_endpoint_contrast(bad, expected_strata=strata, levels=levels)
