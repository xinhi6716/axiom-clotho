from fractions import Fraction
from itertools import product
import unittest

from groupbench.metrics import paired_mse_shared_shift_profile


class PairedMseSharedShiftProfileTests(unittest.TestCase):
    def test_opposite_rank_at_one_shift_and_distinct_exact_crossings(self):
        rows = [("x", 0, 0, 2), ("y", 0, 0, 2), ("z", 0, 0, 4)]
        roster = [("x", "chip1"), ("y", "chip1"), ("z", "chip2")]
        report = paired_mse_shared_shift_profile(rows, expected_roster=roster, shift_bounds=(0, 2))
        self.assertEqual(report["micro"]["mse_b_minus_a_bounds"], (Fraction(-8, 3), 8))
        self.assertEqual(report["micro"]["tie_shift"], Fraction(3, 2))
        self.assertEqual(report["equal_group"]["mse_b_minus_a_bounds"], (-2, 10))
        self.assertEqual(report["equal_group"]["tie_shift"], Fraction(5, 3))
        point = paired_mse_shared_shift_profile(rows, expected_roster=roster,
                                                shift_bounds=(1.625, 1.625))
        self.assertLess(point["micro"]["at_lower_shift_mse_b_minus_a"], 0)
        self.assertGreater(point["equal_group"]["at_lower_shift_mse_b_minus_a"], 0)

    def test_identical_predictions_cancel_and_endpoint_ties(self):
        roster = [("x", "chip1")]
        identical = paired_mse_shared_shift_profile([("x", -1e300, 1e300, 1e300)],
                                                   expected_roster=roster, shift_bounds=(-1e300, 1e300))
        self.assertTrue(identical["micro"]["tied_throughout"])
        self.assertIsNone(identical["micro"]["tie_shift"])
        for bounds, tied_all in [((1, 5), False), ((1, 1), True), ((-2, 0), False)]:
            out = paired_mse_shared_shift_profile([("x", 0, 0, 2)], expected_roster=roster,
                                                   shift_bounds=bounds)["micro"]
            self.assertEqual(out["tie_shift"], 1 if bounds[0] == 1 else None)
            self.assertEqual(out["tied_throughout"], tied_all)

    def test_bounds_match_direct_shared_shift_enumeration(self):
        roster = [("x", "chip1"), ("y", "chip2"), ("z", "chip2")]
        for a, b, c in product((-2, 0, 2), repeat=3):
            rows = [("x", 1, 0, a), ("y", -1, 2, b), ("z", 0, -1, c)]
            result = paired_mse_shared_shift_profile(rows, expected_roster=roster,
                                                     shift_bounds=(-1, 2))
            for key, weights in [("micro", (Fraction(1, 3),) * 3),
                                 ("equal_group", (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)))]:
                direct = [sum(w * ((pb - y - s)**2 - (pa - y - s)**2)
                              for w, (_, y, pa, pb) in zip(weights, rows))
                          for s in (Fraction(-1), Fraction(0), Fraction(1, 2), Fraction(2))]
                self.assertEqual(result[key]["mse_b_minus_a_bounds"], (min(direct), max(direct)))
                if result[key]["tie_shift"] is not None:
                    s = result[key]["tie_shift"]
                    self.assertEqual(sum(w * ((pb - y - s)**2 - (pa - y - s)**2)
                                         for w, (_, y, pa, pb) in zip(weights, rows)), 0)

    def test_rejects_reversed_or_nonfinite_shift_and_missing_or_duplicate_rows(self):
        roster = [("x", "a"), ("y", "b")]
        rows = [("x", 0, 0, 2), ("y", 0, 0, 2)]
        for bad_rows, bounds in ((rows, (1, 0)), (rows, (0, float("inf"))),
                                 (rows, (False, 1)), (rows[:1], (0, 1)),
                                 (rows[:1] * 2, (0, 1))):
            with self.subTest(rows=bad_rows, bounds=bounds), self.assertRaises(ValueError):
                paired_mse_shared_shift_profile(bad_rows, expected_roster=roster, shift_bounds=bounds)
