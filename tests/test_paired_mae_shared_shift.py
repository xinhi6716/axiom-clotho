"""Exact fixed-roster comparisons under a single target calibration shift."""
from fractions import Fraction
import random
import unittest

from groupbench.metrics import paired_mae_missing_bounds, paired_mae_shared_shift_bounds


def sweep(rows, bounds):
    return paired_mae_shared_shift_bounds(
        rows, expected_roster=tuple(row[0] for row in rows), shift_bounds=bounds
    )


class SharedShiftTests(unittest.TestCase):
    def test_interior_gain_can_reverse_favorable_endpoints(self):
        rows = [("a", 0, -3, -2, 1), ("b", 0, -1, -2, 1), ("c", 0, -1, 0, 1)]
        r = sweep(rows, (-2, 2))
        self.assertEqual(r["at_lower_shift_mae_b_minus_a"], Fraction(-1, 3))
        self.assertEqual(r["at_upper_shift_mae_b_minus_a"], Fraction(-1, 3))
        self.assertEqual(r["max_mae_b_minus_a"], Fraction(1, 3))
        self.assertEqual(r["max_attaining_shift"], -1)
        self.assertEqual(r["min_attaining_shift"], -2)

    def test_independent_row_offsets_invent_a_gap(self):
        rows = [("a", 0, -1, 1, 1), ("b", 0, 1, -1, 1)]
        r = sweep(rows, (-1, 1))
        independent = paired_mae_missing_bounds(
            [(key, None, a, b) for key, _, a, b, _ in rows],
            expected_roster=("a", "b"), truth_bounds={"a": (-1, 1), "b": (-1, 1)},
        )
        self.assertEqual((r["min_mae_b_minus_a"], r["max_mae_b_minus_a"]), (0, 0))
        self.assertEqual(r["max_attaining_shift"], -1)
        self.assertEqual(independent["full_mae_b_minus_a_upper"], 2)

    def test_exact_weighted_brute_force_with_extreme_scales_and_ties(self):
        rng = random.Random(8710)
        choices = [-1.0e300, -8, -0.5, -5e-324, 0, 5e-324, 0.5, 8, 1.0e300]
        weights = [0.125, 1, 3, 1e300, 5e-324]
        for _ in range(500):
            rows = [(str(i), *rng.choices(choices, k=3), rng.choice(weights))
                    for i in range(rng.randint(1, 8))]
            lo, hi = sorted(rng.choices(choices, k=2))
            r = sweep(rows, (lo, hi))
            data = [tuple(map(Fraction, row[1:])) for row in rows]
            edges = sorted({Fraction(lo), Fraction(hi)} | {
                p-y for y, a, b, w in data for p in (a, b) if lo < p-y < hi
            })
            scores = [(sum(w * (abs(b-y-x)-abs(a-y-x)) for y, a, b, w in data)
                       / sum(w for y, a, b, w in data), x) for x in edges]
            minimum = min(scores)
            maximum = max(scores, key=lambda pair: (pair[0], -pair[1]))
            self.assertEqual((r["min_mae_b_minus_a"], r["min_attaining_shift"]), minimum)
            self.assertEqual((r["max_mae_b_minus_a"], r["max_attaining_shift"]), maximum)
            self.assertEqual(r["at_lower_shift_mae_b_minus_a"], scores[0][0])
            self.assertEqual(r["at_upper_shift_mae_b_minus_a"], scores[-1][0])
            self.assertEqual(r, sweep(rows[::-1], (lo, hi)))

    def test_invalid_roster_units_and_weights_fail_closed(self):
        row = ("a", 0, 1, 2, 1)
        for rows, roster, limits in [
            ([row], ("a", "b"), (-1, 1)), ([row], ("a", "a"), (-1, 1)),
            ([row, row], ("a",), (-1, 1)), ([row], ("b",), (-1, 1)),
            ([row], ("a",), (1, -1)), ([row], ("a",), (0, None)),
            ([row], ("a",), (0, float("inf"))),
            ([row[:-1] + (0,)], ("a",), (-1, 1)),
            ([row[:-1] + (True,)], ("a",), (-1, 1)),
            ([("a", None, 1, 2, 1)], ("a",), (-1, 1)),
        ]:
            with self.subTest(rows=rows, roster=roster, limits=limits), self.assertRaises(ValueError):
                paired_mae_shared_shift_bounds(rows, expected_roster=roster, shift_bounds=limits)


if __name__ == "__main__":
    unittest.main()
