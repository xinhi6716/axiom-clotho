"""Sharp paired quantile-score bounds from shared, interval-censored truths."""

from fractions import Fraction
from itertools import combinations_with_replacement, product
import unittest

from groupbench.metrics import paired_quantile_pinball_bounds as bounds, paired_quantile_pinball_loss


ROW = 'row_weighted_pinball_b_minus_a_bounds'
GROUP = 'equal_group_pinball_b_minus_a_bounds'


class PairedQuantileBoundsTests(unittest.TestCase):
    def test_shared_truth_restricts_extrema_relative_to_unpaired_scores(self):
        roster = [('a', 'site')]
        report = bounds([('a', 0, 3, 1, 2)], expected_roster=roster, quantile=.75)
        self.assertEqual(report[ROW], (Fraction(-3, 4), Fraction(1, 4)))
        self.assertEqual(report[GROUP], report[ROW])
        independent_a = (0, Fraction(3, 2))
        independent_b = (0, Fraction(3, 4))
        self.assertGreater(report[ROW][0], independent_b[0] - independent_a[1])
        self.assertLess(report[ROW][1], independent_b[1] - independent_a[0])
        for forecasts in ((1, 1), (3, 4)):
            constant = bounds([('a', 0, 3, *forecasts)], expected_roster=roster, quantile=.75)
            self.assertEqual(constant['n_nondegenerate_truth_intervals'], 1)
            self.assertEqual(constant['n_nondegenerate_paired_cost_intervals'], 0)

    def test_endpoints_are_sharp_against_dense_independent_grid(self):
        grid = (-1, 0, 1, 2)
        for (lo, hi), a, b, q in product(combinations_with_replacement(grid, 2), grid,
                                          grid, (Fraction(1, 8), Fraction(1, 2), Fraction(7, 8))):
            def loss(y, prediction):
                return max(q * (y - prediction), (1-q) * (prediction - y))
            oracle = [loss(y, b) - loss(y, a)
                      for y in (Fraction(t, 4) for t in range(lo * 4, hi * 4 + 1))]
            report = bounds([('r', lo, hi, a, b)], expected_roster=[('r', 'g')], quantile=float(q))
            self.assertEqual(report[ROW], (min(oracle), max(oracle)))

    def test_unequal_groups_and_known_truth_reduce_to_the_point_scorer(self):
        rows = [('a', 0, 3, 1, 2), ('b', 2, 2, 0, 0), ('c', 0, 0, 0, 4)]
        roster = [('a', 'big'), ('b', 'big'), ('c', 'small')]
        q = .75
        result = bounds(rows, expected_roster=roster, quantile=q)
        self.assertEqual(result[ROW], (Fraction(1, 12), Fraction(5, 12)))
        self.assertEqual(result[GROUP], (Fraction(5, 16), Fraction(9, 16)))
        fully_observed = [(key, lo, lo, a, b) for key, lo, _, a, b in rows]
        points = [(key, lo, a, b) for key, lo, _, a, b in rows]
        merged = paired_quantile_pinball_loss(points, expected_roster=roster, quantile=q)
        observed = bounds(fully_observed[::-1], expected_roster=roster[::-1], quantile=q)
        for group_key, output in (('row_weighted', ROW), ('equal_group', GROUP)):
            self.assertEqual(observed[output], (merged[group_key]['b_minus_a']['pinball_loss'],) * 2)

    def test_missing_reversed_duplicate_and_invalid_metadata_fail(self):
        roster, row = [('a', 'g')], ('a', 0, 1, 0, 1)
        for q in (None, True, 0, 1, float('inf')):
            with self.assertRaises(ValueError):
                bounds([row], expected_roster=roster, quantile=q)
        for rows, keys in (([row], [*roster, ('b', 'g')]), ([row, row], roster),
                           ([row], [*roster, ('a', 'h')]),
                           ([('a', 1, 0, 0, 1)], roster), ([('a', 0, None, 0, 1)], roster),
                           ([('a', 0, 1, None, 1)], roster)):
            with self.assertRaises(ValueError):
                bounds(rows, expected_roster=keys, quantile=.75)
