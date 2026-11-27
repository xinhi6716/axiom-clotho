"""Exact descriptive coverage bounds with one shared label across arms."""

from fractions import Fraction
from itertools import combinations_with_replacement, product
import unittest

from groupbench.interval_evaluation import bound_paired_interval_coverage as bounds


ROW = 'row_weighted_coverage_b_minus_a_bounds'
GROUP = 'equal_group_coverage_b_minus_a_bounds'


class SharedTruthCoverageBoundsTests(unittest.TestCase):
    def test_endpoints_alone_miss_singleton_exclusive_coverage(self):
        # Neither forecast covers either truth endpoint. At the internal
        # singleton values each forecast can nevertheless win on coverage.
        uncertain = ('r', -1, 2, 0, 0, 1, 1)
        result = bounds([uncertain], expected_roster=[('r', 'g')])
        self.assertEqual(result[ROW], (-1, 1))
        self.assertEqual(result[GROUP], (-1, 1))
        self.assertEqual(result['n_rows_with_unresolved_coverage_difference'], 1)
        same = bounds([('r', -1, 2, 0, 0, 0, 0)], expected_roster=[('r', 'g')])
        self.assertEqual(same[ROW], (0, 0))
        self.assertEqual(same['n_rows_with_unresolved_coverage_difference'], 0)

    def test_known_closed_boundary_and_unequal_group_weighting(self):
        rows = [
            ('unknown', 0, 1, 0, 0, 1, 1),
            ('shared', 0, 1, 0, 1, 0, 1),
            ('known', 1, 1, 0, 0, 0, 1),
        ]
        roster = [('unknown', 'site1'), ('shared', 'site1'), ('known', 'site2')]
        report = bounds(rows[::-1], expected_roster=roster)
        self.assertEqual(report[ROW], (0, Fraction(2, 3)))
        self.assertEqual(report[GROUP], (Fraction(1, 4), Fraction(3, 4)))
        reversed_arms = bounds([row[:3] + row[5:] + row[3:5] for row in rows], expected_roster=roster[::-1])
        for weighting in (ROW, GROUP):
            low, high = report[weighting]
            self.assertEqual(reversed_arms[weighting], (-high, -low))

    def test_exact_grid_oracle_checks_midpoints_and_touching_endpoints(self):
        levels = (-1, 0, 1, 2)
        intervals = tuple(combinations_with_replacement(levels, 2))
        for (lo, hi), (al, ah), (bl, bh) in product(intervals, repeat=3):
            values = [Fraction(i, 2) for i in range(lo * 2, hi * 2 + 1)]
            oracle = [int(bl <= v <= bh) - int(al <= v <= ah) for v in values]
            result = bounds([('r', lo, hi, al, ah, bl, bh)], expected_roster=[('r', 'site')])
            self.assertEqual(result[ROW], (min(oracle), max(oracle)))

    def test_rejects_invalid_or_incomplete_roster(self):
        row = ('a', 0, 1, 0, 1, 0, 1)
        roster = [('a', 'source')]
        for rows, expected in (([row, row], roster), ([row], [*roster, ('a', 'other')]),
                               ([row], [*roster, ('b', 'source')]),
                               ([('a', 1, 0, 0, 1, 0, 1)], roster),
                               ([('a', 0, 1, 1, 0, 0, 1)], roster),
                               ([('a', 0, 1, 0, 1, None, 1)], roster),
                               ([row], [('a', True)])):
            with self.assertRaises(ValueError):
                bounds(rows, expected_roster=expected)
