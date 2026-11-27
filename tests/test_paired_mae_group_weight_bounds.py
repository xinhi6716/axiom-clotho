"""Exact same-roster sensitivity to independently declared group-weight ranges."""
from fractions import Fraction
from itertools import product
import unittest

from groupbench.metrics import paired_mae_group_weight_bounds as score


KEY = "target_group_weighted_mae_b_minus_a_bounds"


class PairedMAEGroupWeightBoundsTests(unittest.TestCase):
    def test_external_target_composition_can_exclude_equal_group_ranking(self):
        rows = [('a', 0, 2, 0), ('b', 0, 0, 1), ('c', 0, 0, 3)]
        roster = [('a', 'priority'), ('b', 'second'), ('c', 'third')]
        weights = {'priority': (.5, .75), 'second': (0, .5), 'third': (0, .25)}
        report = score(rows, expected_roster=roster, group_weight_bounds=weights)
        self.assertEqual(report["equal_group_b_minus_a"], Fraction(2, 3))
        self.assertFalse(report['equal_group_composition_feasible'])
        self.assertEqual(report[KEY], (Fraction(-5, 4), 0))
        swapped = [(key, y, b, a) for key, y, a, b in rows[::-1]]
        reverse = score(swapped, expected_roster=roster[::-1], group_weight_bounds=weights)
        self.assertEqual(reverse[KEY], (0, Fraction(5, 4)))

    def test_exact_simplex_grid_oracle_all_bounded_intervals(self):
        rows = [('a', 0, 2, 0), ('b', 0, 0, 1), ('c', 0, 0, 3), ('d', 0, 0, 5)]
        roster = [('a', 'g0'), ('b', 'g1'), ('c', 'g2'), ('d', 'g2')]
        endpoint_ranges = [(lo, hi) for lo in range(5) for hi in range(lo, 5)]
        allocations = [(a, b, 4-a-b) for a in range(5) for b in range(5-a)]
        for limits in product(endpoint_ranges, repeat=3):
            feasible = [weights for weights in allocations if all(
                lo <= w <= hi for w, (lo, hi) in zip(weights, limits))]
            if not feasible:
                continue
            constraint = {f'g{i}': (lo/4, hi/4) for i, (lo, hi) in enumerate(limits)}
            result = score(rows[::-1], expected_roster=roster, group_weight_bounds=constraint)
            oracle = [Fraction(-2*a+b+4*c, 4) for a, b, c in feasible]
            self.assertEqual(result[KEY], (min(oracle), max(oracle)))
            self.assertEqual(result['equal_group_b_minus_a'], 1)

    def test_zero_weights_still_require_complete_rows_and_bounds(self):
        rows = [('a', 0, 0, 1), ('b', 0, 1, 0)]
        roster = [('a', 'g0'), ('b', 'g1')]
        weights = {'g0': (1, 1), 'g1': (0, 0)}
        result = score(rows, expected_roster=roster, group_weight_bounds=weights)
        self.assertEqual(result[KEY], (1, 1))
        with self.assertRaisesRegex(ValueError, 'frozen roster'):
            score(rows[:1], expected_roster=roster, group_weight_bounds=weights)
        for invalid in ({'g0': (1, 1)}, {'g0': (1, 1), 'outside': (0, 0)},
                        {'g0': (.75, .75), 'g1': (.5, .5)},
                        {'g0': (.125, .125), 'g1': (.75, .75)},
                        {'g0': (.1, .1), 'g1': (.9, .9)},
                        {'g0': (False, 1), 'g1': (0, 0)},
                        {'g0': (.75, .25), 'g1': (0, 1)}):
            with self.assertRaises(ValueError):
                score(rows, expected_roster=roster, group_weight_bounds=invalid)
