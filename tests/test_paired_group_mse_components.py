"""Exact within-group / between-group descriptions on invented rows."""
from fractions import Fraction
from itertools import product
import sys
import unittest

from groupbench.metrics import paired_group_mse_components as score


class PairedGroupMSEComponentsTests(unittest.TestCase):
    def test_matching_aggregate_bias_and_mse_can_have_distinct_causes(self):
        rows = [('a', 0, 1, -1), ('b', 0, 1, 1), ('c', 0, -1, -1), ('d', 0, -1, 1)]
        roster = [('a', 'early'), ('b', 'early'), ('c', 'late'), ('d', 'late')]
        report = score(rows, expected_roster=roster)
        for view in ('row_weighted', 'equal_group'):
            a, b = (report[view][key] for key in ('a', 'b'))
            self.assertEqual((a['bias'], b['bias'], a['mse'], b['mse']), (0, 0, 1, 1))
            self.assertEqual((a['between_group_bias_dispersion'], b['between_group_bias_dispersion']), (1, 0))
            self.assertEqual((a['within_group_centered_mse'], b['within_group_centered_mse']), (0, 1))
            self.assertEqual(report[view]['b_minus_a']['between_group_bias_dispersion'], -1)
        self.assertEqual(report, score(rows[::-1], expected_roster=roster[::-1]))

    def test_exact_oracle_with_unequal_group_size(self):
        roster = [('a', 'large'), ('b', 'large'), ('c', 'small')]
        for a, b, c in product((-3, -1, 0, 2), repeat=3):
            rows = [('a', 0, 0, a), ('b', 0, 0, b), ('c', 0, 0, c)]
            report = score(rows, expected_roster=roster)
            for view, left, right in (('row_weighted', Fraction(2, 3), Fraction(1, 3)),
                                      ('equal_group', Fraction(1, 2), Fraction(1, 2))):
                large, small = Fraction(a + b, 2), Fraction(c)
                bias = left * large + right * small
                within = left * ((a - large)**2 + (b - large)**2) / 2
                between = left * (large - bias)**2 + right * (small - bias)**2
                result = report[view]['b']
                self.assertEqual(result, {
                    'bias': bias, 'mse': left * Fraction(a*a+b*b, 2) + right * c*c,
                    'within_group_centered_mse': within, 'between_group_bias_dispersion': between,
                    'overall_bias_squared': bias**2,
                })
                self.assertEqual(result['mse'], within + between + bias**2)

    def test_exact_arithmetic_and_strict_roster(self):
        huge = sys.float_info.max
        row = ('a', -huge, -huge, huge)
        roster = [('a', 'g')]
        report = score([row], expected_roster=roster)['equal_group']['b']
        self.assertEqual(report['mse'], 4 * Fraction(huge)**2)
        self.assertEqual(report['overall_bias_squared'], report['mse'])
        self.assertEqual(report['between_group_bias_dispersion'], 0)
        with self.assertRaisesRegex(ValueError, 'frozen roster'):
            score([row], expected_roster=[*roster, ('b', 'g')])
        with self.assertRaises(ValueError):
            score([('a', None, 0, 0)], expected_roster=roster)
        with self.assertRaises(ValueError):
            score([row], expected_roster=[('a', True)])
