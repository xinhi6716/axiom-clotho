"""Exact paired score for forecasts targeting a frozen conditional quantile."""

from fractions import Fraction
from itertools import product
import sys
import unittest

from groupbench.metrics import paired_quantile_pinball_loss as score


class PairedQuantilePinballTests(unittest.TestCase):
    def test_upper_quantile_ranking_is_not_absolute_error_ranking(self):
        rows = [('one', 0, -1, 2), ('two', 0, -1, 2)]
        roster = [('one', 'site0'), ('two', 'site1')]
        report = score(rows, expected_roster=roster, quantile=.875)
        a, b = (report['row_weighted'][arm] for arm in ('a', 'b'))
        self.assertEqual(a['underprediction_contribution'], Fraction(7, 8))
        self.assertEqual(b['overprediction_contribution'], Fraction(1, 4))
        self.assertLess(b['pinball_loss'], a['pinball_loss'])
        median = score(rows, expected_roster=roster, quantile=.5)['row_weighted']
        self.assertEqual(median['a']['pinball_loss'], Fraction(1, 2))
        self.assertEqual(median['b']['pinball_loss'], 1)

    def test_exact_independent_oracle_with_unequal_groups(self):
        roster = [('a', 'large'), ('b', 'large'), ('c', 'small')]
        for e0, e1, e2, q in product((-2, -1, 0, .5), (-2, -1, 0, .5),
                                     (-2, -1, 0, .5), (.125, .5, .875)):
            rows = [('a', 1, 1, 1+e0), ('b', 1, 1, 1+e1), ('c', 1, 1, 1+e2)]
            result = score(rows[::-1], expected_roster=roster, quantile=q)
            levels = [Fraction(x) for x in (e0, e1, e2)]
            for view, weights in (('row_weighted', (Fraction(1, 3),) * 3),
                                  ('equal_group', (Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)))):
                losses = [max(-Fraction(q)*e, (1-Fraction(q))*e) for e in levels]
                base = result[view]['b_minus_a']
                self.assertEqual(base['pinball_loss'], sum(w*x for w, x in zip(weights, losses)))
                self.assertEqual(base['pinball_loss'],
                                 base['underprediction_contribution'] + base['overprediction_contribution'])

    def test_exact_extreme_and_fail_closed_inputs(self):
        huge = sys.float_info.max
        row = ('r', huge, -huge, huge)
        roster = [('r', 'g')]
        result = score([row], expected_roster=roster, quantile=.875)
        self.assertEqual(result['equal_group']['a']['pinball_loss'], Fraction(7, 4) * Fraction(huge))
        for invalid in (None, True, 0, 1, -1, float('inf')):
            with self.assertRaises(ValueError):
                score([row], expected_roster=roster, quantile=invalid)
        with self.assertRaisesRegex(ValueError, 'frozen roster'):
            score([row], expected_roster=[*roster, ('unavailable', 'g')], quantile=.5)
        with self.assertRaises(ValueError):
            score([('r', None, 0, 1)], expected_roster=roster, quantile=.5)
