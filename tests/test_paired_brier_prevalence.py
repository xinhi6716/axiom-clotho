"""Fixed-label paired Brier comparison under a declared prevalence shift."""
from fractions import Fraction
from itertools import product
import unittest

from groupbench.binary.metrics import paired_brier_prevalence_sensitivity as score


class PairedBrierPrevalenceTests(unittest.TestCase):
    def test_label_mix_can_reverse_an_unchanged_paired_ranking(self):
        rows = [('n', 0, .25, .5), ('p', 1, .25, .5)]
        roster = [('n', 'site'), ('p', 'site')]
        result = score(rows, expected_roster=roster, prevalence_range=(.25, .75))
        row = result['row_within_label']
        self.assertEqual(row['conditional_brier_b_minus_a'], (Fraction(3,16), Fraction(-5,16)))
        self.assertEqual(row['brier_b_minus_a_at_prevalence_endpoints'],
                         (Fraction(1,16), Fraction(-3,16)))
        self.assertEqual(row['tie_prevalence_in_declared_range'], Fraction(3,8))
        self.assertEqual(result['equal_group_within_label'], row)

    def test_single_class_groups_do_not_silently_get_dropped_or_reweighted(self):
        rows = [('n', 0, 0, 1), ('p', 1, 0, 1), ('only n', 0, 1, 0)]
        roster = [('n', 'site'), ('p', 'site'), ('only n', 'outer site')]
        result = score(rows, expected_roster=roster, prevalence_range=(0, 1))
        self.assertEqual(result['n_rows_by_label'], (2, 1))
        self.assertEqual(result['n_declared_groups_by_label'], (2, 1))
        self.assertIsNone(result['equal_group_within_label'])
        self.assertEqual(result['row_within_label']['conditional_brier_b_minus_a'], (0, -1))
        with self.assertRaisesRegex(ValueError, 'both observed classes'):
            score(rows[:1], expected_roster=roster[:1], prevalence_range=(0, 1))

    def test_independent_direct_cost_oracle_for_unequal_groups(self):
        roster = [('a', 'large'), ('b', 'large'), ('c', 'large'),
                  ('d', 'small'), ('e', 'small')]
        labels = [0, 0, 1, 0, 1]
        probabilities = (0, .5, 1)
        for probs in product(probabilities, repeat=len(labels)):
            rows = [(key, y, .25, p) for (key, _), y, p in zip(roster, labels, probs)]
            result = score(rows[::-1], expected_roster=roster, prevalence_range=(0, 1))
            gaps = [(Fraction(p)-y)**2 - (Fraction(1,4)-y)**2 for p, y in zip(probs, labels)]
            row = ((gaps[0]+gaps[1]+gaps[3])/3, (gaps[2]+gaps[4])/2)
            group = ((gaps[0]+gaps[1])/4 + gaps[3]/2, (gaps[2]+gaps[4])/2)
            for weights, expected in (('row_within_label', row), ('equal_group_within_label', group)):
                self.assertEqual(result[weights]['conditional_brier_b_minus_a'], expected)
                self.assertEqual(result[weights]['brier_b_minus_a_bounds'], tuple(sorted(expected)))

    def test_malformed_truth_probability_range_or_roster_reject(self):
        rows = [('a', 0, .5, 0), ('b', 1, .5, 1)]
        roster = [('a', 'g0'), ('b', 'g0')]
        for invalid in (None, (-1, 1), (1, 0), (0, 2), (0, True)):
            with self.assertRaises(ValueError):
                score(rows, expected_roster=roster, prevalence_range=invalid)
        for row in (('a', False, .5, 0), ('a', 0, 1.01, 0),
                    ('a', 0, .5, None), ('outside', 0, .5, 0)):
            with self.assertRaises(ValueError):
                score([row, rows[1]], expected_roster=roster, prevalence_range=(0, 1))
