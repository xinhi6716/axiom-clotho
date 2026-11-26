"""Explicit costs on the same uncertain threshold outcome, fixed before scoring."""
from fractions import Fraction
from itertools import product
import unittest

from groupbench.metrics import paired_threshold_decision_cost_bounds as bounds


ROW = 'row_weighted_b_minus_a_cost_bounds'
GROUP = 'equal_group_b_minus_a_cost_bounds'


class PairedThresholdDecisionBoundsTests(unittest.TestCase):
    def test_shared_ambiguous_truth_cancels_except_where_actions_differ(self):
        # At retention threshold one could see only a censored interval.
        rows = [
            ('same', 8, 12, 9, 9),        # both say no event
            ('different', 9, 10, 9, 10),  # costs -5 or +2 if B says event
            ('known', 10, 10, 9, 10),      # threshold equality is a known event: -5
        ]
        roster = [('same', 'die1'), ('different', 'die1'), ('known', 'die2')]
        kw = dict(threshold=10, false_positive_cost=2, false_negative_cost=5)
        report = bounds(rows, expected_roster=roster, **kw)
        self.assertEqual(report['n_ambiguous_truth_rows'], 2)
        self.assertEqual(report['n_ambiguous_truth_rows_with_action_disagreement'], 1)
        self.assertEqual(report[ROW], (Fraction(-10, 3), -1))
        self.assertEqual(report[GROUP], (Fraction(-15, 4), -2))
        reverse = bounds([row[:-2] + row[-2:][::-1] for row in rows[::-1]],
                         expected_roster=roster[::-1], **kw)
        for key in (ROW, GROUP):
            got = report[key]
            self.assertEqual(reverse[key], (-got[1], -got[0]))

    def test_sharp_bounds_match_direct_completion_oracle(self):
        options = list(product(((-2, -1), (-1, 0), (0, 2)), (-1, 0), (-1, 0)))
        roster = [('r0', 'g0'), ('r1', 'g0'), ('r2', 'g1')]
        for states in product(options, repeat=3):
            rows = [(f'r{i}', *limits, a, b) for i, (limits, a, b) in enumerate(states)]
            result = bounds(rows, expected_roster=roster, threshold=0,
                            false_positive_cost=1.5, false_negative_cost=4)
            feasible = [tuple({v >= 0 for v in interval}) for interval, _, _ in states]
            differences = []
            def cost(actual, predicted):
                return (Fraction(3, 2) if not actual and predicted else
                        4 if actual and not predicted else 0)
            for actuals in product(*feasible):
                gaps = [cost(actual, b >= 0) - cost(actual, a >= 0)
                        for actual, (_, a, b) in zip(actuals, states)]
                differences.append((Fraction(sum(gaps), 3), Fraction(sum(gaps[:2]), 4) + Fraction(gaps[2], 2)))
            for key, values in zip((ROW, GROUP), zip(*differences)):
                self.assertEqual(result[key], (min(values), max(values)))

    def test_invalid_truth_or_cost_or_roster_rejects(self):
        row = ('r', 0, 1, 0, 1)
        roster = [('r', 'g')]
        for fp, fn in ((0, 0), (1, -1), (False, 1), (float('inf'), 1)):
            with self.assertRaises(ValueError):
                bounds([row], expected_roster=roster, threshold=1, false_positive_cost=fp,
                       false_negative_cost=fn)
        good = dict(threshold=1, false_positive_cost=1, false_negative_cost=0)
        for rows, expected in (([row, row], roster), ([row], [*roster, ('q', 'h')]),
                               ([('r', 1, 0, 0, 1)], roster),
                               ([('r', 0, None, 0, 1)], roster), ([row], [('r', True)])):
            with self.assertRaises(ValueError):
                bounds(rows, expected_roster=expected, **good)
