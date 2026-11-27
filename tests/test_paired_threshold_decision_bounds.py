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

    def test_total_event_count_tightens_independent_bounds(self):
        rows = [('a', 0, 1, 0, 1), ('b', 0, 1, 0, 1)]
        roster = [('a', 'x'), ('b', 'y')]
        policy = dict(threshold=1, false_positive_cost=2, false_negative_cost=5)
        original = bounds(rows, expected_roster=roster, **policy)
        self.assertEqual(original[ROW], (-5, 2))
        self.assertNotIn('known_total_events', original)
        for n, answered in ((0, 2), (1, Fraction(-3, 2)), (2, -5)):
            result = bounds(rows, expected_roster=roster, known_total_events=n, **policy)
            self.assertEqual(result['known_total_events'], n)
            self.assertEqual(result[ROW], (answered, answered))
            self.assertEqual(result[GROUP], result[ROW])

    def test_agreed_actions_can_absorb_an_unresolved_event(self):
        rows = [('same action', -1, 0, 0, 0), ('different', -1, 0, -1, 0),
                ('definite event', 0, 0, -1, 0)]
        policy = dict(threshold=0, false_positive_cost=2, false_negative_cost=5)
        roster = [('same action', 'large'), ('different', 'large'), ('definite event', 'small')]
        result = bounds(rows, expected_roster=roster, known_total_events=2, **policy)
        self.assertEqual(result[ROW], (Fraction(-10, 3), -1))
        self.assertEqual(result[GROUP], (Fraction(-15, 4), -2))
        for invalid in (0, 4, True, 2., '2'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bounds(rows, expected_roster=roster, known_total_events=invalid, **policy)

    def test_count_constrained_bounds_match_complete_outcome_enumeration(self):
        roster = [('a', 'large'), ('b', 'large'), ('c', 'small')]
        policy = dict(threshold=0, false_positive_cost=Fraction(3, 2),
                      false_negative_cost=4)
        # A zero, +1 or -1 predicted action difference; all truths ambiguous.
        choices = ((0, 0), (-1, 0), (0, -1))
        for arms in product(choices, repeat=3):
            rows = [(key, -1, 0, a, b) for (key, _), (a, b) in zip(roster, arms)]
            for n in range(4):
                expected = []
                for actuals in product((False, True), repeat=3):
                    if sum(actuals) != n:
                        continue
                    def cost(actual, positive):
                        return (policy['false_positive_cost'] if not actual and positive
                                else policy['false_negative_cost'] if actual and not positive else 0)
                    deltas = [cost(y, b >= 0) - cost(y, a >= 0) for y, (a, b) in zip(actuals, arms)]
                    expected.append((Fraction(sum(deltas), 3),
                                     Fraction(sum(deltas[:2]), 4) + Fraction(deltas[2], 2)))
                report = bounds(rows[::-1], expected_roster=roster, known_total_events=n,
                                **(policy | {'false_positive_cost': 1.5}))
                for key, candidates in zip((ROW, GROUP), zip(*expected)):
                    self.assertEqual(report[key], (min(candidates), max(candidates)))
