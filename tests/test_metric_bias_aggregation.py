"""Characterize, rather than certify, historical float64 bias aggregation.

These assertions describe the historical implementation and should change with
an explicitly versioned replacement, not be treated as an accuracy guarantee.
"""
from fractions import Fraction
from itertools import permutations
import unittest

import numpy as np

from groupbench.metrics import grouped_metrics, regression_metrics


def legacy_bias(y, pred):
    """The historical NumPy reduction, on this platform and NumPy version."""
    return float(np.mean(np.asarray(pred, dtype=np.float64)
                         - np.asarray(y, dtype=np.float64)))


class MetricBiasCancellationCharacterization(unittest.TestCase):
    def test_permutations_follow_historical_reduction_and_have_same_exact_mean(self):
        pred = [1e16, 1.0, -1e16, -0.5]
        for order in permutations(pred):
            with self.subTest(order=order):
                y = [0.] * len(order)
                self.assertEqual(regression_metrics(y, order)['bias'], legacy_bias(y, order))
                self.assertEqual(sum(map(Fraction.from_float, order)) / len(order), Fraction(1, 8))

    def test_pooled_and_per_group_use_historical_reduction(self):
        positive = [1e16, 1., -1e16, -0.5]
        for order in permutations(positive):
            with self.subTest(order=order):
                other = [-value for value in order]
                pred, y = list(order) + other, [0.] * 8
                overall, groups = grouped_metrics(y, pred, ['a'] * 4 + ['b'] * 4)
                self.assertEqual(overall['bias'], legacy_bias(y, pred))
                self.assertEqual(groups[0]['bias'], legacy_bias(y[:4], pred[:4]))
                self.assertEqual(groups[1]['bias'], legacy_bias(y[4:], pred[4:]))
                self.assertEqual(sum(map(Fraction.from_float, order)) / 4, Fraction(1, 8))
                self.assertEqual(sum(map(Fraction.from_float, other)) / 4, Fraction(-1, 8))


if __name__ == '__main__':
    unittest.main()
