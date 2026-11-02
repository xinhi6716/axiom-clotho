"""Synthetic no-fit checks for deterministic binary arithmetic error policy."""
import unittest
import warnings

import numpy as np

from groupbench.binary.metrics import binary_metrics, grouped_binary_metrics


class BinaryMetricErrorStateTests(unittest.TestCase):
    def test_subnormal_probability_diagnostic_keeps_finite_scores_across_modes(self):
        tiny = np.nextafter(0., 1.)
        for factor in (1, 3, 5, 7):
            values = np.array([factor * tiny, 0.])
            original = values.copy()
            with np.errstate(all='ignore'):
                expected = binary_metrics([1, 1], values)
            for mode in ('ignore', 'warn', 'raise', 'call', 'print', 'log'):
                with self.subTest(factor=factor, mode=mode), np.errstate(all=mode):
                    settings = np.geterr().copy()
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', RuntimeWarning)
                        self.assertEqual(binary_metrics([1, 1], values), expected)
                    self.assertEqual(np.geterr(), settings)
            np.testing.assert_array_equal(values, original)
        self.assertEqual(expected['brier'], 1.)
        self.assertGreater(expected['log_loss_clipped'], 0.)
        self.assertGreater(expected['mean_probability_1'], 0.)

    def test_grouped_single_class_and_mixed_class_scores_are_mode_independent(self):
        tiny = np.nextafter(0., 1.)
        y, p, g = [1, 1, 0, 1], [3 * tiny, 0., .25, .75], ['a', 'a', 'b', 'b']
        with np.errstate(all='ignore'):
            expected = grouped_binary_metrics(y, p, g)
        with np.errstate(all='raise'):
            actual = grouped_binary_metrics(y, p, g)
        self.assertEqual(actual, expected)
        self.assertIsNone(actual[1][0]['specificity_0'])
        self.assertIsNone(actual[1][0]['balanced_accuracy'])
        self.assertIsNotNone(actual[1][1]['balanced_accuracy'])

    def test_positive_loss_rounded_to_zero_still_rejects_and_restores_settings(self):
        initial = np.geterr().copy()
        for mode in ('ignore', 'warn', 'raise'):
            with np.errstate(all=mode):
                settings = np.geterr().copy()
                with self.assertRaisesRegex(ValueError, 'scoring resolution'):
                    binary_metrics([0], [1e-300])
                self.assertEqual(np.geterr(), settings)
        self.assertEqual(np.geterr(), initial)

    def test_ordinary_and_endpoint_scores_remain_exact(self):
        for y, p in (([0, 1], [0., 1.]), ([0, 1], [1., 0.]),
                     ([0, 1, 1], [.25, .5, .75])):
            with np.errstate(all='ignore'):
                expected = binary_metrics(y, p)
            with np.errstate(all='raise'):
                self.assertEqual(binary_metrics(y, p), expected)


if __name__ == '__main__':
    unittest.main()
