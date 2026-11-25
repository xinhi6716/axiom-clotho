"""Validation reuse must not change the declared probability scoring protocol."""

import unittest
from unittest.mock import patch

import numpy as np

from groupbench.binary import metrics


class GroupedBinaryScoringValidation(unittest.TestCase):
    def test_validate_once_but_preserve_exact_scores_and_order(self):
        labels = [1, 0, 0, 1, 1, 0, 1]
        p = [0., np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.), 1., 1e-160, .78]
        groups = ["lab-b", "lab-a", "lab-b", "lab-c", "lab-c", "lab-a", "lab-b"]
        expected = []
        for group in ("lab-b", "lab-a", "lab-c"):
            idx = [i for i, item in enumerate(groups) if item == group]
            expected.append({"group": group, **metrics.binary_metrics(
                [labels[i] for i in idx], [p[i] for i in idx])})
        with patch.object(metrics, "_probability_array", wraps=metrics._probability_array) as probabilities, patch.object(
            metrics, "validate_binary_target", wraps=metrics.validate_binary_target
        ) as target:
            overall, per_group = metrics.grouped_binary_metrics(labels, p, groups)
        self.assertEqual(probabilities.call_count, 1)
        self.assertEqual(target.call_count, 1)
        self.assertEqual(per_group, expected)
        self.assertEqual({key: overall[key] for key in metrics.binary_metrics(labels, p)},
                         metrics.binary_metrics(labels, p))
        self.assertEqual(overall["n_groups"], 3)
        self.assertEqual(overall["group_brier"], float(np.mean([g["brier"] for g in expected])))

    def test_unsupported_types_shape_and_positive_underflow_still_fail(self):
        for labels, p, groups, error in (
            ([0, 1], [0., True], ["a", "b"], "not Boolean"),
            ([0, True], [0., 1.], ["a", "b"], "not Boolean"),
            ([0, 1], [[0.], [1.]], ["a", "b"], "matching one-dimensional"),
            ([0, 1], [1e-170, 1.], ["a", "b"], "below float64 scoring resolution"),
            ([0, 1], np.ma.array([0., 1.], mask=[False, True]), ["a", "b"], "masked entries"),
        ):
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    metrics.grouped_binary_metrics(labels, p, groups)
