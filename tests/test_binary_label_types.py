"""Binary classes are explicit numeric labels, not encoded elapsed times."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import GroupBenchClassifier
from groupbench.binary.data import validate_binary_target
from groupbench.binary.metrics import binary_metrics, checked_probabilities, grouped_binary_metrics


class BinaryLabelTypeTests(unittest.TestCase):
    def test_raw_timedelta_elements_fail_in_all_label_paths(self):
        p = np.array([0.0, 1.0])
        for unit in ("ns", "us", "D"):
            labels = [np.timedelta64(0, unit), np.timedelta64(1, unit)]
            for values in (labels, tuple(labels), [0, labels[1]]):
                calls = (
                    lambda: validate_binary_target(values, 2),
                    lambda: binary_metrics(values, p),
                    lambda: grouped_binary_metrics(values, p, ["a", "b"]),
                    lambda: checked_probabilities(np.column_stack((1 - p, p)), 2, values),
                )
                for i, invoke in enumerate(calls):
                    with self.subTest(unit=unit, type=type(values), api=i):
                        with self.assertRaisesRegex(ValueError, "numeric 0/1.*timedelta"):
                            invoke()

    def test_binary_fit_rejects_elapsed_ticks_before_creating_a_fit_plan(self):
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = [np.timedelta64(i, "ns") for i in (0, 1, 0, 1)]
        groups = ["a", "a", "b", "b"]
        estimator = GroupBenchClassifier(baselines=("dummy",), inner_splits=2)
        with patch("groupbench.binary.model.binary_selection_plan",
                   side_effect=AssertionError("Label validation must precede planning")) as plan:
            with self.assertRaisesRegex(ValueError, "numeric 0/1.*timedelta"):
                estimator.fit(X, y, groups=groups)
            plan.assert_not_called()
            self.assertFalse(hasattr(estimator, "best_estimator_"))
        # Caller-declared, explicit conversion remains accepted; the helper is
        # not responsible for how already-numeric class labels were obtained.
        explicit_labels = [int(element / np.timedelta64(1, "ns")) for element in y]
        self.assertIs(estimator.fit(X, explicit_labels, groups=groups), estimator)
        self.assertEqual(estimator.predict(X).tolist(), [1, 1, 1, 1])

    def test_nonbinary_duration_never_reaches_exact_numeric_range_checks(self):
        for value in (np.timedelta64(2, "ns"), np.timedelta64("NaT")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "numeric 0/1.*timedelta"):
                validate_binary_target([0, value], 2, require_both=False)

    def test_explicit_numpy_numeric_labels_remain_valid(self):
        self.assertEqual(validate_binary_target([np.int8(0), np.float32(1)], 2).tolist(), [0, 1])
        self.assertEqual(validate_binary_target([np.uint64(0)], 1, require_both=False).tolist(), [0])
