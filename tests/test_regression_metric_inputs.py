"""Synthetic, no-fit regression-score input precision and rejection checks."""

from decimal import Decimal
import copy
import unittest
import warnings

import numpy as np
import pandas as pd

from groupbench.metrics import grouped_metrics, regression_metrics


class RegressionMetricInputsTests(unittest.TestCase):
    def assert_rejected(self, y, pred):
        for scorer in (regression_metrics, grouped_metrics):
            with self.subTest(scorer=scorer.__name__), warnings.catch_warnings():
                warnings.simplefilter("error")
                with self.assertRaises(ValueError):
                    if scorer is grouped_metrics:
                        scorer(y, pred, ["a", "b"])
                    else:
                        scorer(y, pred)

    def test_complex_values_are_not_silently_projected_onto_real_axis(self):
        for values in ([1 + 0j, 2 + 1j], np.array([1 + 0j, 2 + 1j]),
                       np.array([1 + 0j, 2 + 0j])):
            self.assert_rejected(values, [1, 2])
            self.assert_rejected([1, 2], values)

    def test_integer_differences_cannot_disappear_before_scoring(self):
        values = [2**53, 2**53 + 1]
        for container in (list, tuple, np.array, lambda x: np.array(x, dtype=object), pd.Series):
            self.assert_rejected(container(values), [2**53, 2**53])
            self.assert_rejected([2**53, 2**53], container(values))

    def test_mixed_float_integer_sequence_is_validated_before_numpy_promotion(self):
        self.assert_rejected([2**53 + 1, 0.5], [2**53, 0.5])
        self.assert_rejected([0.5, 2**53], [0.5, 2**53 + 1])
        self.assert_rejected([np.uint64(2**64 - 1), 0.5], [1, 0.5])

    def test_large_but_exact_integer_values_remain_supported(self):
        for values in ([2**60, 2**60 + 256], [-2**60, -2**60 + 128],
                       np.array([2**63, 2**63 + 2048], dtype=np.uint64),
                       [2**100, 2**100 + 2**48]):
            result = regression_metrics(values, values)
            self.assertEqual(result["mae"], 0)
            self.assertEqual(result["rmse"], 0)
            self.assertEqual(result["r2"], 1)
            self.assertEqual(grouped_metrics(values, values, ["a", "b"])[0]["group_mae"], 0)

    def test_nonrepresentable_unsigned_signed_and_python_integers_reject(self):
        for values in (np.array([0, np.iinfo(np.int64).max], dtype=np.int64),
                       np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64),
                       [0, 10**400], [0, -(2**53 + 1)]):
            self.assert_rejected(values, [0, 1])

    def test_extended_floats_reject_even_when_current_values_would_fit(self):
        if np.dtype(np.longdouble).itemsize <= 8:
            self.skipTest("longdouble is not wider than float64 on this platform")
        for values in (np.array([1, 2], dtype=np.longdouble),
                       [np.longdouble(1), np.longdouble(2)],
                       np.array([1, 1 + np.longdouble(2)**-60], dtype=np.longdouble)):
            self.assert_rejected(values, [1, 2])
            self.assert_rejected([1, 2], values)

    def test_strings_booleans_and_other_coercible_non_numeric_values_reject(self):
        for values in (["1", "2"], [False, 1], np.array([True, False]),
                       [Decimal("1"), Decimal("2")], [None, 2], [pd.NA, 2],
                       np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")):
            self.assert_rejected(values, [1, 2])
            self.assert_rejected([1, 2], values)

    def test_masked_arrays_do_not_turn_hidden_entries_into_observations(self):
        for mask in ([False, True], [False, False]):
            values = np.ma.array([1.0, 2.0], mask=mask)
            self.assert_rejected(values, [1, 2])
            self.assert_rejected([1, 2], values)

    def test_shapes_empty_and_nonfinite_fail_with_valueerror_before_group_slicing(self):
        for y, pred in ((1, 1), ([1, 2], [1]), ([1], [1, 2]),
                        ([], []), ([[1], [2]], [1, 2]), ([1, 2], [[1], [2]]),
                        ([1, 2], [[1], [2, 3]]), ([np.nan, 2], [1, 2]),
                        ([1, 2], [1, np.inf]), ([1, -np.inf], [1, 2])):
            self.assert_rejected(y, pred)

    def test_supported_containers_and_float_widths_agree_without_mutation(self):
        y, pred = [1, 2, 4, 8], [2, 2, 2, 10]
        expected = regression_metrics(y, pred)
        expected_grouped = grouped_metrics(y, pred, ["a", "a", "a", "b"])
        containers = (list, tuple, np.array, lambda x: np.array(x, dtype=object),
                      lambda x: np.array(x, dtype=np.float16),
                      lambda x: np.array(x, dtype=np.float32),
                      lambda x: np.array(x, dtype=np.float64), pd.Series,
                      lambda x: pd.Series(x, dtype="Int64"))
        for container in containers:
            actual_y, actual_pred = container(y), container(pred)
            before_y, before_pred = copy.deepcopy(actual_y), copy.deepcopy(actual_pred)
            with self.subTest(container=repr(container)):
                self.assertEqual(regression_metrics(actual_y, actual_pred), expected)
                self.assertEqual(grouped_metrics(actual_y, actual_pred, ["a", "a", "a", "b"]), expected_grouped)
                np.testing.assert_array_equal(actual_y, before_y)
                np.testing.assert_array_equal(actual_pred, before_pred)
        self.assertEqual(expected["mae"], 1.25)
        self.assertEqual(expected_grouped[0]["group_mae"], 1.5)

    def test_error_does_not_echo_invalid_source_value(self):
        for scorer in (regression_metrics, grouped_metrics):
            with self.assertRaises(ValueError) as caught:
                if scorer is grouped_metrics:
                    scorer(["PRIVATE_TARGET_SECRET", 2], [1, 2], ["a", "b"])
                else:
                    scorer([1, 2], ["PRIVATE_PREDICTION_SECRET", 2])
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assertNotIn("SECRET", str(caught.exception))

    def test_already_rounded_caller_values_are_scored_as_supplied(self):
        # The scorer cannot reconstruct values a caller already rounded away.
        rounded = np.asarray([2**53, 2**53 + 1], dtype=np.float64)
        self.assertEqual(rounded[0], rounded[1])
        result = regression_metrics(rounded, rounded)
        self.assertEqual(result["mae"], 0)
        self.assertIsNone(result["r2"])


if __name__ == "__main__":
    unittest.main()
