"""Synthetic regressions for precision checks on model feature inputs."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import GroupBenchRegressor, plan_benchmark, plan_prediction
from groupbench.data import validate_features, validate_prediction_features
from groupbench.binary import plan_binary_benchmark, plan_binary_prediction


class FeatureNumericPrecisionTests(unittest.TestCase):
    def test_integer_columns_reject_loss_before_float_conversion(self):
        for dtype, values in (("int64", [2**53, 2**53 + 1]),
                              ("uint64", [2**64 - 1, 0]),
                              ("Int64", [2**53 + 1, None]),
                              ("UInt64", [2**64 - 1, None])):
            frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
            before = frame.copy(deep=True)
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "representable in float64"):
                validate_features(frame)
            pd.testing.assert_frame_equal(frame, before)

    def test_exact_integers_and_nullable_missing_values_are_supported(self):
        for dtype, values in (("int64", [2**53, 2**53 + 2, -(2**63)]),
                              ("uint64", [2**63, 2**53 + 2, 0]),
                              ("Int64", [2**53, None, 2**53 + 2])):
            with self.subTest(dtype=dtype):
                result = validate_features(pd.DataFrame({"x": pd.Series(values, dtype=dtype)}))
                self.assertEqual(str(result.x.dtype), "float64")
                for source, converted in zip(values, result.x):
                    if source is None:
                        self.assertTrue(np.isnan(converted))
                    else:
                        self.assertEqual(int(converted), source)

    def test_wider_float_rejected_and_ordinary_float_supported(self):
        if np.dtype(np.longdouble).itemsize > 8:
            frame = pd.DataFrame({"x": np.array([1, 2], dtype=np.longdouble)})
            with self.assertRaisesRegex(ValueError, "wider than float64"):
                validate_features(frame)
        for dtype in (np.float16, np.float32, np.float64, "Float64"):
            frame = pd.DataFrame({"x": pd.Series([0.5, None], dtype=dtype)})
            result = validate_features(frame)
            self.assertEqual(result.x.iloc[0], 0.5)
            self.assertTrue(np.isnan(result.x.iloc[1]))

    def test_sparse_numeric_features_keep_existing_dtype_support(self):
        for dtype in (pd.SparseDtype("float64"), pd.SparseDtype("float32"), pd.SparseDtype("int64")):
            with self.subTest(dtype=dtype):
                frame = pd.DataFrame({"x": pd.Series([0, 1], dtype=dtype)})
                result = validate_features(frame)
                np.testing.assert_array_equal(result.x.to_numpy(), [0.0, 1.0])
        bad = pd.DataFrame({"x": pd.Series([2**53, 2**53 + 1], dtype=pd.SparseDtype("int64"))})
        with self.assertRaisesRegex(ValueError, "representable in float64"):
            validate_features(bad)

    def test_training_and_prediction_reject_before_model_work(self):
        frame = pd.DataFrame({"lot": np.repeat(list("abcdef"), 2),
                              "x": np.tile(np.array([2**53, 2**53 + 1], dtype=np.int64), 6),
                              "y": np.arange(12, dtype=float)})
        with patch("groupbench.splitting.pd.factorize", side_effect=AssertionError("unexpected split")) as split:
            with self.assertRaisesRegex(ValueError, "representable in float64"):
                plan_benchmark(frame, target="y", group="lot", outer_splits=3, inner_splits=2, baselines=("dummy",))
            split.assert_not_called()
        X = frame[["x"]]
        with self.assertRaisesRegex(ValueError, "representable in float64"):
            GroupBenchRegressor(inner_splits=2, baselines=("dummy",)).fit(X, frame.y, groups=frame.lot)
        good = frame.copy()
        good["x"] = np.arange(12, dtype=float)
        with self.assertRaisesRegex(ValueError, "representable in float64"):
            plan_prediction(good, X, target="y", group="lot", inner_splits=2, baselines=("dummy",))
        with self.assertRaisesRegex(ValueError, "representable in float64"):
            validate_prediction_features(X, ["x"], ["float64"])

    def test_binary_paths_and_object_categorical_contract(self):
        frame = pd.DataFrame({"lot": np.repeat(list("abcdef"), 2),
                              "x": np.tile(np.array([2**53, 2**53 + 1], dtype=np.int64), 6),
                              "y": np.tile([0, 1], 6)})
        with self.assertRaisesRegex(ValueError, "representable in float64"):
            plan_binary_benchmark(frame, target="y", group="lot", outer_splits=3, inner_splits=2)
        good = frame.copy()
        good["x"] = np.arange(12, dtype=float)
        with self.assertRaisesRegex(ValueError, "representable in float64"):
            plan_binary_prediction(good, frame[["x"]], target="y", group="lot", inner_splits=2)
        source = pd.DataFrame({"x": pd.Series([2**53, 2**53 + 1], dtype=object)})
        categorical = validate_features(source)
        self.assertEqual(categorical.x.tolist(), [str(2**53), str(2**53 + 1)])
        self.assertEqual(str(categorical.x.dtype), "object")

    def test_upstream_float_rounding_cannot_be_recovered(self):
        already_rounded = np.array([2**53, 2**53 + 1], dtype=float)
        self.assertEqual(already_rounded[0], already_rounded[1])
        np.testing.assert_array_equal(validate_features(pd.DataFrame({"x": already_rounded})).x, already_rounded)


if __name__ == "__main__":
    unittest.main()
