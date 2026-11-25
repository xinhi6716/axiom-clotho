"""Direct balanced-Ridge feature precision, before sklearn's float conversion."""

import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from groupbench.baselines import GroupBalancedRidgeRegressor


class GroupBalancedRidgeFeaturePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.y = [0., 10., 0., 10.]
        self.groups = ["a", "a", "b", "b"]
        self.training = np.array([[0.], [1.], [0.], [1.]])

    def test_adjacent_integers_cannot_collapse_in_direct_fit_or_predict(self):
        hi = 2**53 + 1
        values = np.array([[hi - 1], [hi], [hi - 1], [hi]], dtype=np.int64)
        model = GroupBalancedRidgeRegressor(alpha=.1).fit(self.training, self.y, groups=self.groups)
        original = model.predict(self.training)
        self.assertGreater(original[1] - original[0], 9.0)
        sparse_object = sparse.csr_matrix(values)
        # SciPy does not construct object sparse matrices directly, but some
        # callers assign raw data buffers. Inspect before the existing cast.
        sparse_object.data = sparse_object.data.astype(object)
        forms = (values, values.astype(np.uint64), values.tolist(), values.astype(object),
                 sparse.csr_matrix(values), sparse.csc_matrix(values), sparse_object)
        for X in forms:
            with self.subTest(container=type(X).__name__, dtype=getattr(X, "dtype", None)):
                with self.assertRaisesRegex(ValueError, "exactly representable in float64"):
                    model.fit(X, self.y, groups=self.groups)
                with self.assertRaisesRegex(ValueError, "exactly representable in float64"):
                    model.predict(X)
                np.testing.assert_array_equal(model.predict(self.training), original)

    def test_named_and_nullable_or_mixed_columns_are_checked_before_promotion(self):
        hi = 2**53 + 1
        y = self.y
        for dtype in ("int64", "Int64", "uint64", "UInt64", object):
            with self.subTest(dtype=dtype):
                exact = pd.DataFrame({"time_tick": pd.Series([0, 1, 0, 1], dtype=dtype),
                                      "gate_V": [0., .25, 0., .25]})
                candidate = pd.DataFrame({"time_tick": pd.Series([hi-1, hi, hi-1, hi], dtype=dtype),
                                          "gate_V": [0., .25, 0., .25]})
                model = GroupBalancedRidgeRegressor().fit(exact, y, groups=self.groups)
                with self.assertRaisesRegex(ValueError, "exactly representable in float64"):
                    model.predict(candidate)
                with self.assertRaisesRegex(ValueError, "exactly representable in float64"):
                    model.fit(candidate, y, groups=self.groups)

    def test_exact_large_integer_narrow_floats_and_sparse_values_keep_parity(self):
        original = np.array([[2**53], [2**53 + 2], [2**53], [2**53 + 2]], dtype=np.int64)
        for dtype in (np.int64, np.uint64, np.float16, np.float32, np.float64):
            numeric = original.astype(dtype) if dtype in (np.int64, np.uint64) else self.training.astype(dtype)
            marshals = (np.asarray,) if dtype == np.float16 else (np.asarray, sparse.csr_matrix, sparse.csc_matrix)
            for marshal in marshals:
                with self.subTest(dtype=dtype, container=marshal):
                    supplied = marshal(numeric)
                    f64 = marshal(numeric.astype(np.float64))
                    actual = GroupBalancedRidgeRegressor().fit(supplied, self.y, groups=self.groups)
                    reference = GroupBalancedRidgeRegressor().fit(f64, self.y, groups=self.groups)
                    self.assertEqual(actual.fit_details_, reference.fit_details_)
                    np.testing.assert_array_equal(actual.predict(supplied), reference.predict(f64))

    @unittest.skipUnless(np.dtype(np.longdouble).itemsize > 8, "no wider float dtype")
    def test_wider_numpy_float_rejects_before_fit_or_prediction(self):
        extended = self.training.astype(np.longdouble)
        with self.assertRaisesRegex(ValueError, "wider than float64"):
            GroupBalancedRidgeRegressor().fit(extended, self.y, groups=self.groups)
        trained = GroupBalancedRidgeRegressor().fit(self.training, self.y, groups=self.groups)
        for X in (extended, extended.astype(object), extended.tolist(), sparse.csr_matrix(extended)):
            with self.subTest(container=type(X).__name__):
                with self.assertRaisesRegex(ValueError, "wider than float64"):
                    trained.predict(X)
