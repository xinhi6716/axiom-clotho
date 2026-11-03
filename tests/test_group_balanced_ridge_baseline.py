"""Tiny handwritten software fixtures, not scientific performance evidence."""

from collections import Counter
from datetime import datetime, timedelta
import math
import pickle
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import LinAlgWarning
from sklearn.base import clone, is_regressor
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import Ridge

from groupbench import baselines as module
from groupbench.baselines import GroupBalancedMedianRegressor, GroupBalancedRidgeRegressor
from groupbench.data import MAX_ROWS, validate_groups


X = np.asarray([[0.0], [1.0], [3.0], [8.0], [10.0]])
Y = np.asarray([0.0, 1.0, 4.0, 6.0, 12.0])
GROUPS = np.asarray(["a", "a", "a", "b", "c"])


def oracle(x, y, groups, alpha):
    counts = Counter(groups)
    weights = [len(y) / (len(counts) * counts[group]) for group in groups]
    xbar = math.fsum(w * value for w, value in zip(weights, x)) / math.fsum(weights)
    ybar = math.fsum(w * value for w, value in zip(weights, y)) / math.fsum(weights)
    beta = (math.fsum(w * (value - xbar) * (target - ybar)
                      for w, value, target in zip(weights, x, y)) /
            (math.fsum(w * (value - xbar) ** 2 for w, value in zip(weights, x)) + alpha))
    return beta, ybar - beta * xbar


def group_mae(y, prediction, groups):
    return math.fsum(math.fsum(abs(target - prediction) for target, label in zip(y, groups)
                              if label == group) / sum(label == group for label in groups)
                     for group in set(groups)) / len(set(groups))


class GroupBalancedRidgeTests(unittest.TestCase):
    def fit(self, X=X, y=Y, groups=GROUPS, alpha=1.0):
        return GroupBalancedRidgeRegressor(alpha=alpha).fit(X, y, groups=groups)

    def test_unequal_groups_independent_analytic_oracle_and_local_weights(self):
        for alpha in (0.1, 1.0, 10.0):
            for indices in (np.asarray([0, 1, 2, 3, 4]), np.asarray([0, 2, 3, 4])):
                x, y, groups = X[indices], Y[indices], GROUPS[indices]
                beta, intercept = oracle(x[:, 0], y, groups, alpha)
                model = self.fit(x, y, groups, alpha)
                np.testing.assert_allclose(model.coef_, [beta], rtol=1e-12, atol=1e-12)
                self.assertAlmostEqual(model.intercept_, intercept, places=12)
                np.testing.assert_allclose(model.predict(x), intercept + beta * x[:, 0], rtol=1e-12, atol=1e-12)
                self.assertEqual(model.fit_details_["training_rows"], len(indices))

    def test_weight_arithmetic_equal_totals_and_literal_ones(self):
        cases = [["a", "a", "a", "b", "c"], ["a", "a", "b", "b"],
                 ["a"] * 5, ["a", "b", "c", "d", "e"]]
        for groups in cases:
            normalized = validate_groups(groups, len(groups))
            weights, details = module._group_balanced_weights(normalized)
            counts = Counter(normalized)
            self.assertEqual(weights.dtype, np.dtype("float64"))
            self.assertTrue(np.all(weights > 0))
            self.assertTrue(math.isclose(math.fsum(weights), len(groups), rel_tol=module._WEIGHT_RTOL))
            for group in counts:
                total = math.fsum(weight for weight, label in zip(weights, normalized) if label == group)
                self.assertTrue(math.isclose(total, len(groups) / len(counts), rel_tol=module._WEIGHT_RTOL))
            if len(set(counts.values())) == 1:
                np.testing.assert_array_equal(weights, np.ones(len(groups)))
            self.assertEqual(details["min_row_weight"], len(groups) / (len(counts) * max(counts.values())))
            self.assertEqual(details["max_row_weight"], len(groups) / (len(counts) * min(counts.values())))

    def test_equal_group_dense_csr_csc_parity_all_grid_alphas(self):
        matrix = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [4.0, 1.0]])
        y = np.asarray([0.0, 2.0, 1.0, 6.0])
        for alpha in (0.1, 1.0, 10.0):
            for groups in (["a", "a", "b", "b"], ["one"] * 4, ["a", "b", "c", "d"]):
                for factory in (np.asarray, sparse.csr_matrix, sparse.csc_matrix):
                    values = factory(matrix)
                    actual = self.fit(values, y, groups, alpha)
                    expected = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True, tol=1e-4,
                                     max_iter=None, copy_X=True, positive=False).fit(values, y)
                    np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=1e-11, atol=1e-11)
                    self.assertAlmostEqual(actual.intercept_, expected.intercept_, places=11)
                    np.testing.assert_allclose(actual.predict(values), expected.predict(values), rtol=1e-11, atol=1e-11)

    def test_no_group_mae_guarantee_constant_counterexample(self):
        matrix, y, groups = np.zeros((5, 1)), [0, 0, 0, 0, 9], ["a", "a", "a", "b", "c"]
        balanced = self.fit(matrix, y, groups)
        row_mean = Ridge(solver="lsqr").fit(matrix, y).intercept_
        median = GroupBalancedMedianRegressor().fit(matrix, y, groups=groups).constant_
        self.assertAlmostEqual(balanced.intercept_, 3.0)
        self.assertAlmostEqual(row_mean, 9 / 5)
        self.assertEqual(median, 0)
        self.assertAlmostEqual(group_mae(y, balanced.intercept_, groups), 4.0)
        self.assertAlmostEqual(group_mae(y, row_mean, groups), 18 / 5)
        self.assertEqual(group_mae(y, median, groups), 3)

    def test_details_exact_keys_recipe_types_and_no_raw_retention(self):
        model = self.fit()
        details = model.fit_details_
        expected = {"implementation", "training_rows", "training_groups", "transformed_features",
                    "alpha", "solver", "tol", "max_iter", "fit_intercept", "copy_X", "positive",
                    "threads", "fitting_loss", "weight_rule", "normalization", "preprocessing_weighting",
                    "min_training_group_size", "max_training_group_size", "min_row_weight", "max_row_weight",
                    "sum_row_weights"}
        self.assertEqual(set(details), expected)
        self.assertEqual(details["implementation"], "group-balanced-ridge-v1")
        self.assertEqual(details["fitting_loss"], "weighted_squared_error")
        self.assertEqual(details["normalization"], "sum_training_rows")
        self.assertEqual(details["preprocessing_weighting"], "unweighted_training_rows")
        for name in ("training_rows", "training_groups", "transformed_features", "threads",
                     "min_training_group_size", "max_training_group_size"):
            self.assertIs(type(details[name]), int)
        for name in ("alpha", "tol", "min_row_weight", "max_row_weight", "sum_row_weights"):
            self.assertIs(type(details[name]), float)
        for name, expected_value in {"solver": "lsqr", "tol": 1e-4, "max_iter": None,
                                     "fit_intercept": True, "copy_X": True, "positive": False}.items():
            self.assertEqual(getattr(model.ridge_, name), expected_value)
        for state in (model.__dict__, model.ridge_.__dict__):
            for name in ("groups_", "y_", "X_", "sample_weight_", "weights_", "target_"):
                self.assertNotIn(name, state)
        with self.assertRaises(TypeError):
            model.predict([[1]], groups=["query_group"])

    def test_exact_numeric_and_text_group_identity(self):
        for groups, sizes in [([1, 1.0, 2], (1, 2)),
                              ([2**53, 2**53 + 1, 2**53], (1, 2)),
                              ([np.float32(0.1), 0.1, np.float32(0.1)], (1, 2)),
                              (["a", " a", "a "], (1, 1))]:
            model = self.fit([[0], [1], [2]], [0, 1, 3], groups)
            self.assertEqual((model.fit_details_["min_training_group_size"],
                              model.fit_details_["max_training_group_size"]), sizes)
        if np.finfo(np.longdouble).nmant > np.finfo(float).nmant:
            first = np.longdouble(1)
            second = np.nextafter(first, np.longdouble(2))
            self.assertEqual(self.fit([[0], [1]], [0, 1], [first, second]).n_groups_seen_, 2)

    def test_invalid_groups_and_caller_weights(self):
        invalid = [None, [], ["a"], [["a"], ["b"]], ["a", None], ["a", " "], ["a", "\x00"],
                   [1, "1"], [True, False], [np.bool_(False), np.bool_(True)], [0, np.inf], [{}, {}],
                   np.ma.array(["a", "b"], mask=[False, True])]
        for groups in invalid:
            with self.subTest(groups=repr(groups)), self.assertRaises((ValueError, TypeError)):
                self.fit([[0], [1]], [0, 1], groups)
        for weight in ([1, 1], [], False, np.nan):
            with self.assertRaisesRegex(ValueError, "sample_weight"):
                GroupBalancedRidgeRegressor().fit([[0], [1]], [0, 1], groups=["a", "b"], sample_weight=weight)

    def test_target_precision_temporal_complex_and_nonfinite_rejections(self):
        bad = [[2**53 + 1, 0],
               [datetime(2020, 1, 1), datetime(2020, 1, 2)], [timedelta(1), timedelta(2)],
               np.asarray([1, 2], dtype="datetime64[D]"), [1 + 0j, 2],
               np.asarray([np.complex128(1), 2], dtype=object), [0, np.inf], [0, np.nan], [[0], [1]]]
        if np.dtype(np.longdouble).itemsize > 8:
            bad.append(np.asarray([np.longdouble(1), np.longdouble(2)]))
        for y in bad:
            with self.subTest(y=repr(y)), self.assertRaises(ValueError):
                self.fit([[0], [1]], y, ["a", "b"])
        self.fit([[0], [1]], [2**53, 0], ["a", "b"])

    def test_alpha_validation_and_numerical_rescaling_failures(self):
        for alpha in (True, False, 0, -1, np.nan, np.inf, "1", 1j, 10**1000):
            with self.subTest(alpha=repr(alpha)), self.assertRaisesRegex(ValueError, "alpha"):
                self.fit(alpha=alpha)
        for matrix, y, alpha in (([[1e200], [0]], [0, 1], 1),
                                 ([[0], [1]], [0, 1e200], 1),
                                 ([[0], [1]], [0, 1], np.finfo(float).max),
                                 ([[0], [1]], [0, 1], np.nextafter(0.0, 1.0))):
            with self.assertRaisesRegex(ValueError, "rescale"):
                self.fit(matrix, y, ["a", "b"], alpha)

    def test_dense_boundaries_names_and_input_nonmutation(self):
        frame = pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 3.0]})
        before = frame.copy(deep=True)
        model = self.fit(frame, [0, 1], ["a", "b"])
        for values in (frame[["b", "a"]], frame.rename(columns={"a": "c"}), frame.to_numpy()):
            with self.assertRaisesRegex(ValueError, "names and order"):
                model.predict(values)
        pd.testing.assert_frame_equal(frame, before)
        for columns in (["a", "a"], [0, 1]):
            with self.assertRaisesRegex(ValueError, "distinct strings"):
                self.fit(frame.set_axis(columns, axis=1), [0, 1], ["a", "b"])
        for matrix in (np.zeros((0, 1)), np.zeros((2, 0)), np.zeros((2, 6401)),
                       np.zeros((20001, 1)), np.zeros((2, 1, 1)), [[np.inf], [0]],
                       [[1j], [0]], np.asarray([[1], [2]], dtype="datetime64[D]"),
                       [[datetime(2020, 1, 1)], [datetime(2020, 1, 2)]]):
            with self.subTest(shape=np.shape(matrix)), self.assertRaises(ValueError):
                self.fit(matrix, [0, 1], ["a", "b"])
        model = self.fit(np.zeros((1, 6400)), [1], ["a"])
        self.assertEqual(model.n_features_in_, 6400)
        np.testing.assert_array_equal(model.predict(np.zeros((1, 6400))), [1])
        with self.assertRaisesRegex(ValueError, "feature count"):
            model.predict([[0]])

    def test_maximum_rows_weight_bounds_and_over_limit_rejection(self):
        groups = ["a"] * (MAX_ROWS - 1) + ["b"]
        model = self.fit(np.zeros((MAX_ROWS, 1)), np.zeros(MAX_ROWS), groups)
        self.assertEqual(model.n_samples_seen_, MAX_ROWS)
        self.assertEqual(model.fit_details_["max_row_weight"], MAX_ROWS / 2)
        self.assertEqual(model.fit_details_["sum_row_weights"], MAX_ROWS)
        with self.assertRaisesRegex(ValueError, "rows"):
            model.predict(np.zeros((MAX_ROWS + 1, 1)))

    def test_sparse_canonical_copy_no_densification_and_bad_structure(self):
        for factory in (sparse.csr_matrix, sparse.csc_matrix):
            values = factory([[1.0, 0.0], [0.0, 2.0]])
            # Two unsorted duplicate entries per major axis, equivalent to a
            # valid tiny matrix after canonicalization.
            values.data = np.asarray([0.25, 0.75, 1.0, 1.0])
            values.indices = np.asarray([0, 0, 1, 1], dtype=np.int32)
            values.indptr = np.asarray([0, 2, 4], dtype=np.int32)
            originals = [getattr(values, field).copy() for field in ("data", "indices", "indptr")]
            with patch.object(factory, "toarray", side_effect=AssertionError("densified")):
                model = self.fit(values, [0, 1], ["a", "b"])
                self.assertTrue(np.isfinite(model.predict(values)).all())
            for name, original in zip(("data", "indices", "indptr"), originals):
                np.testing.assert_array_equal(getattr(values, name), original)
            for field, bad in (("indices", np.asarray([0, 2])), ("indices", np.asarray([-1, 1])),
                               ("indices", np.asarray([0.0, 1.0])), ("indptr", np.asarray([0, 2, 1])),
                               ("indptr", np.asarray([1, 1, 2])), ("indptr", np.asarray([0, 2])),
                               ("data", np.asarray([1.0]))):
                broken = factory([[1.0, 0.0], [0.0, 2.0]])
                setattr(broken, field, bad)
                with self.subTest(format=factory.__name__, field=field), self.assertRaisesRegex(ValueError, "structure"):
                    self.fit(broken, [0, 1], ["a", "b"])
        for factory in (sparse.coo_matrix, sparse.lil_matrix, sparse.dok_matrix, sparse.bsr_matrix, sparse.dia_matrix):
            with self.assertRaisesRegex(ValueError, "CSR or CSC"):
                self.fit(factory([[0], [1]]), [0, 1], ["a", "b"])

    def test_nested_masks_cycles_depth_and_work_are_rejected(self):
        masked = np.ma.array(99.0, mask=True)
        features = [[masked], [1.0]]
        model = self.fit()
        for X_bad in (features, pd.DataFrame(features), np.asarray(features, dtype=object)):
            with self.assertRaisesRegex(ValueError, "masked"):
                self.fit(X_bad, [0, 1], ["a", "b"])
            with self.assertRaisesRegex(ValueError, "masked"):
                model.predict(X_bad)
        for parameter in ("y", "groups"):
            for values in ([masked, 1], np.asarray([masked, 1], dtype=object), pd.Series([masked, 1])):
                kwargs = dict(X=[[0], [1]], y=[0, 1], groups=["a", "b"])
                kwargs[parameter] = values
                with self.assertRaisesRegex(ValueError, "masked"):
                    self.fit(**kwargs)
        cycle = []
        cycle.append(cycle)
        deep = [1]
        for _ in range(65):
            deep = [deep]
        for values in (cycle, deep):
            with self.assertRaisesRegex(ValueError, "cyclic|depth"):
                model.predict(values)
        with patch.object(module, "_MAX_MASK_VISITS", 3):
            with self.assertRaisesRegex(ValueError, "inspection"):
                model.predict([[0], [1]])

    def test_sparse_masks_are_rejected_before_conversion_no_mask_is_valid(self):
        for factory in (sparse.csr_matrix, sparse.csc_matrix):
            for field in ("data", "indices", "indptr"):
                values = factory([[0.0], [1.0]])
                original = getattr(values, field).copy()
                setattr(values, field, np.ma.array(original, mask=False))
                self.fit(values, [0, 1], ["a", "b"])
                getattr(values, field).mask[0] = True
                with patch.object(module, "check_array", side_effect=AssertionError("conversion")):
                    with self.assertRaisesRegex(ValueError, "masked"):
                        self.fit(values, [0, 1], ["a", "b"])
                np.testing.assert_array_equal(getattr(values, field).data, original)

    def test_failed_fits_and_numeric_warnings_are_atomic_and_recover(self):
        initial = GroupBalancedRidgeRegressor()
        with self.assertRaises(ValueError):
            initial.fit([[0]], [np.nan], groups=["a"])
        with self.assertRaises(NotFittedError):
            initial.predict([[0]])
        model = self.fit(pd.DataFrame(X, columns=["x"]))
        original = pickle.dumps(model)
        for error in (RuntimeWarning("overflow"), LinAlgWarning("unstable"),
                      ConvergenceWarning("not converged"), FloatingPointError("overflow"),
                      np.linalg.LinAlgError("singular")):
            with patch.object(module.Ridge, "fit", side_effect=error):
                with self.assertRaisesRegex(ValueError, "rescale"):
                    model.fit(X, Y, groups=GROUPS)
            self.assertEqual(pickle.dumps(model), original)
        def invalid_fit(estimator, *args, **kwargs):
            estimator.coef_, estimator.intercept_ = np.asarray([np.nan]), 0.0
            return estimator
        with patch.object(module.Ridge, "fit", invalid_fit):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                model.fit(X, Y, groups=GROUPS)
        self.assertEqual(pickle.dumps(model), original)
        model.fit(X, Y, groups=GROUPS)
        self.assertFalse(hasattr(model, "feature_names_in_"))
        self.assertTrue(np.isfinite(model.predict(X)).all())

    def test_real_warning_filter_and_small_coefficient_prediction(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = self.fit([[0], [1], [2]], [0, 2, 1], ["a", "a", "b"])
            self.assertLess(abs(model.coef_[0]), 1)
            self.assertTrue(np.isfinite(model.predict([[0], [1], [2]])).all())
        prior = pickle.dumps(model)
        def warn_during_fit(*args, **kwargs):
            warnings.warn("invalid intermediate", RuntimeWarning)
            raise AssertionError("warning was not caught")
        with patch.object(module.Ridge, "fit", warn_during_fit):
            with self.assertRaisesRegex(ValueError, "rescale"):
                model.fit(X, Y, groups=GROUPS)
        self.assertEqual(pickle.dumps(model), prior)

    def test_prediction_overflow_and_bad_outputs_never_change_state(self):
        model = self.fit()
        original = pickle.dumps(model)
        with self.assertRaisesRegex(ValueError, "rescale"):
            model.predict([[1e200]])
        for result in (np.asarray([np.inf]), np.zeros((1, 1))):
            with patch.object(module.Ridge, "predict", return_value=result):
                with self.assertRaisesRegex(ValueError, "rescale"):
                    model.predict([[0]])
        with patch.object(module.Ridge, "predict", side_effect=RuntimeWarning("overflow")):
            with self.assertRaisesRegex(ValueError, "rescale"):
                model.predict([[0]])
        self.assertEqual(pickle.dumps(model), original)
        model.ridge_.coef_[:] = np.finfo(float).max
        with self.assertRaisesRegex(ValueError, "range"):
            model.predict([[2]])

    def test_clone_score_and_handwritten_permutation(self):
        model = self.fit()
        self.assertTrue(is_regressor(model))
        self.assertEqual(model.get_params(), {"alpha": 1.0})
        with self.assertRaises(NotFittedError):
            clone(model).predict([[0]])
        permuted = np.asarray([4, 1, 3, 0, 2])
        reordered = self.fit(X[permuted], Y[permuted], GROUPS[permuted])
        np.testing.assert_allclose(model.predict(X), reordered.predict(X), rtol=1e-12, atol=1e-12)
        expected = 1 - np.sum((Y - model.predict(X)) ** 2) / np.sum((Y - np.mean(Y)) ** 2)
        self.assertAlmostEqual(model.score(X, Y), expected)


if __name__ == "__main__":
    unittest.main()
