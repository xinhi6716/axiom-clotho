"""Anonymous software fixtures; no research-data or performance evaluation."""

from copy import deepcopy
from datetime import datetime, timedelta
import json
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import LinAlgWarning
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from groupbench.data import validate_features, validate_target
from groupbench.kernel import NystroemRidgeRegressor
import groupbench.kernel as kernel
from groupbench.model import Candidate, candidates_for, make_pipeline
from groupbench.splitting import GroupKFold


class KernelRegressionTests(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack([np.linspace(-1, 1, 12), np.arange(12) % 3])
        self.y = np.arange(12, dtype=float) / 3

    def fresh(self, **kwargs):
        return NystroemRidgeRegressor(rank_cap=4, **kwargs)

    def assert_unfitted(self, model):
        self.assertFalse(any(name.endswith("_") for name in vars(model)))
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def snapshot(self, model):
        return ({name: value for name, value in vars(model).items() if name.endswith("_")},
                deepcopy(model.fit_details_), model.predict(self.X))

    def assert_preserved(self, model, before):
        state, details, prediction = before
        actual = {name: value for name, value in vars(model).items() if name.endswith("_")}
        self.assertEqual(set(actual), set(state))
        for name, value in state.items():
            self.assertIs(actual[name], value, name)
        self.assertEqual(model.fit_details_, details)
        np.testing.assert_array_equal(model.predict(self.X), prediction)

    def test_dense_and_csr_match_explicit_seeded_reference(self):
        for factory in (np.asarray, sparse.csr_matrix):
            with self.subTest(representation=factory.__name__), threadpool_limits(limits=1):
                X = factory(self.X)
                expected_map = Nystroem(kernel="rbf", gamma=0.5, n_components=4,
                                        random_state=7, n_jobs=1).fit(X)
                expected = Ridge(alpha=0.1, solver="svd", fit_intercept=True)
                expected.fit(expected_map.transform(X), self.y)
                actual = self.fresh(alpha=0.1, random_state=7).fit(X, self.y)
                np.testing.assert_array_equal(actual.predict(X), expected.predict(expected_map.transform(X)))
                np.testing.assert_array_equal(actual.mapper_.component_indices_, expected_map.component_indices_)
                self.assertEqual(actual.fit_details_["solver"], "svd")
                self.assertEqual(actual.fit_details_["gamma"], 0.5)
                self.assertEqual(actual.fit_details_["training_map_bytes"], 12 * 4 * 8)
                self.assertEqual(actual.mapper_.n_jobs, 1)
                self.assertEqual(sparse.issparse(actual.mapper_.components_), sparse.issparse(X))
                json.dumps(actual.fit_details_, allow_nan=False)

    def test_tiny_and_capped_folds_have_explicit_component_counts_without_warning(self):
        for rows in (1, 2, 63, 64, 65):
            with self.subTest(rows=rows), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                X = np.column_stack([np.arange(rows, dtype=float), np.ones(rows)])
                model = NystroemRidgeRegressor().fit(X, np.ones(rows))
                self.assertEqual(model.effective_n_components_, min(64, rows))
                self.assertEqual(model.mapper_.n_components, min(64, rows))
                self.assertEqual(model.mapper_.normalization_.shape, (min(64, rows),) * 2)
                np.testing.assert_allclose(model.predict(X[:1]), [1.0], rtol=0, atol=1e-12)
                np.testing.assert_allclose(model.predict(np.repeat(X[:1], 3, axis=0)), np.ones(3), rtol=0, atol=1e-12)
                self.assertEqual(caught, [])

    def test_duplicate_constant_rows_are_supported_without_claiming_algebraic_rank(self):
        model = self.fresh().fit(np.ones((12, 2)), self.y)
        self.assertEqual(model.effective_n_components_, 4)
        self.assertNotIn("algebraic_rank", model.fit_details_)
        self.assertTrue(np.isfinite(model.mapper_.normalization_).all())
        np.testing.assert_allclose(model.predict(np.ones((3, 2))), np.full(3, self.y.mean()), rtol=0, atol=1e-12)

    def test_seeded_basis_is_shared_across_alphas_and_clone_is_unfitted(self):
        models = [self.fresh(alpha=alpha, random_state=11).fit(self.X, self.y)
                  for alpha in (0.1, 1.0, 10.0)]
        for model in models:
            np.testing.assert_array_equal(model.mapper_.component_indices_, models[0].mapper_.component_indices_)
            self.assertEqual(model.fit_details_["basis_indices_sha256"], models[0].fit_details_["basis_indices_sha256"])
            copied = clone(model)
            self.assertEqual(copied.get_params(), model.get_params())
            self.assert_unfitted(copied)
            copied.fit(self.X, self.y)
            np.testing.assert_array_equal(copied.predict(self.X), model.predict(self.X))

    def test_configuration_fails_before_map_fitting(self):
        configurations = [{"alpha": value} for value in (True, 0, -1, np.inf, np.nan, "1", [])]
        configurations += [{"rank_cap": value} for value in (True, 0, 65, 1.5, None)]
        configurations += [{"random_state": value} for value in (None, True, -1, 1.5, 2**32)]
        if np.finfo(np.longdouble).max > np.finfo(float).max:
            configurations += [{"alpha": np.longdouble(value)} for value in ("1e-4000", "1e4000")]
        for configuration in configurations:
            with self.subTest(configuration=configuration):
                model = NystroemRidgeRegressor(**configuration)
                with patch.object(Nystroem, "fit") as fitting, self.assertRaises(ValueError):
                    model.fit(self.X, self.y)
                fitting.assert_not_called()
                self.assert_unfitted(model)

    def test_active_masks_in_supported_containers_fail_before_feature_coercion(self):
        masked = np.ma.array(self.X, mask=False)
        masked.mask[0, 0] = True
        objects = self.X.astype(object)
        objects[0, 0] = np.ma.array(1.0, mask=True)
        rows = [np.ma.array(row, mask=False) for row in self.X]
        rows[0].mask[0] = True
        cases = [masked, rows, tuple(rows), objects, pd.DataFrame(objects)]
        fitted = self.fresh().fit(self.X, self.y)
        for X in cases:
            for action in (lambda: self.fresh().fit(X, self.y), lambda: fitted.predict(X)):
                with self.subTest(container=type(X).__name__):
                    with patch.object(kernel, "check_array") as checking:
                        with self.assertRaisesRegex(ValueError, "masked entries"):
                            action()
                        checking.assert_not_called()
        for masked_y in (np.ma.array(self.y, mask=[True] + [False] * 11),
                         [np.ma.array(0.0, mask=True)] + list(self.y[1:]),
                         np.array([np.ma.masked] + list(self.y[1:]), dtype=object),
                         pd.Series([np.ma.masked] + list(self.y[1:]), dtype=object)):
            with self.subTest(target=type(masked_y).__name__):
                with patch.object(Nystroem, "fit") as fitting:
                    with self.assertRaisesRegex(ValueError, "masked entries"):
                        self.fresh().fit(self.X, masked_y)
                    fitting.assert_not_called()

    def test_nomask_and_all_false_masks_preserve_observed_values(self):
        expected = self.fresh().fit(self.X, self.y).predict(self.X)
        for mask in (np.ma.nomask, False):
            X, y = np.ma.array(self.X, mask=mask), np.ma.array(self.y, mask=mask)
            actual = self.fresh().fit(X, y)
            np.testing.assert_array_equal(actual.predict(X), expected)
            self.assertFalse(np.ma.getmaskarray(X).any())
            self.assertFalse(np.ma.getmaskarray(y).any())
        rows = [np.ma.array(row, mask=False) for row in self.X]
        np.testing.assert_array_equal(self.fresh().fit(rows, self.y).predict(rows), expected)

    def test_sparse_numeric_and_structural_masks_are_checked_without_format_conversion(self):
        fitted = self.fresh().fit(self.X, self.y)
        for field in ("data", "indices", "indptr"):
            for active in (False, True):
                with self.subTest(field=field, active=active):
                    X = sparse.csr_matrix(self.X)
                    buffer = np.ma.array(getattr(X, field), mask=False)
                    buffer.mask[0] = active
                    setattr(X, field, buffer)
                    before = buffer.copy()
                    if active:
                        with patch.object(kernel, "check_array") as checking:
                            with self.assertRaisesRegex(ValueError, "masked entries"):
                                self.fresh().fit(X, self.y)
                            with self.assertRaisesRegex(ValueError, "masked entries"):
                                fitted.predict(X)
                            checking.assert_not_called()
                    else:
                        self.assertTrue(np.isfinite(self.fresh().fit(X, self.y).predict(X)).all())
                    np.testing.assert_array_equal(buffer.data, before.data)
                    np.testing.assert_array_equal(buffer.mask, before.mask)
        for factory in (sparse.csc_matrix, sparse.coo_matrix):
            X = factory(self.X)
            with self.subTest(format=X.format), patch.object(kernel, "check_array") as checking:
                with self.assertRaisesRegex(ValueError, "CSR format"):
                    self.fresh().fit(X, self.y)
                with self.assertRaisesRegex(ValueError, "CSR format"):
                    fitted.predict(X)
                checking.assert_not_called()

    def test_temporal_values_are_rejected_before_unit_or_nat_coercion(self):
        fitted = self.fresh().fit(self.X, self.y)
        for temporal in (np.datetime64("NaT"), np.datetime64("2020-01-01"),
                         np.timedelta64("NaT"), np.timedelta64(1, "ns"),
                         datetime(2020, 1, 1), timedelta(days=1), pd.NaT):
            with self.subTest(temporal=repr(temporal)):
                X = self.X.astype(object)
                X[0, 0] = temporal
                for action in (lambda: self.fresh().fit(X, self.y), lambda: fitted.predict(X),
                               lambda: self.fresh().fit(self.X, [temporal] + list(self.y[1:]))):
                    with patch.object(Nystroem, "fit") as fitting:
                        with self.assertRaisesRegex(ValueError, "datetime or timedelta"):
                            action()
                        fitting.assert_not_called()
        for dtype in ("datetime64[ns]", "timedelta64[ns]"):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "datetime or timedelta"):
                self.fresh().fit(np.zeros((12, 2), dtype=dtype), self.y)

    def test_target_precision_policy_is_delegated_with_the_original_object(self):
        original_y = [0.0, 2**53 + 1] + [0.0] * 10
        with patch.object(kernel, "validate_target", side_effect=ValueError("central target policy")) as validation:
            with patch.object(Nystroem, "fit") as fitting:
                with self.assertRaisesRegex(ValueError, "central target policy"):
                    self.fresh().fit(self.X, original_y)
                fitting.assert_not_called()
        self.assertIs(validation.call_args.args[0], original_y)
        self.assertEqual(validation.call_args.args[1], len(original_y))

    def test_typed_target_cases_follow_the_central_policy_without_narrowing_first(self):
        cases = [np.full(12, 2**53 + 1, dtype=np.int64),
                 [0.0, 2**53 + 1] + [0.0] * 10,
                 np.arange(12, dtype=np.longdouble),
                 ["1"] * 12, [True, False] * 6]
        for y in cases:
            with self.subTest(target_type=type(y).__name__):
                try:
                    normalized = validate_target(y, 12)
                except ValueError:
                    with patch.object(Nystroem, "fit") as fitting:
                        with self.assertRaises(ValueError):
                            self.fresh().fit(self.X, y)
                        fitting.assert_not_called()
                else:
                    direct = self.fresh().fit(self.X, y)
                    normalized_fit = self.fresh().fit(self.X, normalized)
                    np.testing.assert_array_equal(direct.predict(self.X), normalized_fit.predict(self.X))

    def test_cycles_deep_containers_and_shared_views_are_handled_before_coercion(self):
        cyclic = []
        cyclic.append(cyclic)
        object_cycle = np.empty(1, dtype=object)
        object_cycle[0] = object_cycle
        deep = 1.0
        for _ in range(70):
            deep = [deep]
        for value in (cyclic, object_cycle, deep):
            with self.subTest(type=type(value).__name__), patch.object(kernel, "check_array") as checking:
                with self.assertRaisesRegex(ValueError, "cyclic|nesting"):
                    self.fresh().fit(value, self.y)
                checking.assert_not_called()
        shared = [1.0, 2.0]
        self.assertTrue(np.isfinite(self.fresh().fit([shared] * 12, self.y).predict([shared])).all())

    def test_invalid_feature_and_target_shapes_fail_before_fitting(self):
        cases = [(self.X, self.y[:, None]), (self.X, self.y[:-1]),
                 (self.X, np.full(12, np.nan)), (self.X, np.full(12, np.inf)),
                 (self.X, self.y.astype(complex)),
                 (np.full((12, 2), np.nan), self.y), (np.full((12, 2), np.inf), self.y),
                 (np.ones((1, 6401)), np.ones(1)), (np.ones((20001, 1)), np.ones(20001))]
        for X, y in cases:
            with self.subTest(shape=np.shape(X), target_shape=np.shape(y)):
                model = self.fresh()
                with patch.object(Nystroem, "fit") as fitting, self.assertRaises(ValueError):
                    model.fit(X, y)
                fitting.assert_not_called()
                self.assert_unfitted(model)

    def test_byte_limit_is_exact_and_refuses_before_fit_or_predict_allocation(self):
        self.assertEqual(kernel._map_bytes(20000, 64), 10_240_000)
        self.assertEqual(kernel._map_bytes(32768, 64), 16 * 1024 * 1024)
        for dimensions in ((32769, 64), (2**63, 64), (True, 4), (0, 4), (3, -1)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                kernel._map_bytes(*dimensions)
        model = self.fresh().fit(self.X, self.y)
        with patch.object(kernel, "MAX_FEATURE_MAP_BYTES", 8):
            with patch.object(Nystroem, "fit") as fit, self.assertRaisesRegex(ValueError, "per-array limit"):
                self.fresh().fit(self.X, self.y)
            fit.assert_not_called()
            with patch.object(Nystroem, "transform") as transform, self.assertRaisesRegex(ValueError, "per-array limit"):
                model.predict(self.X)
            transform.assert_not_called()

    def test_sparse_original_matrix_is_not_densified(self):
        # A rank-limited Gram product may legitimately become dense (including
        # inside sklearn 1.4). Its width differs from this raw feature space.
        width = kernel.MAX_RANK + 1
        rank = self.fresh().rank_cap
        X = sparse.csr_matrix(np.arange(8 * width, dtype=float).reshape(8, width) / (8 * width))
        target = np.arange(8, dtype=float)
        original = sparse.csr_matrix.toarray
        dense_shapes = []
        def guarded(matrix, *args, **kwargs):
            if matrix.shape[1] == width:
                raise AssertionError("raw densification")
            self.assertLessEqual(matrix.shape[1], rank)
            dense_shapes.append(matrix.shape)
            return original(matrix, *args, **kwargs)
        with patch.object(sparse.csr_matrix, "toarray", guarded):
            # Prove the trap still rejects raw data and copies/subsets of it.
            for raw in (X, X.copy(), X[:2]):
                with self.assertRaisesRegex(AssertionError, "raw densification"):
                    raw.toarray()
            model = self.fresh().fit(X, target)
            prediction = model.predict(X[:2])
        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(sparse.issparse(model.mapper_.components_))
        self.assertTrue(all(columns <= rank for _, columns in dense_shapes))

    def test_noncanonical_csr_is_equivalent_without_mutating_caller_storage(self):
        X = sparse.csr_matrix((np.array([1., 2., 4., 5., 6.]),
                               np.array([1, 0, 1, 1, 0]), np.array([0, 3, 5])), shape=(2, 2))
        self.assertFalse(X.has_canonical_format)
        before = (X.data.copy(), X.indices.copy(), X.indptr.copy())
        dense = X.toarray()
        actual = self.fresh().fit(X, [1., 2.])
        expected = self.fresh().fit(dense, [1., 2.])
        np.testing.assert_allclose(actual.predict(X), expected.predict(dense), rtol=0, atol=1e-12)
        for values, original in zip((X.data, X.indices, X.indptr), before):
            np.testing.assert_array_equal(values, original)

    def test_basis_and_normalization_corruption_prevent_publish_and_preserve_previous_fit(self):
        model = self.fresh().fit(self.X, self.y)
        before = self.snapshot(model)
        original = Nystroem.fit
        for attribute in ("components_", "normalization_"):
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(attribute=attribute, bad=bad):
                    def corrupt(mapper, X, y=None):
                        result = original(mapper, X, y)
                        getattr(mapper, attribute).flat[0] = bad
                        return result
                    with patch.object(Nystroem, "fit", corrupt), patch.object(Nystroem, "transform") as transform:
                        with self.assertRaisesRegex(ValueError, "non-finite"):
                            model.fit(self.X + 10, self.y + 1)
                        transform.assert_not_called()
                    self.assert_preserved(model, before)
        model.fit(self.X + 10, self.y + 1)
        self.assertIsNot(model.mapper_, before[0]["mapper_"])

    def test_out_of_training_basis_and_invalid_indices_are_rejected(self):
        original = Nystroem.fit
        for change in ("basis", "indices", "duplicate", "gamma"):
            with self.subTest(change=change):
                def corrupt(mapper, X, y=None):
                    result = original(mapper, X, y)
                    if change == "basis":
                        mapper.components_.flat[0] += 100
                    elif change == "indices":
                        mapper.component_indices_[0] = len(X)
                    elif change == "duplicate":
                        mapper.component_indices_[0] = mapper.component_indices_[1]
                    else:
                        mapper.gamma = np.inf
                    return result
                model = self.fresh()
                with patch.object(Nystroem, "fit", corrupt), self.assertRaises(ValueError):
                    model.fit(self.X, self.y)
                self.assert_unfitted(model)

    def test_nonfinite_training_map_and_ridge_state_cannot_be_published(self):
        model = self.fresh().fit(self.X, self.y)
        before = self.snapshot(model)
        with patch.object(Nystroem, "transform", return_value=np.full((12, 4), np.nan)):
            with patch.object(Ridge, "fit") as fit, self.assertRaisesRegex(ValueError, "feature map"):
                model.fit(self.X, self.y)
            fit.assert_not_called()
        self.assert_preserved(model, before)
        original = Ridge.fit
        for attribute in ("coef_", "intercept_"):
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(attribute=attribute, bad=bad):
                    def corrupt(ridge, X, y, *args, **kwargs):
                        result = original(ridge, X, y, *args, **kwargs)
                        if attribute == "coef_":
                            ridge.coef_.flat[0] = bad
                        else:
                            ridge.intercept_ = bad
                        return result
                    with patch.object(Ridge, "fit", corrupt), self.assertRaisesRegex(ValueError, "non-finite"):
                        model.fit(self.X, self.y)
                    self.assert_preserved(model, before)

    def test_prediction_checks_saved_state_map_output_and_width(self):
        model = self.fresh().fit(self.X, self.y)
        with self.assertRaisesRegex(ValueError, "width"):
            model.predict(np.ones((2, 3)))
        with patch.object(Nystroem, "transform", return_value=np.full((12, 4), np.inf)):
            with self.assertRaisesRegex(ValueError, "feature map"):
                model.predict(self.X)
        with patch.object(Ridge, "predict", return_value=np.full(12, np.nan)):
            with self.assertRaisesRegex(ValueError, "predictions"):
                model.predict(self.X)
        value = model.ridge_.coef_[0]
        model.ridge_.coef_[0] = np.nan
        with patch.object(Nystroem, "transform") as transform, self.assertRaisesRegex(ValueError, "coefficients"):
            model.predict(self.X)
        transform.assert_not_called()
        model.ridge_.coef_[0] = value

    def test_numerical_warnings_fail_but_unrelated_warnings_are_not_hidden_or_failed(self):
        original = Nystroem.fit
        for category in (RuntimeWarning, LinAlgWarning, UserWarning, DeprecationWarning, FutureWarning):
            with self.subTest(category=category.__name__):
                def warned(mapper, X, y=None):
                    warnings.warn("anonymous software warning", category)
                    return original(mapper, X, y)
                with patch.object(Nystroem, "fit", warned):
                    if category in (RuntimeWarning, LinAlgWarning):
                        model = self.fresh()
                        with self.assertRaisesRegex(ValueError, "numerical failure during basis fitting"):
                            model.fit(self.X, self.y)
                        self.assert_unfitted(model)
                    else:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            self.fresh().fit(self.X, self.y)
                        self.assertEqual([item.category for item in caught], [category])

    def test_tail_underflow_is_allowed_but_distance_overflow_is_rejected(self):
        model = self.fresh().fit(self.X, self.y)
        with np.errstate(all="raise"):
            self.assertTrue(np.isfinite(model.predict(np.full((1, 2), 1e100))).all())
            with self.assertRaisesRegex(ValueError, "numerical failure"):
                model.predict(np.full((1, 2), 1e200))

    def test_interrupt_preserves_previous_fit_and_unsupported_metadata_is_not_ignored(self):
        model = self.fresh().fit(self.X, self.y)
        before = self.snapshot(model)
        with patch.object(Nystroem, "fit", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            model.fit(self.X, self.y)
        self.assert_preserved(model, before)
        for metadata in ({"groups": np.arange(12)}, {"sample_weight": np.ones(12)}):
            with self.assertRaises(TypeError):
                model.fit(self.X, self.y, **metadata)

    def test_existing_preprocessing_is_fold_local_with_missing_and_unseen_categories(self):
        frame = pd.DataFrame({"x": np.arange(12, dtype=float),
                              "kind": ["a", "b", None] * 4, "empty": np.full(12, np.nan)})
        frame.loc[2, "x"] = np.nan
        X = validate_features(frame)
        groups = np.repeat([f"g{i}" for i in range(6)], 2)
        for train, test in GroupKFold(3, 7).split(X, groups=groups):
            def fit(source, targets):
                training = source.iloc[train]
                preprocess = make_pipeline(training, Candidate("reference", "ridge", {"alpha": 1.0}), 7).named_steps["preprocess"]
                return Pipeline([("preprocess", preprocess), ("model", self.fresh(random_state=7))]).fit(training, targets[train])
            first = fit(X, self.y)
            changed_y = self.y.copy()
            changed_y[test] += 1000
            second = fit(X, changed_y)
            np.testing.assert_array_equal(first.predict(X.iloc[test]), second.predict(X.iloc[test]))
            self.assertEqual(first.named_steps["model"].fit_details_, second.named_steps["model"].fit_details_)
            changed_X = X.copy()
            changed_X.loc[test, "x"] = 12345
            changed_X.loc[test, "kind"] = "unseen"
            third = fit(changed_X, changed_y)
            self.assertEqual(first.named_steps["model"].fit_details_, third.named_steps["model"].fit_details_)
            numeric = first.named_steps["preprocess"].named_transformers_["numeric"]
            self.assertEqual(numeric.named_steps["impute"].statistics_[0], np.nanmedian(X.iloc[train].x))
            self.assertTrue(np.isfinite(third.predict(changed_X.iloc[test])).all())

    def test_categorical_only_and_empty_numeric_preprocessing_remain_usable(self):
        cases = [(pd.DataFrame({"kind": ["a", "b", None] * 4}), pd.DataFrame({"kind": ["unseen"]})),
                 (pd.DataFrame({"empty": np.full(12, np.nan)}), pd.DataFrame({"empty": [np.nan]}))]
        for training, incoming in cases:
            with self.subTest(columns=list(training)):
                training, incoming = validate_features(training), validate_features(incoming)
                preprocess = make_pipeline(training, Candidate("reference", "ridge", {"alpha": 1.0}), 0).named_steps["preprocess"]
                pipeline = Pipeline([("preprocess", preprocess), ("model", self.fresh())]).fit(training, self.y)
                self.assertTrue(np.isfinite(pipeline.predict(incoming)).all())

    def test_ordinary_numeric_unit_conversion_preserves_fold_preprocessed_predictions(self):
        training = pd.DataFrame({"current": np.arange(12, dtype=float) * 1e-9,
                                 "setting": np.arange(12) % 3})
        incoming = pd.DataFrame({"current": [3.5e-9, 9.5e-9], "setting": [1., 2.]})
        predictions = []
        for multiplier in (1., 1e9):
            train, new = training.copy(), incoming.copy()
            train["current"] *= multiplier
            new["current"] *= multiplier
            preprocess = make_pipeline(train, Candidate("reference", "ridge", {"alpha": 1.0}), 0).named_steps["preprocess"]
            pipeline = Pipeline([("preprocess", preprocess), ("model", self.fresh())]).fit(train, self.y)
            predictions.append(pipeline.predict(new))
        np.testing.assert_allclose(predictions[0], predictions[1], rtol=0, atol=1e-10)

    def test_optional_family_does_not_change_existing_grid(self):
        self.assertEqual([candidate.name for candidate in candidates_for(("dummy", "ridge", "extra_trees"))],
                         ["dummy_mean", "ridge_alpha_0.1", "ridge_alpha_1", "ridge_alpha_10",
                          "extra_trees_depth_6", "extra_trees_depth_none"])
        self.assertEqual(len(candidates_for(("nystroem_ridge",))), 3)


if __name__ == "__main__":
    unittest.main()
