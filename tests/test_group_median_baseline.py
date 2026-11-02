"""Anonymous software fixtures only; not another scientific benchmark."""

from collections import Counter
from fractions import Fraction
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from groupbench.baselines import GroupBalancedMedianRegressor
from groupbench import baselines as baseline_module
from groupbench.data import MAX_ROWS


def exact_loss(target, groups, value):
    identities = set(groups)
    return sum((sum((abs(Fraction(float(y)) - Fraction(float(value)))
                     for y, group in zip(target, groups) if group == identity), Fraction())
                / sum(group == identity for group in groups)
                for identity in identities), Fraction()) / len(identities)


def fraction_oracle(target, groups):
    counts = Counter(groups)
    cumulative = Fraction()
    for y, group in sorted(zip(target, groups), key=lambda row: row[0]):
        cumulative += Fraction(1, counts[group])
        if cumulative >= Fraction(len(counts), 2):
            return float(y)
    raise AssertionError("Fixture has no median")


class GroupMedianTests(unittest.TestCase):
    def estimator(self, target, groups, X=None):
        X = np.ones((len(target), 1)) if X is None else X
        return GroupBalancedMedianRegressor().fit(X, target, groups=groups)

    def test_two_group_counterexample_and_exact_training_minimum(self):
        target, groups = [0, 100, 101, 1], ["a", "a", "a", "b"]
        estimator = self.estimator(target, groups)
        self.assertEqual(estimator.constant_, 1)
        self.assertNotEqual(estimator.constant_, np.median([100, 1]))
        for value in target:
            self.assertLessEqual(exact_loss(target, groups, estimator.constant_), exact_loss(target, groups, value))

    def test_three_group_counterexample(self):
        target = [0, 0, 100, 1, 1, 100, 50, 50, 50]
        groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
        self.assertEqual(self.estimator(target, groups).constant_, 50)
        self.assertEqual(float(np.median([0, 1, 50])), 1)

    def test_fraction_oracle_on_fixed_varied_sizes_and_signs(self):
        cases = [([-8, -2, 0, 3, 9], ["a", "a", "b", "c", "c"]),
                 ([1, 1, 8, 9, 9, 10, 100], ["a", "b", "b", "c", "c", "c", "c"]),
                 ([0, 2, 8, 10], ["a", "a", "b", "b"])]
        for target, groups in cases:
            with self.subTest(target=target):
                actual = self.estimator(target, groups).constant_
                self.assertEqual(actual, fraction_oracle(target, groups))
                for value in target:
                    self.assertLessEqual(exact_loss(target, groups, actual), exact_loss(target, groups, value))

    def test_exact_half_tie_uses_lower_target(self):
        self.assertEqual(self.estimator([0, 10], ["a", "b"]).constant_, 0)
        self.assertEqual(self.estimator([0] * 3 + [10] * 7, ["a"] * 3 + ["b"] * 7).constant_, 0)

    def test_row_order_and_relabel_invariance(self):
        target = np.asarray([0, 0, 100, 1, 1, 100, 50, 50, 50])
        groups = np.asarray(["a"] * 3 + ["b"] * 3 + ["c"] * 3)
        rng = np.random.default_rng(20261103)  # Fixed software-test permutations.
        for _ in range(12):
            order = rng.permutation(len(target))
            self.assertEqual(self.estimator(target[order], groups[order]).constant_, 50)
        self.assertEqual(self.estimator(target, ["source_" + g for g in groups]).constant_, 50)

    def test_within_group_replication_invariance(self):
        first = self.estimator([0, 100, 101, 1], ["a"] * 3 + ["b"])
        second = self.estimator([0, 100, 101] * 11 + [1], ["a"] * 33 + ["b"])
        self.assertEqual(first.constant_, second.constant_)
        self.assertEqual(second.n_samples_seen_, 34)
        self.assertEqual(second.n_groups_seen_, 2)

    def test_training_groups_only_change_training_weighting(self):
        target = [0, 100, 101, 1]
        grouped = self.estimator(target, ["a", "a", "a", "b"])
        per_row = self.estimator(target, ["a", "b", "c", "d"])
        self.assertEqual(grouped.constant_, 1)
        self.assertEqual(per_row.constant_, 1)  # Lower middle for four rows.
        subset = self.estimator(target[:3], ["a"] * 3)
        self.assertEqual(subset.constant_, 100)
        with self.assertRaises(TypeError):
            grouped.predict(np.ones((2, 1)), groups=["never_seen", "new"])

    def test_no_predictor_signal_or_future_values_used(self):
        estimator = self.estimator([1, 2, 100], ["a", "a", "b"])
        np.testing.assert_array_equal(estimator.predict([[1e300], [-1e300], [0]]), [2, 2, 2])
        self.assertFalse(any(key in estimator.__dict__ for key in ("groups_", "y_", "sample_weight_", "X_")))

    def test_extreme_finite_targets_without_averaging_overflow(self):
        large = np.finfo(float).max
        self.assertEqual(self.estimator([-large, large], ["a", "b"]).constant_, -large)
        tiny = np.nextafter(0.0, 1.0)
        self.assertEqual(self.estimator([tiny, tiny * 2], ["a", "b"]).constant_, tiny)

    def test_signed_zero_is_canonical(self):
        for target in ([-0.0, 0.0], [0.0, -0.0]):
            value = self.estimator(target, ["a", "b"]).constant_
            self.assertEqual(value, 0)
            self.assertFalse(np.signbit(value))

    def test_one_group_and_numeric_zero_identifier_are_valid(self):
        estimator = self.estimator([5], [0])
        self.assertEqual(estimator.constant_, 5)
        self.assertEqual(estimator.n_groups_seen_, 1)

    def test_sample_weight_is_never_silently_combined(self):
        estimator = self.estimator([1, 2], ["a", "b"])
        for weights in ([0, 0], [1, 1], [1, 2], [], "unknown", False, float("nan")):
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, "sample_weight"):
                estimator.fit([[1], [2]], [1, 2], groups=["a", "b"], sample_weight=weights)
        self.assertEqual(estimator.constant_, 1)

    def test_missing_invalid_or_masked_groups_rejected(self):
        invalid = [None, [], ["a"], [["a"], ["b"]], ["a", None], ["a", ""], ["a", "\x00"],
                   [1, "1"], [True, False], [0, float("inf")], [{}, {}],
                   np.ma.array(["a", "b"], mask=[False, True])]
        for groups in invalid:
            with self.subTest(groups=repr(groups)), self.assertRaises((ValueError, TypeError)):
                GroupBalancedMedianRegressor().fit([[1], [2]], [1, 2], groups=groups)

    def test_target_invalid_and_masked_values_rejected(self):
        for target in ([1], [[1], [2]], [1, np.nan], [1, np.inf], [1 + 2j, 2],
                       np.ma.array([1, 999], mask=[False, True])):
            with self.subTest(target=repr(target)), self.assertRaises(ValueError):
                GroupBalancedMedianRegressor().fit([[1], [2]], target, groups=["a", "b"])

    def test_numeric_matrix_contract_and_feature_count(self):
        for matrix in ([], [1, 2], np.ones((2, 0)), [[[1]], [[2]]], [[np.nan], [1]],
                       [[np.inf], [1]], [[1j], [2]], np.ma.array([[1], [2]], mask=[[False], [True]])):
            with self.subTest(matrix=repr(matrix)), self.assertRaises(ValueError):
                GroupBalancedMedianRegressor().fit(matrix, [1, 2], groups=["a", "b"])
        estimator = self.estimator([1, 2], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "feature count"):
            estimator.predict(np.ones((2, 2)))

    def test_named_schema_unknown_reordered_or_missing_columns_rejected(self):
        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        estimator = self.estimator([1, 2], ["a", "b"], frame)
        for new in (frame.rename(columns={"a": "unknown"}), frame[["b", "a"]], frame.to_numpy()):
            with self.assertRaisesRegex(ValueError, "names and order"):
                estimator.predict(new)
        with self.assertRaises(ValueError):
            self.estimator([1, 2], ["a", "b"], frame.set_axis(["a", "a"], axis=1))
        with self.assertRaises(ValueError):
            self.estimator([1, 2], ["a", "b"], frame.set_axis([0, 1], axis=1))
        np.testing.assert_array_equal(estimator.predict(frame), [1, 1])

    def test_dense_csr_csc_and_no_sparse_densification(self):
        X = np.asarray([[0, 1], [2, 0], [0, 3], [4, 0]])
        for matrix in (X, sparse.csr_matrix(X), sparse.csc_matrix(X)):
            with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("dense CSR")), patch.object(sparse.csc_matrix, "toarray", side_effect=AssertionError("dense CSC")):
                estimator = self.estimator([0, 100, 101, 1], ["a", "a", "a", "b"], matrix)
                np.testing.assert_array_equal(estimator.predict(matrix), [1] * 4)

    def test_pipeline_explicit_training_group_forwarding_and_clone(self):
        pipeline = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()),
                             ("model", GroupBalancedMedianRegressor())])
        X = np.asarray([[0], [np.nan], [2], [3]])
        trained = pipeline.fit(X, [0, 100, 101, 1], model__groups=["a", "a", "a", "b"])
        np.testing.assert_array_equal(trained.predict([[999], [-999]]), [1, 1])
        self.assertEqual(trained.named_steps["impute"].statistics_[0], 5 / 3)
        fresh = clone(trained)
        with self.assertRaises(NotFittedError):
            fresh.predict([[1]])
        with self.assertRaisesRegex(ValueError, "groups is required"):
            fresh.fit(X, [0, 100, 101, 1])

    def test_clone_unknown_parameters_and_trusted_memory_pickle(self):
        estimator = self.estimator([0, 100, 101, 1], ["a", "a", "a", "b"])
        self.assertTrue(is_regressor(estimator))
        self.assertEqual(estimator.get_params(), {})
        with self.assertRaises(ValueError):
            estimator.set_params(strategy="unknown")
        fresh = clone(estimator)
        with self.assertRaises(NotFittedError):
            fresh.predict([[1]])
        # Only our own freshly-created in-memory bytes; never untrusted input.
        restored = pickle.loads(pickle.dumps(estimator))
        np.testing.assert_array_equal(restored.predict([[1], [2]]), [1, 1])
        self.assertEqual(restored.__dict__, estimator.__dict__)

    def test_failed_initial_and_refit_state_and_schema_replacement(self):
        initial = GroupBalancedMedianRegressor()
        with self.assertRaises(ValueError):
            initial.fit([[1]], [float("nan")], groups=["a"])
        with self.assertRaises(NotFittedError):
            initial.predict([[1]])
        frame = pd.DataFrame({"a": [1, 2]})
        estimator = self.estimator([1, 3], ["a", "b"], frame)
        prior = pickle.dumps(estimator)
        with self.assertRaises(ValueError):
            estimator.fit([[1]], [5], groups=[True])
        self.assertEqual(pickle.dumps(estimator), prior)
        estimator.fit([[0], [1]], [8, 9], groups=["a", "b"])
        self.assertFalse(hasattr(estimator, "feature_names_in_"))
        np.testing.assert_array_equal(estimator.predict([[0]]), [8])

    def test_max_rows_and_exact_large_denominator(self):
        groups = np.repeat(np.arange(1, 200), np.arange(1, 200))
        target = np.arange(len(groups), dtype=float) % 23 - 11
        self.assertEqual(len(target), 19900)
        estimator = self.estimator(target, groups)
        self.assertEqual(estimator.constant_, fraction_oracle(target, groups))
        self.assertEqual(self.estimator(np.zeros(MAX_ROWS), np.zeros(MAX_ROWS)).constant_, 0)
        with self.assertRaisesRegex(ValueError, "rows"):
            self.estimator(np.zeros(MAX_ROWS + 1), np.zeros(MAX_ROWS + 1))

    def test_nested_masked_feature_rows_rejected_before_conversion_fit_predict(self):
        def cases():
            row = np.ma.array([1.0], mask=[True])
            yield [row, [2.0]]
            yield (row, (2.0,))
            objects = np.empty(2, dtype=object)
            objects[0], objects[1] = row, np.asarray([2.0])
            yield objects
            cells = np.empty((2, 1), dtype=object)
            cells[0, 0], cells[1, 0] = np.ma.array(1.0, mask=True), 2.0
            yield cells
            yield pd.DataFrame(cells, columns=["x"])
            yield np.ma.array(cells, mask=False)

        for X in cases():
            with self.subTest(type=type(X).__name__):
                estimator = self.estimator([0, 1], ["a", "b"])
                before = pickle.dumps(estimator)
                with patch.object(baseline_module, "check_array", side_effect=AssertionError("masked input reached numeric conversion")):
                    with self.assertRaisesRegex(ValueError, "masked"):
                        estimator.fit(X, [0, 1], groups=["a", "b"])
                    with self.assertRaisesRegex(ValueError, "masked"):
                        estimator.predict(X)
                self.assertEqual(pickle.dumps(estimator), before)

    def test_nested_masks_in_target_and_groups_rejected_without_hidden_payload(self):
        for parameter in ("y", "groups"):
            scalar = np.ma.array(99.0 if parameter == "y" else "hidden", mask=True)
            objects = np.empty(2, dtype=object)
            objects[0], objects[1] = scalar, (1.0 if parameter == "y" else "visible")
            for value in ([scalar, objects[1]], (scalar, objects[1]), objects, pd.Series(objects)):
                kwargs = dict(y=[0, 1], groups=["a", "b"])
                kwargs[parameter] = value
                with self.subTest(parameter=parameter, kind=type(value).__name__), self.assertRaisesRegex(ValueError, "masked"):
                    GroupBalancedMedianRegressor().fit([[1], [2]], **kwargs)

    def test_nested_no_mask_control_and_input_nonmutation(self):
        row = np.ma.array([1.0], mask=np.ma.nomask)
        second = np.ma.array([2.0], mask=[False])
        for X in ([row, second], (row, second), np.asarray([[1.0], [2.0]], dtype=object)):
            estimator = self.estimator([0, 1], ["a", "b"], X)
            np.testing.assert_array_equal(estimator.predict(X), [0, 0])
        np.testing.assert_array_equal(row.data, [1])
        self.assertIs(row.mask, np.ma.nomask)
        np.testing.assert_array_equal(second.data, [2])
        np.testing.assert_array_equal(second.mask, [False])
        row.mask = [True]
        for action in ("fit", "predict"):
            with self.assertRaisesRegex(ValueError, "masked"):
                if action == "fit":
                    estimator.fit([row, second], [0, 1], groups=["a", "b"])
                else:
                    estimator.predict([row, second])
        np.testing.assert_array_equal(row.data, [1])
        np.testing.assert_array_equal(row.mask, [True])

    def test_cycles_deep_and_excessive_container_work_fail_without_recursion(self):
        cyclic = []
        cyclic.append(cyclic)
        object_cycle = np.empty(1, dtype=object)
        object_cycle[0] = object_cycle
        tuple_cycle = (cyclic,)
        deep = [1.0]
        for _ in range(65):
            deep = [deep]
        estimator = self.estimator([0, 1], ["a", "b"])
        for X in (cyclic, object_cycle, tuple_cycle, deep):
            for action in ("fit", "predict"):
                with self.subTest(action=action, kind=type(X).__name__), self.assertRaisesRegex(ValueError, "cyclic|depth"):
                    if action == "fit":
                        estimator.fit(X, [0, 1], groups=["a", "b"])
                    else:
                        estimator.predict(X)
        with patch.object(baseline_module, "_MAX_MASK_VISITS", 3):
            with self.assertRaisesRegex(ValueError, "inspection"):
                estimator.predict([[1], [2]])

    def test_shared_containers_and_guard_does_not_consume_generators(self):
        row = np.ma.array([1.0], mask=[False])
        estimator = self.estimator([0, 1], ["a", "b"], [row, row])
        np.testing.assert_array_equal(estimator.predict([row, row]), [0, 0])
        def forbidden():
            raise AssertionError("mask guard consumed arbitrary iterable")
            yield None
        baseline_module._reject_masked(forbidden(), "X")

    def test_sparse_data_and_index_buffer_masks_rejected_without_conversion(self):
        factories = [sparse.csr_matrix, sparse.csc_matrix]
        if hasattr(sparse, "csr_array"):
            factories.extend([sparse.csr_array, sparse.csc_array])
        for factory in factories:
            for buffer_name in ("data", "indices", "indptr"):
                X = factory([[1.0], [2.0]])
                original = getattr(X, buffer_name).copy()
                mask = np.zeros(len(original), dtype=bool)
                mask[0] = True
                setattr(X, buffer_name, np.ma.array(original, mask=mask))
                estimator = self.estimator([0, 1], ["a", "b"])
                prior = pickle.dumps(estimator)
                with self.subTest(format=factory.__name__, buffer=buffer_name), patch.object(baseline_module, "check_array", side_effect=AssertionError("masked sparse buffer reached conversion")):
                    with self.assertRaisesRegex(ValueError, "masked"):
                        estimator.fit(X, [0, 1], groups=["a", "b"])
                    with self.assertRaisesRegex(ValueError, "masked"):
                        estimator.predict(X)
                self.assertEqual(pickle.dumps(estimator), prior)
                np.testing.assert_array_equal(getattr(X, buffer_name).data, original)
                np.testing.assert_array_equal(getattr(X, buffer_name).mask, mask)

    def test_sparse_no_mask_buffers_remain_valid_without_densification(self):
        for factory in (sparse.csr_matrix, sparse.csc_matrix):
            for buffer_name in ("data", "indices", "indptr"):
                X = factory([[1.0], [2.0]])
                original = getattr(X, buffer_name).copy()
                setattr(X, buffer_name, np.ma.array(original, mask=False))
                with self.subTest(format=factory.__name__, buffer=buffer_name), patch.object(factory, "toarray", side_effect=AssertionError("sparse densification")):
                    estimator = self.estimator([0, 1], ["a", "b"], X)
                    np.testing.assert_array_equal(estimator.predict(X), [0, 0])
                np.testing.assert_array_equal(getattr(X, buffer_name).data, original)
                self.assertFalse(np.ma.getmaskarray(getattr(X, buffer_name)).any())

    def test_other_sparse_formats_are_rejected_not_silently_converted(self):
        estimator = self.estimator([0, 1], ["a", "b"])
        for factory in (sparse.coo_matrix, sparse.lil_matrix, sparse.dok_matrix, sparse.bsr_matrix, sparse.dia_matrix):
            X = factory([[1.0], [2.0]])
            for action in ("fit", "predict"):
                with self.subTest(format=factory.__name__, action=action), patch.object(baseline_module, "check_array", side_effect=AssertionError("unsupported sparse conversion")):
                    with self.assertRaisesRegex(ValueError, "CSR or CSC"):
                        if action == "fit":
                            estimator.fit(X, [0, 1], groups=["a", "b"])
                        else:
                            estimator.predict(X)
        coo = sparse.coo_matrix([[1.0], [2.0]])
        coo.data = np.ma.array(coo.data, mask=[True, False])
        with self.assertRaisesRegex(ValueError, "CSR or CSC"):
            estimator.predict(coo)
        self.assertTrue(coo.data.mask[0])


if __name__ == "__main__":
    unittest.main()
