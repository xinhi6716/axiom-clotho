"""A hidden group payload is not an observed dependency identifier."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import GroupBenchClassifier, GroupBenchRegressor, GroupKFold
from groupbench.binary.metrics import grouped_binary_metrics
from groupbench.data import validate_groups
from groupbench.metrics import grouped_metrics


class MaskedGroupIdentifierTests(unittest.TestCase):
    def test_active_masks_rejected_without_revealing_or_mutating_hidden_ids(self):
        for raw in (["invented-wafer-A", "hidden-invented-wafer"], [1, 999]):
            for mask in ([False, True], [True, True]):
                for hard_mask in (False, True):
                    groups = np.ma.array(raw, mask=mask, hard_mask=hard_mask)
                    data, original_mask = groups.data.copy(), groups.mask.copy()
                    original_fill = groups.fill_value
                    with self.subTest(raw_type=type(raw[0]).__name__, mask=mask, hard_mask=hard_mask):
                        with self.assertRaisesRegex(ValueError, "non-missing group identifier") as caught:
                            validate_groups(groups, 2)
                        self.assertNotIn("hidden-invented-wafer", str(caught.exception))
                        self.assertNotIn("999", str(caught.exception))
                        np.testing.assert_array_equal(groups.data, data)
                        np.testing.assert_array_equal(groups.mask, original_mask)
                        self.assertEqual(groups.fill_value, original_fill)
                        self.assertEqual(groups.hardmask, hard_mask)

    def test_splitter_rejects_missing_ids_before_factorization(self):
        groups = np.ma.array(["invented-A", "invented-B"], mask=[False, True])
        with patch("groupbench.splitting.pd.factorize", side_effect=AssertionError("unexpected factorization")) as factorize:
            with self.assertRaisesRegex(ValueError, "non-missing group identifier"):
                list(GroupKFold(2).split(groups=groups))
            factorize.assert_not_called()

    def test_estimators_reject_masks_before_splits_or_model_construction(self):
        X = pd.DataFrame({"invented_feature": np.arange(12, dtype=float)})
        y = np.tile([0, 1], 6)
        groups = np.ma.array(np.repeat(list("abcdef"), 2), mask=False)
        groups.mask[3] = True
        for estimator_type in (GroupBenchRegressor, GroupBenchClassifier):
            with self.subTest(estimator=estimator_type.__name__), ExitStack() as stack:
                blocked = [stack.enter_context(patch(name, side_effect=AssertionError("unexpected model work")))
                           for name in ("groupbench.splitting.pd.factorize", "groupbench.model.make_pipeline",
                                        "groupbench.binary.model.binary_pipeline")]
                with self.assertRaisesRegex(ValueError, "non-missing group identifier"):
                    estimator_type(inner_splits=2, baselines=("dummy",)).fit(X, y, groups=groups)
                for operation in blocked:
                    operation.assert_not_called()

    def test_grouped_scores_reject_masks_before_assigning_rows_to_groups(self):
        groups = np.ma.array(["invented-A", "invented-B"], mask=[False, True])
        scorers = ((grouped_metrics, "groupbench.metrics.regression_metrics"),
                   (grouped_binary_metrics, "groupbench.binary.metrics.binary_metrics"))
        for scorer, target in scorers:
            with self.subTest(scorer=scorer.__name__):
                with patch(target, side_effect=AssertionError("unexpected scoring")) as score:
                    with self.assertRaisesRegex(ValueError, "non-missing group identifier"):
                        scorer([0, 1], [0.25, 0.75], groups)
                    score.assert_not_called()

    def test_unmasked_values_keep_exact_existing_identity_and_order(self):
        raw_values = (["1", "01", "1.0", " 1", "1 "],
                      np.array([1, 1.0, 2 ** 60, 2 ** 60 + 1], dtype=object),
                      np.array([np.float32(0.1), float(np.float32(0.1)), 0.1], dtype=object))
        for values in raw_values:
            expected = validate_groups(values, len(values))
            for mask in (False, np.ma.nomask):
                groups = np.ma.array(values, mask=mask)
                with self.subTest(dtype=str(groups.dtype), mask=mask):
                    np.testing.assert_array_equal(validate_groups(groups, len(groups)), expected)

    def test_unmasked_split_assignments_and_row_coverage_are_unchanged(self):
        raw = np.repeat(["invented-A", "invented-B", "invented-C", "invented-D"], [1, 3, 2, 4])
        for seed in range(3):
            expected = list(GroupKFold(3, seed).split(groups=raw))
            for mask in (False, np.ma.nomask):
                actual = list(GroupKFold(3, seed).split(groups=np.ma.array(raw, mask=mask)))
                for (train, test), (expected_train, expected_test) in zip(actual, expected):
                    np.testing.assert_array_equal(train, expected_train)
                    np.testing.assert_array_equal(test, expected_test)
                    self.assertFalse(set(raw[train]) & set(raw[test]))
                self.assertEqual(sorted(np.concatenate([test for _, test in actual]).tolist()),
                                 list(range(len(raw))))

    def test_failed_group_validation_preserves_existing_fitted_models(self):
        X = pd.DataFrame({"invented_feature": np.arange(8, dtype=float)})
        y, raw = np.tile([0, 1], 4), np.repeat(list("abcd"), 2)
        for estimator_type in (GroupBenchRegressor, GroupBenchClassifier):
            with self.subTest(estimator=estimator_type.__name__):
                estimator = estimator_type(inner_splits=2, baselines=("dummy",)).fit(X, y, groups=raw)
                before = estimator.predict(X)
                state = dict(estimator.__dict__)
                masked = np.ma.array(raw, mask=False)
                masked.mask[2] = True
                with self.assertRaisesRegex(ValueError, "non-missing group identifier"):
                    estimator.fit(X, y, groups=masked)
                self.assertEqual(set(estimator.__dict__), set(state))
                for key, value in state.items():
                    self.assertIs(estimator.__dict__[key], value)
                np.testing.assert_array_equal(estimator.predict(X), before)


if __name__ == "__main__":
    unittest.main()
