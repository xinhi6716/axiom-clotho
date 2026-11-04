"""Split alignment validation must never turn targets into split inputs."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench.splitting import GroupKFold


class SplitTargetAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat(list("abcdef"), 2)
        self.X = pd.DataFrame({"x": np.arange(12)})
        self.splitter = GroupKFold(n_splits=3, random_state=7)

    def test_wrong_target_length_rejected_before_group_assignment(self):
        for length in (0, 11, 13):
            for targets in (np.zeros(length), pd.Series(np.zeros(length))):
                for features in (self.X, None):
                    with self.subTest(length=length, kind=type(targets).__name__,
                                      has_features=features is not None):
                        with patch("groupbench.splitting.pd.factorize", wraps=pd.factorize) as factorize:
                            with self.assertRaisesRegex(
                                ValueError, "y and groups must contain the same number of rows"
                            ):
                                list(self.splitter.split(features, targets, self.groups))
                            factorize.assert_not_called()

    def test_targets_are_length_only_not_read_or_coerced(self):
        class LengthOnlyTargets:
            def __len__(self):
                return 12

            def __array__(self, *args, **kwargs):
                raise AssertionError("Target values must not be coerced")

            def __iter__(self):
                raise AssertionError("Target values must not be iterated")

            def __getitem__(self, key):
                raise AssertionError("Target values must not be indexed")

        expected = list(self.splitter.split(self.X, groups=self.groups))
        actual = list(self.splitter.split(self.X, LengthOnlyTargets(), self.groups))
        self.assert_splits_equal(expected, actual)

    def assert_splits_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for (train_a, test_a), (train_b, test_b) in zip(expected, actual):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)

    def test_changing_targets_and_index_labels_does_not_change_splits(self):
        expected = list(self.splitter.split(self.X, groups=self.groups))
        for targets in (
            np.arange(12), np.arange(12)[::-1], np.full(12, np.nan),
            pd.Series(np.arange(12), index=np.arange(100, 112)[::-1]),
        ):
            self.assert_splits_equal(
                expected, list(self.splitter.split(self.X, targets, self.groups))
            )
        tested = np.concatenate([test for _, test in expected])
        np.testing.assert_array_equal(np.sort(tested), np.arange(12))
        for train, test in expected:
            self.assertFalse(set(self.groups[train]) & set(self.groups[test]))

    def test_existing_X_length_check_remains(self):
        with self.assertRaisesRegex(ValueError, "X and groups"):
            list(self.splitter.split(self.X.iloc[:-1], np.zeros(12), self.groups))


if __name__ == "__main__":
    unittest.main()
