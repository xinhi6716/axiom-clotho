"""Observable row-balance and seed semantics of the public GroupKFold."""
from __future__ import annotations

import unittest

import numpy as np

from groupbench import GroupKFold


class GroupSplitBalancingTests(unittest.TestCase):
    def test_unequal_groups_are_balanced_by_rows_not_by_group_count(self):
        sizes = {"e": 3, "f": 1, "b": 8, "c": 7, "d": 5, "a": 12}
        # Interleave rows so the large groups are neither encountered first nor
        # contiguous. With descending sizes and least row load, all folds reach
        # 12 rows: [12], [8, 3, 1] and [7, 5]. Group counts must differ.
        groups = np.array([
            group for repeat in range(max(sizes.values()))
            for group, size in sizes.items() if repeat < size
        ])
        all_rows = np.arange(len(groups))

        for seed in (0, 5, 17):
            with self.subTest(seed=seed):
                folds = list(GroupKFold(3, seed).split(groups=groups))
                self.assertEqual([len(test) for _, test in folds], [12, 12, 12])
                self.assertEqual(sorted(len(set(groups[test])) for _, test in folds), [1, 2, 3])
                for train, test in folds:
                    np.testing.assert_array_equal(train, np.setdiff1d(all_rows, test))
                    self.assertFalse(set(groups[train]) & set(groups[test]))
                np.testing.assert_array_equal(
                    np.sort(np.concatenate([test for _, test in folds])), all_rows
                )

    def test_seed_changes_tied_membership_not_just_fold_numbers(self):
        groups = np.tile(list("abcdefghi"), 2)

        def partition(seed):
            folds = GroupKFold(3, seed).split(groups=groups)
            return frozenset(frozenset(groups[test]) for _, test in folds)

        expected = partition(7)
        # Ignore fold ordering: relabeling identical test partitions is not
        # enough to show that the seed changed the tied assignment.
        self.assertGreater(len({partition(seed) for seed in range(8)}), 1)
        self.assertEqual(partition(7), expected)


if __name__ == "__main__":
    unittest.main()
