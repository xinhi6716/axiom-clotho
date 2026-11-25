"""Observable row-balance and seed semantics of the public GroupKFold."""
from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from groupbench import GroupKFold


def _legacy_folds(groups, n_splits, seed):
    """Reference: encounter-order groups and choice over each minimum-load array."""
    counts = Counter(groups)
    rng = np.random.default_rng(seed)
    names = list(counts)
    shuffled = [names[i] for i in rng.permutation(len(names))]
    loads, membership = [0] * n_splits, {}
    for group in sorted(shuffled, key=lambda name: -counts[name]):
        minimum = min(loads)
        candidates = np.array([i for i, load in enumerate(loads) if load == minimum])
        fold = int(rng.choice(candidates))
        loads[fold] += counts[group]
        membership[group] = fold
    return [(np.array([i for i, g in enumerate(groups) if membership[g] != fold]),
             np.array([i for i, g in enumerate(groups) if membership[g] == fold]))
            for fold in range(n_splits)]


class GroupSplitBalancingTests(unittest.TestCase):
    def test_exact_legacy_train_test_positions_across_tie_and_singleton_draws(self):
        for counts in (np.ones(29, dtype=int), np.arange(1, 30),
                       1 + np.random.default_rng(89).integers(12, size=29)):
            groups = np.repeat([f"device-{i}" for i in range(len(counts))], counts)
            np.random.default_rng(37).shuffle(groups)
            for n_splits in (2, 3, 5, 11, 29):
                for seed in (0, 1, 17, 97, 2**32 - 101):
                    with self.subTest(n_splits=n_splits, seed=seed, rows=len(groups)):
                        expected = _legacy_folds(groups, n_splits, seed)
                        actual = list(GroupKFold(n_splits, seed).split(groups=groups))
                        self.assertEqual(len(actual), len(expected))
                        for (train, test), (train_ref, test_ref) in zip(actual, expected):
                            np.testing.assert_array_equal(train, train_ref)
                            np.testing.assert_array_equal(test, test_ref)

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
