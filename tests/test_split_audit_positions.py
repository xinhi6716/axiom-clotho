"""Split auditing must count row positions, not implicit NumPy indexing modes."""
import unittest

import numpy as np

from groupbench.splitting import audit_split


class SplitAuditPositionTests(unittest.TestCase):
    def setUp(self):
        self.groups = np.array(["A", "A", "B", "C"])

    def test_duplicate_positions_do_not_inflate_audit_counts(self):
        for train, test in (([0, 0], [2]), ([0], [2, 2])):
            with self.subTest(train=train, test=test):
                with self.assertRaisesRegex(ValueError, "duplicate row positions"):
                    audit_split(np.array(train), np.array(test), self.groups)

    def test_negative_positions_do_not_alias_tail_rows(self):
        for train, test in (([-1], [0]), ([0], [-1])):
            with self.subTest(train=train, test=test):
                with self.assertRaisesRegex(ValueError, "outside.*row bounds"):
                    audit_split(np.array(train), np.array(test), self.groups)

    def test_boolean_masks_are_not_positional_partitions(self):
        mask = np.array([True, True, False, False])
        with self.assertRaisesRegex(ValueError, "integer row positions"):
            audit_split(mask, np.array([2]), self.groups)

    def test_large_unsigned_and_empty_partitions_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "outside.*row bounds"):
            audit_split(np.array([2**64 - 1], dtype=np.uint64), np.array([2]), self.groups)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            audit_split(np.array([], dtype=np.int64), np.array([2]), self.groups)

    def test_masked_inputs_do_not_silently_drop_mask(self):
        hidden = np.ma.array([0], mask=[True])
        with self.assertRaisesRegex(ValueError, "integer row positions"):
            audit_split(hidden, np.array([2]), self.groups)
        with self.assertRaisesRegex(ValueError, "integer row positions"):
            audit_split([0, True], np.array([2]), self.groups)

    def test_valid_subset_still_reports_exact_counts_and_group_overlap(self):
        report = audit_split(np.array([1, 0]), np.array([2], dtype=np.uint64), self.groups)
        self.assertEqual(report, {
            "train_rows": 2, "test_rows": 1, "train_group_count": 1,
            "test_group_count": 1, "overlap_count": 0,
            "train_groups": ["A"], "test_groups": ["B"],
        })
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_split(np.array([0]), np.array([1]), self.groups)
