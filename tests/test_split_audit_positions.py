"""Split auditing must count row positions, not implicit NumPy indexing modes."""
import unittest
from itertools import product

import numpy as np

from groupbench.splitting import audit_split, audit_split_row_eligibility


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

    def test_unknown_groups_cannot_be_certified_disjoint(self):
        rosters = (
            np.array([np.nan, np.nan]),
            np.ma.array(["A", "B"], mask=[True, False]),
            np.array(["A", "B", None], dtype=object),
        )
        for groups in rosters:
            with self.subTest(groups=groups):
                with self.assertRaisesRegex(ValueError, "non-missing group identifier"):
                    audit_split([0], [1], groups)

    def test_valid_subset_still_reports_exact_counts_and_group_overlap(self):
        report = audit_split(np.array([1, 0]), np.array([2], dtype=np.uint64), self.groups)
        self.assertEqual(report, {
            "train_rows": 2, "test_rows": 1, "train_group_count": 1,
            "test_group_count": 1, "overlap_count": 0,
            "train_groups": ["A"], "test_groups": ["B"],
        })
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_split(np.array([0]), np.array([1]), self.groups)


class SplitEligibilityTests(unittest.TestCase):
    def test_valid_split_can_lose_a_whole_declared_test_group(self):
        groups = np.array(["fit", "good", "not-known", "not-known", "failed", "unused"])
        statuses = ["eligible", "eligible", "unknown", "ineligible", "ineligible", "unknown"]
        fit, test = [0], [1, 2, 3, 4]
        self.assertEqual(audit_split(fit, test, groups)["test_group_count"], 3)
        result = audit_split_row_eligibility(fit, test, groups, eligibility=statuses)
        self.assertEqual(result["test"], {
            "n_declared_rows": 4, "n_declared_groups": 3,
            "eligible_rows": 1, "ineligible_rows": 2, "unknown_rows": 1,
            "groups_with_eligible_rows": 1, "groups_with_all_rows_eligible": 1,
            "groups_with_no_eligible_rows_but_unknown": 1, "groups_with_only_ineligible_rows": 1,
        })
        self.assertEqual(result["unused"]["unknown_rows"], 1)
        self.assertEqual(result, audit_split_row_eligibility(fit, test[::-1], groups,
                                                             eligibility=statuses))

    def test_every_small_status_pattern_matches_direct_group_checks(self):
        ids = ["fit", "fit", "left", "left", "right"]
        states = ("eligible", "ineligible", "unknown")
        for statuses in product(states, repeat=len(ids)):
            report = audit_split_row_eligibility([0, 1], [2, 3, 4], ids, eligibility=statuses)
            observed = (statuses[2:4], statuses[4:])
            test = report["test"]
            self.assertEqual(test["groups_with_eligible_rows"], sum("eligible" in x for x in observed))
            self.assertEqual(test["groups_with_no_eligible_rows_but_unknown"], sum(
                "eligible" not in x and "unknown" in x for x in observed))
            self.assertEqual(test["groups_with_only_ineligible_rows"], sum(
                all(y == "ineligible" for y in x) for x in observed))
            self.assertEqual(test["groups_with_all_rows_eligible"], sum(
                all(y == "eligible" for y in x) for x in observed))
            self.assertTrue(all(count == 0 for count in report["unused"].values()))

    def test_masked_statuses_and_positions_never_uncover_payload(self):
        ids = ["fit", "test", "unused"]
        statuses = ["eligible", "ineligible", "unknown"]
        for invalid in (statuses[:-1], ["eligible", "ineligible", None],
                        ["eligible", "ineligible", np.str_("unknown")],
                        np.ma.array(statuses, mask=[False, False, True])):
            with self.assertRaises(ValueError):
                audit_split_row_eligibility([0], [1], ids, eligibility=invalid)
        for fit, test in (([0, 0], [1]), ([0], [0]), ([True], [1]),
                          (np.ma.array([0], mask=[True]), [1])):
            with self.assertRaises(ValueError):
                audit_split_row_eligibility(fit, test, ids, eligibility=statuses)
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_split_row_eligibility([0], [1], ["same", "same", "unused"], eligibility=statuses)
