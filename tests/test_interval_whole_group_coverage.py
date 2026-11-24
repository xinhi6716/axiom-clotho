"""Synthetic contract tests for simultaneous coverage of declared groups."""
from copy import deepcopy
import unittest

from groupbench.interval_evaluation import (
    diagnose_saved_interval_whole_group_coverage,
    evaluate_saved_intervals,
)


def fixture():
    rows = [
        {"row_id": f"long-{i}", "group": "long", "observed": 2 if i == 7 else 1,
         "lower": 0, "upper": 1, "status": "ok"}
        for i in range(8)
    ]
    rows.append({"row_id": "short-0", "group": "short", "observed": 0,
                 "lower": 0, "upper": 0, "status": "ok"})
    return {
        "schema_version": 1,
        "metadata": {
            "target_definition": "invented response",
            "value_unit": "test units",
            "group_definition": "invented units",
            "cohort_definition": "all nine synthetic rows",
        },
        "alpha": 0.2,
        "expected_row_ids": [row["row_id"] for row in rows],
        "rows": rows,
    }


class WholeGroupCoverageTests(unittest.TestCase):
    def test_distinguishes_pooled_equal_group_row_and_whole_group_rates(self):
        doc = fixture()
        row_report = evaluate_saved_intervals(doc)
        result = diagnose_saved_interval_whole_group_coverage(doc)
        self.assertEqual(row_report["row_weighted"]["empirical_coverage_approx"], 8 / 9)
        self.assertEqual(row_report["equal_group"]["empirical_coverage_approx"], 15 / 16)
        self.assertEqual(result["whole_group"], {
            "observed_coverage_exact": "1/2",
            "n_groups": 2,
            "n_groups_all_rows_covered": 1,
        })
        self.assertEqual(result["groups"], [
            {"group": "long", "n_rows": 8, "n_covered": 7, "all_rows_covered": False},
            {"group": "short", "n_rows": 1, "n_covered": 1, "all_rows_covered": True},
        ])
        self.assertFalse(result["provenance_verified"])

    def test_all_groups_covered_then_all_groups_missed(self):
        doc = fixture()
        doc["rows"][7]["observed"] = 1  # A closed endpoint is covered.
        self.assertEqual(diagnose_saved_interval_whole_group_coverage(doc)["whole_group"]
                         ["observed_coverage_exact"], "1")
        doc["rows"][0]["observed"] = -1
        doc["rows"][-1]["observed"] = 1
        self.assertEqual(diagnose_saved_interval_whole_group_coverage(doc)["whole_group"]
                         ["observed_coverage_exact"], "0")

    def test_replication_within_group_keeps_one_vote_per_group(self):
        doc = fixture()
        before = diagnose_saved_interval_whole_group_coverage(doc)
        copied = dict(doc["rows"][0], row_id="long-copy")
        doc["rows"].append(copied)
        doc["expected_row_ids"].append(copied["row_id"])
        after = diagnose_saved_interval_whole_group_coverage(doc)
        self.assertEqual(after["whole_group"], before["whole_group"])
        self.assertEqual(after["groups"][0]["n_rows"], 9)

    def test_refuses_incomplete_duplicate_or_failed_declared_rows(self):
        missing = fixture()
        missing["rows"].pop(7)  # The failing group still has seven rows.
        duplicate = fixture()
        duplicate["rows"][1] = duplicate["rows"][0].copy()
        failed = fixture()
        failed["rows"][0]["status"] = "failed"
        for doc in (missing, duplicate, failed):
            with self.subTest(doc=doc), self.assertRaises(ValueError):
                diagnose_saved_interval_whole_group_coverage(doc)

    def test_default_evaluation_unchanged_and_sidecar_reorder_stable(self):
        doc = fixture()
        frozen = deepcopy(doc)
        default_before = evaluate_saved_intervals(doc)
        result = diagnose_saved_interval_whole_group_coverage(doc)
        self.assertEqual(doc, frozen)
        self.assertEqual(evaluate_saved_intervals(doc), default_before)
        self.assertNotIn("whole_group", default_before)
        doc["rows"].reverse()
        doc["expected_row_ids"].reverse()
        self.assertEqual(diagnose_saved_interval_whole_group_coverage(doc), result)
