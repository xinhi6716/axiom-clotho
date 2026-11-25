"""Invented paired central intervals; no fit or calibration claim."""
from copy import deepcopy
from decimal import Decimal
import unittest

import numpy as np

from groupbench.interval_evaluation import (
    compare_saved_interval_scores,
    diagnose_saved_interval_whole_group_coverage,
)


def document(last, scale=1, offset=0):
    ids = [str(i) for i in range(10)]
    rows = [{"row_id": row_id, "group": "early" if i < 9 else "late", "status": "ok",
             "observed": offset + scale * int(i == 9),
             "lower": offset + scale * (last if i == 9 else 0),
             "upper": offset + scale * (last + 1 if i == 9 else 1)}
            for i, row_id in enumerate(ids)]
    return {"schema_version": 1, "alpha": Decimal("0.1"), "expected_row_ids": ids, "rows": rows,
            "metadata": {"target_definition": "toy", "value_unit": "toy unit",
                         "group_definition": "declared toy grouping", "cohort_definition": "frozen toy"}}


class PairedIntervalScoreTests(unittest.TestCase):
    def test_direct_python_status_must_be_literal_text(self):
        good = document(-1)
        for status in (np.array("ok"), np.array(["ok"]), np.array(["failed", "ok"])):
            bad = deepcopy(good)
            bad["rows"][0]["status"] = status
            with self.subTest(shape=status.shape):
                with self.assertRaisesRegex(ValueError, "non-ok row"):
                    diagnose_saved_interval_whole_group_coverage(bad)
                with self.assertRaisesRegex(ValueError, "non-ok row"):
                    compare_saved_interval_scores(good, bad)
        self.assertEqual(compare_saved_interval_scores(good, good)["n_rows"], 10)

    def test_same_coverage_and_width_unequal_miss_distance_and_score(self):
        a, b = document(-1), document(-10)
        before = deepcopy((a, b))
        results = compare_saved_interval_scores(a, b)
        self.assertEqual((a, b), before)
        score = results["row_weighted"]
        for arm, total, distance in (("a", "3", 0.1), ("b", "21", 1)):
            self.assertEqual(score[arm]["coverage_approx"], 0.9)
            self.assertEqual(score[arm]["width_approx"], 1)
            self.assertEqual(score[arm]["lower_miss_distance_approx"], 0)
            self.assertEqual(score[arm]["upper_miss_distance_approx"], distance)
            self.assertEqual(score[arm]["interval_score_exact"], total)
        self.assertEqual(score["b_minus_a"]["interval_score_exact"], "18")
        self.assertEqual(results["equal_group"]["b_minus_a"]["interval_score_exact"], "90")
        b["rows"].reverse()
        b["expected_row_ids"].reverse()
        self.assertEqual(compare_saved_interval_scores(a, b), results)

    def test_shared_positive_unit_transform(self):
        results = compare_saved_interval_scores(document(-1, 1000, 10), document(-10, 1000, 10))
        row = results["row_weighted"]
        self.assertEqual(row["a"]["coverage_approx"], 0.9)
        self.assertEqual(row["a"]["width_approx"], 1000)
        self.assertEqual(row["a"]["interval_score_exact"], "3000")
        self.assertEqual(row["b"]["interval_score_exact"], "21000")

    def test_aligned_contract_rejects_mismatch(self):
        a = document(-1)
        for case, error in (
            ("alpha", "nonnull exact alpha"), ("null", "nonnull exact alpha"),
            ("unit", "metadata"), ("target", "metadata"), ("cohort", "metadata"),
            ("group", "bindings"), ("observed", "exact observations"),
            ("roster", "roster"), ("missing", "Missing or extra"), ("duplicate", "Duplicate observed"),
            ("reversed", "Reversed interval"), ("failed", "non-ok"),
        ):
            b = document(-10)
            if case == "alpha":  # float and Decimal 0.1 differ exactly.
                b["alpha"] = 0.1
            elif case == "null":
                b["alpha"] = None
            elif case in ("unit", "target", "cohort"):
                b["metadata"][{"unit": "value_unit", "target": "target_definition",
                               "cohort": "cohort_definition"}[case]] = "other"
            elif case in ("group", "observed"):
                b["rows"][0][case] = "late" if case == "group" else 1
            elif case == "roster":
                b["expected_row_ids"][0] = b["rows"][0]["row_id"] = "other"
            elif case == "missing":
                b["rows"].pop()
            elif case == "duplicate":
                b["rows"][0]["row_id"] = b["rows"][1]["row_id"]
            elif case == "reversed":
                b["rows"][0]["lower"] = 2
            elif case == "failed":
                b["rows"][0]["status"] = "failed"
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, error):
                compare_saved_interval_scores(a, b)


    def test_same_display_group_labels_do_not_merge_text_and_numeric_namespaces(self):
        a, b = document(-1), document(-10)
        for i, (one, two) in enumerate(zip(a["rows"], b["rows"])):
            one["group"] = i
            two["group"] = str(i)
        with self.assertRaisesRegex(ValueError, "row-to-group bindings"):
            compare_saved_interval_scores(a, b)
