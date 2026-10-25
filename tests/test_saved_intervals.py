"""Saved synthetic intervals only: independent arithmetic and no-fit checks."""
import copy
from decimal import Decimal
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from groupbench.interval_evaluation import evaluate_saved_intervals, load_document

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/saved-interval-evaluation/invented_voltage.json"


def fixture():
    return load_document(FIXTURE.read_text())


def single(y=0, lower=0, upper=0, alpha=None):
    doc = fixture()
    doc.update(alpha=alpha, expected_row_ids=["a1"], rows=[doc["rows"][0]])
    doc["rows"][0].update(observed=y, lower=lower, upper=upper)
    return doc


class SavedIntervalTests(unittest.TestCase):
    def test_hand_oracle_same_cohort_and_units(self):
        report = evaluate_saved_intervals(fixture())
        self.assertEqual(report["row_weighted"], {"n_rows": 4, "n_covered": 2,
                         "empirical_coverage_approx": 0.5, "mean_width_approx": 1.5,
                         "mean_interval_score_approx": 29})
        self.assertEqual(report["equal_group"], {"n_groups": 2, "empirical_coverage_approx": 1/3,
                         "mean_width_approx": 1, "mean_interval_score_approx": 158/3})
        self.assertEqual([r["interval_score_approx"] for r in report["observations"]], [2, 2, 12, 100])
        self.assertEqual(report["cohort"], {"expected_rows": 4, "input_rows": 4, "scored_rows": 4, "omitted_rows": 0})
        self.assertEqual(report["nominal_coverage_exact"], "4/5")
        self.assertEqual(report["metadata"]["value_unit"], "V")
        self.assertFalse(report["provenance_verified"])

    def test_closed_endpoints_degenerate_and_both_penalties(self):
        for y, lower, upper, hit, width, score in [(0, 0, 2, True, 2, 2), (2, 0, 2, True, 2, 2),
                (1, 1, 1, True, 0, 0), (0, 1, 1, False, 0, 10), (-1, 0, 2, False, 2, 12), (3, 0, 2, False, 2, 12)]:
            with self.subTest(y=y, lower=lower, upper=upper):
                report = evaluate_saved_intervals(single(y, lower, upper, Decimal(".2")))
                row = report["observations"][0]
                self.assertEqual((row["covered"], row["width_approx"], row["interval_score_approx"]), (hit, width, score))

    def test_no_alpha_means_no_score_or_nominal(self):
        report = evaluate_saved_intervals(single(1, 0, 2))
        for key in ("alpha_exact", "alpha_approx", "nominal_coverage_exact", "nominal_coverage_approx"):
            self.assertIsNone(report[key])
        for row in [report["row_weighted"], report["equal_group"], *report["groups"]]:
            self.assertIsNone(row["mean_interval_score_approx"])
        self.assertIsNone(report["observations"][0]["interval_score_approx"])

    def test_one_group_and_singleton_groups_agree_with_rows(self):
        for groups in (["A"]*4, ["A", "B", "C", "D"]):
            doc = fixture()
            for row, group in zip(doc["rows"], groups): row["group"] = group
            report = evaluate_saved_intervals(doc)
            for key in ("empirical_coverage_approx", "mean_width_approx", "mean_interval_score_approx"):
                self.assertEqual(report["equal_group"][key], report["row_weighted"][key])

    def test_duplicate_entire_group_does_not_reweight_equal_groups(self):
        doc = fixture(); original = evaluate_saved_intervals(doc)
        for row in copy.deepcopy(doc["rows"][:3]):
            row["row_id"] += "-repeat"
            doc["expected_row_ids"].append(row["row_id"]); doc["rows"].append(row)
        report = evaluate_saved_intervals(doc)
        self.assertEqual(report["equal_group"], original["equal_group"])
        self.assertNotEqual(report["row_weighted"], original["row_weighted"])

    def test_order_invariance_and_no_mutation(self):
        doc = fixture(); before = copy.deepcopy(doc)
        result = evaluate_saved_intervals(doc)
        self.assertEqual(doc, before)
        doc["rows"].reverse(); doc["expected_row_ids"].reverse()
        self.assertEqual(evaluate_saved_intervals(doc), result)

    def test_translation_and_positive_unit_scaling(self):
        doc = fixture(); original = evaluate_saved_intervals(doc)
        for row in doc["rows"]:
            for key in ("observed", "lower", "upper"): row[key] = 7*row[key]+10
        report = evaluate_saved_intervals(doc)
        for summary in ("row_weighted", "equal_group"):
            self.assertEqual(report[summary]["empirical_coverage_approx"], original[summary]["empirical_coverage_approx"])
            for key in ("mean_width_approx", "mean_interval_score_approx"):
                self.assertAlmostEqual(report[summary][key], 7*original[summary][key])

    def test_large_numeric_group_identity_and_float_alias(self):
        doc = fixture()
        for row, group in zip(doc["rows"], [2**60, 2**60, 2**60+1, 0.5]): row["group"] = group
        report = evaluate_saved_intervals(doc)
        self.assertEqual(report["equal_group"]["n_groups"], 3)
        self.assertEqual(report["equal_group"]["empirical_coverage_approx"], 1/3)
        self.assertEqual({r["group"] for r in report["groups"]}, {str(2**60), str(2**60+1), "0.5"})
        for row, group in zip(doc["rows"], [1, 1.0, -0.0, 0]): row["group"] = group
        self.assertEqual(evaluate_saved_intervals(doc)["equal_group"]["n_groups"], 2)

    def test_numeric_groups_from_json_preserve_large_integers(self):
        doc = fixture()
        for row, group in zip(doc["rows"], [2**60, 2**60, 2**60+1, Decimal("1.0")]): row["group"] = group
        self.assertEqual(evaluate_saved_intervals(doc)["equal_group"]["n_groups"], 3)
        doc["rows"][0]["group"] = Decimal("0.1")
        with self.assertRaisesRegex(ValueError, "text IDs"): evaluate_saved_intervals(doc)

    def test_exact_decimal_boundary_and_already_lost_float_boundary(self):
        report = evaluate_saved_intervals(single(Decimal(2**60+1), 2**60, 2**60))
        self.assertFalse(report["observations"][0]["covered"])
        self.assertEqual(report["observations"][0]["observed_exact"], str(2**60+1))
        self.assertTrue(evaluate_saved_intervals(single(float(2**60+1), float(2**60), float(2**60)))["observations"][0]["covered"])
        # JSON decimal .1 and Python's already represented float .1 differ.
        report = evaluate_saved_intervals(single(0.1, Decimal(".1"), Decimal(".1")))
        self.assertFalse(report["observations"][0]["covered"])
        self.assertNotEqual(report["observations"][0]["observed_exact"], report["observations"][0]["lower_exact"])

    def test_tiny_alpha_finite_score_without_float_intermediate_overflow(self):
        report = evaluate_saved_intervals(single(Decimal("1e-310"), 0, 0, Decimal("1e-310")))
        self.assertEqual(report["row_weighted"]["mean_interval_score_approx"], 2)
        self.assertEqual(report["nominal_coverage_approx"], 1)
        self.assertLess(Fraction(report["nominal_coverage_exact"]), 1)

    def test_alpha_near_one_has_explicit_exact_companion(self):
        alpha = Decimal("0." + "9"*99)
        report = evaluate_saved_intervals(single(0, 0, 0, alpha))
        self.assertEqual(report["alpha_approx"], 1)
        self.assertLess(Fraction(report["alpha_exact"]), 1)
        self.assertGreater(report["nominal_coverage_approx"], 0)

    def test_input_output_overflow_and_underflow_reject(self):
        for doc in [single(Decimal("1e309")), single(Decimal("1e-400")),
                    single(0, -1e308, 1e308), single(1e308, 0, 0, Decimal(".1"))]:
            with self.subTest(doc=repr(doc)[:100]):
                with self.assertRaises(ValueError): evaluate_saved_intervals(doc)
        # Per-row widths are representable, but their positive mean rounds to 0.
        doc = fixture(); doc["alpha"] = None
        for row in doc["rows"]: row.update(observed=0, lower=0, upper=0, group="A")
        doc["rows"][0]["upper"] = math.ulp(0.0)
        with self.assertRaisesRegex(ValueError, "underflows"): evaluate_saved_intervals(doc)

    def test_invalid_numbers_reject_without_partial_scores(self):
        for key in ("observed", "lower", "upper"):
            for value in (None, True, "1", complex(1), float("nan"), float("inf"), Decimal("sNaN"), Decimal("Infinity"), np.float32(1)):
                doc = fixture(); doc["rows"][-1][key] = value
                with self.subTest(key=key, value=repr(value)):
                    with self.assertRaises(ValueError): evaluate_saved_intervals(doc)

    def test_invalid_alpha_and_reversed_bounds(self):
        for alpha in (0, 1, -1, 2, True, "0.1", [], float("nan")):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError): evaluate_saved_intervals(single(alpha=alpha))
        with self.assertRaisesRegex(ValueError, "Reversed"): evaluate_saved_intervals(single(1, 2, 0))

    def test_cohort_missing_extra_duplicate_failure(self):
        mutations = [lambda d: d["rows"].pop(), lambda d: d["rows"].append(copy.deepcopy(d["rows"][0])),
                     lambda d: d["expected_row_ids"].append("missing"), lambda d: d["expected_row_ids"].append("a1"),
                     lambda d: d["rows"][0].update(row_id="extra"), lambda d: d["rows"][0].update(status="failed"),
                     lambda d: d["rows"][0].pop("observed")]
        for mutate in mutations:
            doc = fixture(); mutate(doc)
            with self.assertRaises(ValueError): evaluate_saved_intervals(doc)

    def test_schema_labels_and_counts_are_bounded(self):
        for key, value in (("schema_version", True), ("schema_version", 2), ("rows", []),
                           ("rows", [None]*20001), ("expected_row_ids", ["a"]*20001), ("metadata", {})):
            doc = fixture(); doc[key] = value
            with self.assertRaises(ValueError): evaluate_saved_intervals(doc)
        for value in ("", " ", "a\x00", "a"*257, 1):
            doc = fixture(); doc["expected_row_ids"][0] = value
            with self.assertRaises(ValueError): evaluate_saved_intervals(doc)
        doc = fixture(); doc["metadata"]["value_unit"] = "x"*2049
        with self.assertRaises(ValueError): evaluate_saved_intervals(doc)
        doc = fixture(); doc["unknown"] = True
        with self.assertRaises(ValueError): evaluate_saved_intervals(doc)

    def test_numeric_limits_before_expansion(self):
        for value in (10**100, Decimal("1e999999999"), Decimal("1e-999999999"), Decimal("1."+"1"*100)):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError): evaluate_saved_intervals(single(value))
                doc = fixture()
                for row in doc["rows"]: row["group"] = value
                with self.assertRaises(ValueError): evaluate_saved_intervals(doc)
        self.assertEqual(evaluate_saved_intervals(single(10**99))["cohort"]["scored_rows"], 1)

    def test_invalid_groups_and_private_label_not_echoed(self):
        for value in (True, None, float("inf"), "", " ", "PRIVATE\x00SECRET", "x"*1025, ["a"], Decimal("sNaN")):
            doc = fixture(); doc["rows"][0]["group"] = value
            with self.assertRaises(ValueError) as error: evaluate_saved_intervals(doc)
            self.assertNotIn("PRIVATE", str(error.exception))
        doc = fixture(); doc["rows"][0]["group"] = 1
        with self.assertRaisesRegex(ValueError, "mix"): evaluate_saved_intervals(doc)

    def test_text_group_identity_not_trimmed_or_numeric_coerced(self):
        doc = fixture()
        for row, group in zip(doc["rows"], ["1", "01", " 1", "1 "]): row["group"] = group
        self.assertEqual(evaluate_saved_intervals(doc)["equal_group"]["n_groups"], 4)

    def test_fit_and_predict_are_never_called(self):
        with patch("groupbench.model.GroupBenchRegressor.fit", side_effect=AssertionError("fit called")), \
             patch("groupbench.model.GroupBenchRegressor.predict", side_effect=AssertionError("predict called")), \
             patch("groupbench.binary.model.GroupBenchClassifier.fit", side_effect=AssertionError("fit called")):
            self.assertEqual(evaluate_saved_intervals(fixture())["cohort"]["scored_rows"], 4)

    def test_strict_json_loader(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '['*20000+'0'+']'*20000, ' '*8000001):
            with self.assertRaises(ValueError): load_document(text)

    def test_json_numeric_tokens_bounded_before_conversion(self):
        for text in ('{"x":'+'1'*101+'}', '{"x":0.'+'1'*1025+'}',
                     '{"x":1e999999999}', '{"x":1e-401}', '{"x":1.'+'1'*100+'}'):
            with self.assertRaises(ValueError): load_document(text)
        # The utility's bound works even if Python's unrelated int guard is off.
        limit = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else None
        try:
            if limit is not None: sys.set_int_max_str_digits(0)
            with self.assertRaisesRegex(ValueError, "100 decimal digits"):
                load_document('{"x":'+'1'*10000+'}')
            with self.assertRaises(ValueError): evaluate_saved_intervals(single(10**10000))
        finally:
            if limit is not None: sys.set_int_max_str_digits(limit)

    def test_group_ratio_bounds_and_numpy_identity(self):
        doc = fixture()
        for row, group in zip(doc["rows"], [np.float32(.1), float(np.float32(.1)), .1, 1]):
            row["group"] = group
        self.assertEqual(evaluate_saved_intervals(doc)["equal_group"]["n_groups"], 3)
        with np.errstate(over="ignore"):
            extreme = np.longdouble(2)**5000
        for row in doc["rows"]: row["group"] = extreme
        with self.assertRaises(ValueError): evaluate_saved_intervals(doc)


class SavedIntervalCliTests(unittest.TestCase):
    def run_cli(self, path):
        return subprocess.run([sys.executable, "-m", "groupbench.interval_evaluation", str(path)],
                              text=True, capture_output=True, timeout=15)

    def test_stdout_report_preserves_input_no_new_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "input.json"); source.write_bytes(FIXTURE.read_bytes())
            before = source.read_bytes(); result = self.run_cli(source)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), evaluate_saved_intervals(fixture()))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [source])

    def test_errors_produce_no_partial_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "input.json")
            for text in ('{"x":NaN}', FIXTURE.read_text().replace('"ok"', '"failed"', 1), '{bad', '\ud800'):
                source.write_bytes(text.encode("utf-8", errors="surrogatepass"))
                result = self.run_cli(source)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("rejected", result.stderr)
            self.assertEqual(self.run_cli(directory).returncode, 2)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "named pipes")
    def test_nonregular_rejected_without_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "fifo"); os.mkfifo(source)
            result = self.run_cli(source)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")

    @unittest.skipUnless(os.name == "posix", "symlinks")
    def test_explicit_regular_symlink_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "input.json"); source.write_bytes(FIXTURE.read_bytes())
            link = Path(directory, "link.json"); link.symlink_to(source)
            direct, linked = self.run_cli(source), self.run_cli(link)
            self.assertEqual(direct.returncode, 0, direct.stderr)
            self.assertEqual(linked.returncode, 0, linked.stderr)
            self.assertEqual(direct.stdout, linked.stdout)


if __name__ == "__main__":
    unittest.main()
