"""Standard-library unittest suite: no additional test dependency needed."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from groupbench import GroupBenchRegressor, GroupKFold, benchmark, make_materials_fixture
from groupbench.data import data_fingerprint, duplicate_diagnostics, prepare_data, validate_features, validate_groups, validate_target
from groupbench.metrics import grouped_metrics, regression_metrics
from groupbench.model import Candidate, make_pipeline
from groupbench.report import render_html
from groupbench.verification import _SAFE_OPEN_SUPPORTED, verify_bundle


class DataTests(unittest.TestCase):
    def test_fixture_reproducible_and_synthetic(self):
        pd.testing.assert_frame_equal(make_materials_fixture(), make_materials_fixture())
        self.assertEqual(make_materials_fixture().shape, (288, 6))
        self.assertEqual(make_materials_fixture()["batch"].nunique(), 24)

    def test_invalid_targets_are_rejected(self):
        for values in ([1, np.nan], [1, np.inf], [[1], [2]], [1], [1 + 2j, 2 + 3j], ["a", "b"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_target(values, 2)

    def test_bad_groups_are_rejected(self):
        for values in ([None, "a"], ["", "a"], [1, "1"], [True, False], [1, np.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_groups(values, 2)

    def test_numeric_group_equivalence_and_large_integer_identity(self):
        self.assertEqual(list(validate_groups([1, 1.0, 2.0], 3)), ["1", "1", "2"])
        self.assertNotEqual(*validate_groups([2**60, 2**60 + 1], 2))

    def test_bad_schema_rejected(self):
        with self.assertRaises(ValueError):
            validate_features(pd.DataFrame([[1, 2]], columns=["a", "a"]))
        with self.assertRaises(ValueError):
            validate_features(pd.DataFrame({"a": [np.inf]}))
        with self.assertRaises(ValueError):
            validate_features(pd.DataFrame({"date": pd.to_datetime(["2020-01-01"])}))
        with self.assertRaises(TypeError):
            validate_features(np.ones((2, 2)))

    def test_target_and_group_excluded(self):
        frame = make_materials_fixture(6, 3)
        X, _, _ = prepare_data(frame, "response", "batch")
        self.assertNotIn("response", X)
        self.assertNotIn("batch", X)
        for feature in ("response", "batch"):
            with self.assertRaises(ValueError):
                prepare_data(frame, "response", "batch", features=[feature])

    def test_fingerprint_sensitive_to_values_and_order(self):
        X = pd.DataFrame({"a": [1.0, 2.0]})
        y, groups = np.array([3, 4]), np.array(["a", "b"])
        original = data_fingerprint(X, y, groups)
        self.assertNotEqual(original, data_fingerprint(X, y + 1, groups))
        self.assertNotEqual(original, data_fingerprint(X.iloc[::-1], y, groups))
        self.assertEqual(original, data_fingerprint(X.copy(), y.copy(), groups.copy()))

    def test_duplicates_cross_groups_diagnostic(self):
        X = pd.DataFrame({"a": [1, 1, 3], "b": [np.nan, np.nan, "x"]})
        result = duplicate_diagnostics(X, np.array([2, 2, 9]), np.array(["a", "b", "b"]))
        self.assertEqual(result["identical_feature_target_rows_across_groups"], 2)
        self.assertEqual(result["cross_group_duplicate_patterns"], 1)
        self.assertEqual(result["action"], "diagnostic_only_no_rows_removed")


class SplitTests(unittest.TestCase):
    def test_exact_once_and_disjoint(self):
        groups = np.repeat(["a", "b", "c", "d", "e", "f"], [1, 3, 9, 2, 5, 2])
        splits = list(GroupKFold(3, 17).split(groups=groups))
        seen = []
        for train, test in splits:
            self.assertFalse(set(groups[train]) & set(groups[test]))
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(train) + len(test), len(groups))
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(len(groups))))

    def test_seed_is_deterministic(self):
        groups = np.repeat(np.arange(20), 3)
        a = list(GroupKFold(5, 7).split(groups=groups))
        b = list(GroupKFold(5, 7).split(groups=groups))
        c = list(GroupKFold(5, 8).split(groups=groups))
        for (ta, va), (tb, vb) in zip(a, b):
            np.testing.assert_array_equal(ta, tb)
            np.testing.assert_array_equal(va, vb)
        self.assertTrue(any(not np.array_equal(x[1], y[1]) for x, y in zip(a, c)))

    def test_impossible_split_is_error(self):
        for splitter, groups in ((GroupKFold(3), ["a", "b"]), (GroupKFold(1), ["a", "b"]), (GroupKFold(2, -1), ["a", "b"])):
            with self.assertRaises(ValueError):
                list(splitter.split(groups=groups))
        with self.assertRaises(ValueError):
            list(GroupKFold(2).split(X=[1, 2]))


class MetricTests(unittest.TestCase):
    def test_macro_weights_groups_equally(self):
        scores, per_group = grouped_metrics([0, 0, 0, 0], [0, 0, 0, 4], ["a", "a", "a", "b"])
        self.assertEqual(scores["mae"], 1)
        self.assertEqual(scores["group_mae"], 2)
        self.assertEqual(scores["rmse"], 2)
        self.assertIsNone(scores["r2"])
        self.assertEqual(per_group[1]["bias"], 4)

    def test_undefined_r2_serializes_as_null(self):
        self.assertIsNone(regression_metrics([1], [2])["r2"])
        json.dumps(regression_metrics([1, 1], [2, 2]), allow_nan=False)

    def test_nonfinite_prediction_rejected(self):
        with self.assertRaises(ValueError):
            regression_metrics([1, 2], [1, np.nan])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = make_materials_fixture(9, 4)
        cls.X, cls.y, cls.groups = prepare_data(cls.frame, "response", "batch")

    def test_train_only_imputation_and_unknown_categories(self):
        train = pd.DataFrame({"numeric": [1.0, 3.0, np.nan], "kind": ["A", "B", "A"]})
        holdout = pd.DataFrame({"numeric": [1e9], "kind": ["UNSEEN"]})
        pipe = make_pipeline(train, Candidate("ridge", "ridge", {"alpha": 1.0}), 0)
        pipe.fit(train, [1, 2, 3])
        imputer = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        encoder = pipe.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"]
        self.assertEqual(imputer.statistics_[0], 2)
        self.assertNotIn("UNSEEN", encoder.categories_[0])
        self.assertTrue(np.isfinite(pipe.predict(holdout)).all())
        self.assertEqual(imputer.statistics_[0], 2)

    def test_all_missing_features_supported(self):
        X = pd.DataFrame({"numeric": [np.nan] * 12, "kind": [None] * 12})
        est = GroupBenchRegressor(baselines=("ridge",), inner_splits=2).fit(X, np.arange(12), groups=np.repeat(np.arange(4), 3))
        self.assertTrue(np.isfinite(est.predict(X)).all())

    def test_clone_fit_predict_and_schema(self):
        original = GroupBenchRegressor(baselines=("dummy", "ridge"), inner_splits=3)
        est = clone(original).fit(self.X, self.y, groups=self.groups)
        self.assertEqual(est.n_features_in_, 4)
        np.testing.assert_allclose(est.predict(self.X), est.predict(self.X[self.X.columns[::-1]]))
        with self.assertRaises(ValueError):
            est.predict(self.X.drop(columns=[self.X.columns[0]]))
        changed = self.X.copy()
        changed[changed.columns[0]] = changed.iloc[:, 0].astype(str)
        with self.assertRaises(ValueError):
            est.predict(changed)
        with self.assertRaises(NotFittedError):
            original.predict(self.X)
        with self.assertRaises(ValueError):
            original.fit(self.X, self.y)

    def test_deterministic_selection_and_predictions(self):
        a = GroupBenchRegressor(random_state=4, inner_splits=2).fit(self.X, self.y, groups=self.groups)
        b = GroupBenchRegressor(random_state=4, inner_splits=2).fit(self.X, self.y, groups=self.groups)
        self.assertEqual(a.selection_results_, b.selection_results_)
        np.testing.assert_array_equal(a.predict(self.X), b.predict(self.X))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = make_materials_fixture(9, 4, random_state=11)
        cls.result = benchmark(cls.frame, target="response", group="batch", outer_splits=3, inner_splits=2, random_state=5)

    def test_nested_audit_and_exact_predictions(self):
        result = self.result
        self.assertEqual(len(result.predictions), len(self.frame))
        self.assertEqual(result.predictions.row_position.tolist(), list(range(len(self.frame))))
        np.testing.assert_array_equal(result.predictions.observed, self.frame.response)
        self.assertTrue(np.isfinite(result.predictions.filter(like="prediction_").to_numpy()).all())
        for fold in result.manifest["outer_folds"]:
            outer_test = set(fold["test_groups"])
            outer_train = set(fold["train_groups"])
            self.assertFalse(outer_train & outer_test)
            for inner in fold["inner_folds"]:
                self.assertFalse(set(inner["train_groups"]) & set(inner["test_groups"]))
                self.assertFalse(outer_test & (set(inner["train_groups"]) | set(inner["test_groups"])))
                self.assertEqual(outer_train, set(inner["train_groups"]) | set(inner["test_groups"]))
            rows = result.predictions.outer_fold == fold["fold"]
            np.testing.assert_array_equal(result.predictions.loc[rows, "prediction_selected"], result.predictions.loc[rows, "prediction_" + fold["selected_family"]])
            expected = min(fold["inner_selection"], key=lambda row: row["selection_score"])["candidate"]
            self.assertEqual(expected, fold["selected_candidate"])

    def test_outer_targets_do_not_affect_their_predictions(self):
        fold = self.result.manifest["outer_folds"][0]
        changed = self.frame.copy()
        mask = changed.batch.isin(fold["test_groups"])
        changed.loc[mask, "response"] += 1e6
        rerun = benchmark(changed, target="response", group="batch", outer_splits=3, inner_splits=2, random_state=5)
        for col in self.result.predictions.filter(like="prediction_"):
            np.testing.assert_array_equal(self.result.predictions.loc[mask, col], rerun.predictions.loc[mask, col])
        self.assertEqual(fold["selected_candidate"], rerun.manifest["outer_folds"][0]["selected_candidate"])

    def test_safe_strict_report_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.result.write(directory)
            self.assertTrue(path.exists())
            for filename in ("summary.json", "manifest.json"):
                json.loads(Path(directory, filename).read_text(), parse_constant=lambda val: self.fail(f"Non-finite {val}"))
            checks = json.loads(Path(directory, "checksums.json").read_text())
            for filename, digest in checks["files"].items():
                self.assertEqual(hashlib.sha256(Path(directory, filename).read_bytes()).hexdigest(), digest)
            if _SAFE_OPEN_SUPPORTED:
                self.assertEqual(verify_bundle(Path(directory).resolve()).bundle_type, "evaluation")
            with self.assertRaises(FileExistsError):
                self.result.write(directory)
            self.result.write(directory, overwrite=True)
            html = path.read_text()
            self.assertIn("Content-Security-Policy", html)
            self.assertNotIn("<script", html)
            self.assertNotIn('src="http', html)

    def test_html_escapes_group_and_target(self):
        frame = self.frame.rename(columns={"response": '<script>alert("target")</script>'}).copy()
        frame["batch"] = frame["batch"] + '<img src=x onerror="alert(1)">'
        result = benchmark(frame, target='<script>alert("target")</script>', group="batch", outer_splits=3, inner_splits=2, baselines=("dummy",))
        html = render_html(result)
        self.assertNotIn('<script>alert', html)
        self.assertNotIn('<img src=x', html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img", html)

    def test_impossible_inner_fails_before_any_fit(self):
        with patch("groupbench.model.Pipeline.fit") as fit:
            with self.assertRaises(ValueError):
                benchmark(make_materials_fixture(4, 3), target="response", group="batch", outer_splits=2, inner_splits=3)
            fit.assert_not_called()

    def test_cli_csv_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory, "input.csv")
            self.frame.to_csv(csv_path, index=False)
            command = [sys.executable, "-m", "groupbench", "run", str(csv_path), "--target", "response", "--group", "batch", "--out", str(Path(directory, "out")), "--outer-splits", "3", "--inner-splits", "2", "--baselines", "dummy,ridge"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Offline report", completed.stdout)
            manifest = json.loads(Path(directory, "out", "manifest.json").read_text())
            self.assertEqual(manifest["source_file_sha256"], hashlib.sha256(csv_path.read_bytes()).hexdigest())
            repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("Output already contains", repeated.stderr)

    def test_duplicate_csv_headers_error(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory, "bad.csv")
            csv_path.write_text("a,a,response,batch\n1,2,3,x\n")
            command = [sys.executable, "-m", "groupbench", "run", str(csv_path), "--target", "response", "--group", "batch", "--out", str(Path(directory, "out"))]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unique column names", result.stderr)


if __name__ == "__main__":
    unittest.main()
