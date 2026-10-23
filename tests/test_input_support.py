"""Synthetic source-level diagnostics, no-fit plans and inference parity checks."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import benchmark, fit_predict, plan_benchmark, plan_prediction
from groupbench.cli import main
from groupbench.data import validate_features
from groupbench.input_support import prediction_input_support
from groupbench.model import make_pipeline
from groupbench.verification import _SAFE_OPEN_SUPPORTED, verify_bundle


def diagnostics(training, new):
    return prediction_input_support(validate_features(training), new)


def fixture():
    x = np.arange(24, dtype=float)
    return pd.DataFrame({"batch": np.repeat(["001", "b", "c", "d", "e", "f"], 4),
                         "x": x, "kind": np.where(x % 2, "A", "B"),
                         "target": 2 * x + np.sin(x), "unused": x * -3})


class SourceSupportTests(unittest.TestCase):
    def test_numeric_boundaries_missing_and_constant_feature(self):
        train = pd.DataFrame({"x": [1., 3., np.nan], "constant": [2., 2., 2.]})
        new = pd.DataFrame({"constant": [2., 1., 3., 2., np.nan], "x": [1., 3., 0., 4., np.nan]})
        result = diagnostics(train, new)
        x, constant = result["features"]
        self.assertEqual(x, {"name": "x", "kind": "numeric", "training_missing_count": 1,
                             "training_support": "observed_non_missing_values",
                             "prediction_missing_count": 1,
                             "prediction_non_missing_without_training_values_count": 0,
                             "below_training_min_count": 1, "above_training_max_count": 1})
        self.assertEqual(constant["below_training_min_count"], 1)
        self.assertEqual(constant["above_training_max_count"], 1)
        self.assertEqual(result["rows_with_any_flag"], 4)
        self.assertEqual(result["row_flag_counts"], {
            "missing_value": 1, "numeric_outside_training_range": 3,
            "unseen_category": 0, "non_missing_value_in_all_missing_training_feature": 0})

    def test_query_values_never_expand_numeric_or_categorical_reference(self):
        train = pd.DataFrame({"x": [0., 1.], "kind": ["A", "B"]})
        new = pd.DataFrame({"x": [2., 3., 1e100], "kind": ["new", "new", "other"]})
        result = diagnostics(train, new)
        self.assertEqual(result["features"][0]["above_training_max_count"], 3)
        self.assertEqual(result["features"][1]["unseen_category_count"], 3)
        singleton = diagnostics(train, new.iloc[:1])
        self.assertEqual(singleton["features"][0]["above_training_max_count"], 1)
        self.assertEqual(singleton["features"][1]["unseen_category_count"], 1)
        # Adding observed training values, not query extrema, updates support.
        expanded = diagnostics(pd.concat([train, new.iloc[:1]]), new)
        self.assertEqual(expanded["features"][0]["above_training_max_count"], 2)
        self.assertEqual(expanded["features"][1]["unseen_category_count"], 1)

    def test_extreme_finite_numeric_values_need_no_range_arithmetic(self):
        largest = np.finfo(float).max
        train = pd.DataFrame({"x": [-largest, largest], "constant": [largest, largest]})
        new = pd.DataFrame({"x": [-largest, 0., largest], "constant": [-largest, 0., largest]})
        with np.errstate(all="raise"):
            result = diagnostics(train, new)
        self.assertEqual(result["features"][0]["below_training_min_count"], 0)
        self.assertEqual(result["features"][0]["above_training_max_count"], 0)
        self.assertEqual(result["features"][1]["below_training_min_count"], 2)
        json.dumps(result, allow_nan=False)

    def test_all_missing_training_is_explicitly_unavailable(self):
        train = pd.DataFrame({"x": [np.nan, np.nan], "kind": pd.Series([None, None], dtype=object)})
        new = pd.DataFrame({"x": [1., np.nan, -1.], "kind": ["new", None, "__GROUPBENCH_MISSING__"]})
        result = diagnostics(train, new)
        x, kind = result["features"]
        for feature in (x, kind):
            self.assertEqual(feature["training_support"], "unavailable_all_missing")
            self.assertEqual(feature["training_missing_count"], 2)
            self.assertEqual(feature["prediction_missing_count"], 1)
            self.assertEqual(feature["prediction_non_missing_without_training_values_count"], 2)
        self.assertIsNone(x["below_training_min_count"])
        self.assertIsNone(x["above_training_max_count"])
        self.assertEqual(kind["unseen_category_count"], 2)
        self.assertEqual(result["row_flag_counts"]["numeric_outside_training_range"], 0)
        self.assertEqual(result["row_flag_counts"]["non_missing_value_in_all_missing_training_feature"], 2)
        self.assertEqual(result["rows_with_any_flag"], 3)

    def test_literal_missing_marker_is_a_source_category_not_imputation(self):
        marker = "__GROUPBENCH_MISSING__"
        new = pd.DataFrame({"kind": [marker, None, "A", "never-seen"]})
        result = diagnostics(pd.DataFrame({"kind": ["A", None]}), new)
        self.assertEqual(result["features"][0]["unseen_category_count"], 2)
        literal_seen = diagnostics(pd.DataFrame({"kind": ["A", marker]}), new)
        self.assertEqual(literal_seen["features"][0]["unseen_category_count"], 1)
        self.assertEqual(literal_seen["features"][0]["prediction_missing_count"], 1)

    def test_stateless_category_normalization_and_all_missing_query(self):
        train = pd.DataFrame({"kind": pd.Series([True, 7, None], dtype=object), "flag": [True, False, True]})
        new = pd.DataFrame({"flag": ["True", "False", "unknown"], "kind": ["True", "7", "unseen"]})
        result = diagnostics(train, new)
        self.assertEqual([f["unseen_category_count"] for f in result["features"]], [1, 1])
        missing = diagnostics(train, pd.DataFrame({"flag": [np.nan], "kind": [np.nan]}))
        self.assertEqual([f["prediction_missing_count"] for f in missing["features"]], [1, 1])
        self.assertEqual([f["unseen_category_count"] for f in missing["features"]], [0, 0])

    def test_large_category_inventory_is_neither_collapsed_nor_reported(self):
        train = pd.DataFrame({"kind": [f"private-category-{i}" for i in range(20_000)]})
        new = pd.DataFrame({"kind": ["private-category-0", "private-category-19999", "new-secret"]})
        result = diagnostics(train, new)
        self.assertEqual(result["features"][0]["unseen_category_count"], 1)
        encoded = json.dumps(result, allow_nan=False)
        self.assertNotIn("private-category", encoded)
        self.assertNotIn("new-secret", encoded)
        self.assertLess(len(encoded), 1500)
        self.assertEqual(len(result["features"]), 1)

    def test_no_mutation_index_alignment_or_silent_schema_relaxation(self):
        train = pd.DataFrame({"x": [1., 2.], "kind": ["A", "B"]}, index=[99, 99])
        new = pd.DataFrame({"kind": ["unseen", "B"], "x": [3., 2.]}, index=[-7, -7])
        before_train, before_new = train.copy(deep=True), new.copy(deep=True)
        result = diagnostics(train, new)
        self.assertEqual(result["rows_with_any_flag"], 1)
        pd.testing.assert_frame_equal(train, before_train)
        pd.testing.assert_frame_equal(new, before_new)
        for invalid in (new.assign(extra=1), new.drop(columns="x"), new.assign(x=["bad", "type"]),
                        new.assign(x=[np.inf, 1]), pd.concat([new, new[["x"]]], axis=1)):
            with self.subTest(columns=list(invalid)), self.assertRaises(ValueError):
                diagnostics(train, invalid)


class PredictionSupportIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.training = fixture()
        self.new = pd.DataFrame({"kind": ["A", "unseen", None], "x": [2., 30., np.nan]}, index=[18, 18, -5])
        self.options = dict(target="target", group="batch", features=["x", "kind"], inner_splits=2, random_state=11)

    def test_diagnostics_do_not_change_predictions_selection_preprocessing_or_fit_schedule(self):
        calls = []
        def factory(X, candidate, random_state):
            pipeline = make_pipeline(X, candidate, random_state)
            original_fit = pipeline.fit
            def counted_fit(frame, y, **kwargs):
                calls.append((candidate.name, random_state, frame.index.tolist(), frame.to_json()))
                return original_fit(frame, y, **kwargs)
            pipeline.fit = counted_fit
            return pipeline
        with patch("groupbench.model.make_pipeline", side_effect=factory):
            actual = fit_predict(self.training, self.new, **self.options)
        actual_calls = calls.copy()
        calls.clear()
        with patch("groupbench.model.make_pipeline", side_effect=factory), patch(
            "groupbench.prediction.prediction_input_support", return_value={"not_computed": True}):
            without_diagnostics = fit_predict(self.training, self.new, **self.options)
        self.assertEqual(actual_calls, calls)
        self.assertEqual(len(calls), 15)  # 6 candidates * 2 folds + 3 family refits.
        pd.testing.assert_frame_equal(actual.predictions, without_diagnostics.predictions, check_exact=True)
        for key in ("selected_candidate", "selected_family", "inner_selection", "inner_folds", "config",
                    "training_data_sha256", "new_features_sha256", "feature_schema", "data_diagnostics"):
            self.assertEqual(actual.manifest[key], without_diagnostics.manifest[key])
        for family in actual.estimator.family_models_:
            np.testing.assert_array_equal(actual.estimator.family_models_[family].predict(
                self.new[["x", "kind"]].replace({None: np.nan})),
                without_diagnostics.estimator.family_models_[family].predict(
                self.new[["x", "kind"]].replace({None: np.nan})))
        self.assertEqual(actual.predictions.columns.tolist(), ["row_position", "prediction"])
        self.assertEqual(actual.predictions.row_position.tolist(), [0, 1, 2])

    def test_no_fit_plan_matches_inference_and_uses_only_allowed_training_features(self):
        with patch("groupbench.model.make_pipeline", side_effect=AssertionError("No fitting")):
            plan = plan_prediction(self.training, self.new, **self.options)
        result = fit_predict(self.training, self.new, **self.options, baselines=("dummy",))
        self.assertEqual(plan["input_support"], result.manifest["input_support"])
        self.assertEqual(plan["schema_version"], "1.1")
        self.assertEqual(result.manifest["schema_version"], "1.1")
        self.assertEqual([f["name"] for f in plan["input_support"]["features"]], ["x", "kind"])
        changed = self.training.copy()
        changed["target"] = changed.target.iloc[::-1].to_numpy()
        changed["batch"] = changed.batch.iloc[::-1].to_numpy()
        changed["unused"] = -1e100
        other = plan_prediction(changed, self.new, **self.options)
        self.assertEqual(plan["input_support"], other["input_support"])

    def test_evaluation_schema_and_behavior_are_not_extended(self):
        plan = plan_benchmark(self.training, **self.options, baselines=("dummy",), outer_splits=3)
        result = benchmark(self.training, **self.options, baselines=("dummy",), outer_splits=3)
        for report in (plan, result.manifest, result.summary):
            self.assertEqual(report["schema_version"], "1.0")
            self.assertNotIn("input_support", report)

    def test_all_missing_training_and_new_columns_still_predict(self):
        train = self.training.copy()
        train["x"] = np.nan
        train["kind"] = pd.Series([None] * len(train), dtype=object)
        result = fit_predict(train, self.new, **self.options, baselines=("dummy", "ridge"))
        self.assertTrue(np.isfinite(result.predictions.prediction).all())
        self.assertEqual(result.manifest["input_support"]["row_flag_counts"][
            "non_missing_value_in_all_missing_training_feature"], 2)
        entirely_missing = self.new.assign(x=np.nan, kind=np.nan)
        result = fit_predict(train, entirely_missing, **self.options, baselines=("dummy",))
        self.assertEqual(result.manifest["input_support"]["row_flag_counts"]["missing_value"], 3)
        self.assertEqual(result.manifest["input_support"]["row_flag_counts"][
            "non_missing_value_in_all_missing_training_feature"], 0)

    def test_cli_bundle_and_plan_share_diagnostics_without_new_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, new, out = root / "train.csv", root / "new.csv", root / "output"
            self.training.to_csv(train, index=False)
            self.new.to_csv(new, index=False)
            args = [str(train), str(new), "--target", "target", "--group", "batch", "--features", "x,kind",
                    "--inner-splits", "2", "--seed", "11", "--baselines", "dummy"]
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["plan", "predict", *args])
            self.assertEqual(code, 0, stderr.getvalue())
            plan = json.loads(stdout.getvalue())
            self.assertFalse(out.exists())
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = main(["predict", *args, "--out", str(out)])
            self.assertEqual(code, 0, stderr.getvalue())
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(plan["input_support"], manifest["input_support"])
            self.assertEqual({p.name for p in out.iterdir()}, {"manifest.json", "predictions.csv", "checksums.json"})
            self.assertEqual(pd.read_csv(out / "predictions.csv").columns.tolist(), ["row_position", "prediction"])
            if _SAFE_OPEN_SUPPORTED:
                self.assertEqual(verify_bundle(out.resolve()).bundle_type, "prediction")


if __name__ == "__main__":
    unittest.main()
