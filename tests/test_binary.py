"""Original synthetic binary tests; no claim of real predictive performance."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss

from groupbench import (GroupBenchClassifier, benchmark, fit_predict, benchmark_binary,
    fit_predict_binary, plan_binary_benchmark, plan_binary_prediction, make_binary_fixture)
from groupbench.binary.data import validate_binary_target
from groupbench.binary.metrics import binary_metrics, grouped_binary_metrics, checked_probabilities, _mean, EPSILON
from groupbench.binary.model import binary_pipeline, binary_candidates
from groupbench.binary.report import ensure_fresh_binary_output, render_binary_html
from groupbench.cli import main
from groupbench.verification import verify_bundle, _SAFE_OPEN_SUPPORTED


def fixture():
    n = 32
    return pd.DataFrame({"lot": np.repeat([f"g{i}" for i in range(8)], 4),
        "x": np.arange(n, dtype=float), "kind": np.tile(["a", "b", "a", "b"], 8),
        "empty": np.full(n, np.nan), "fail": np.tile([0, 1, 1, 0], 8)})


def options(**overrides):
    return dict(target="fail", group="lot", inner_splits=2, random_state=3,
                **overrides)


def no_fit(stack):
    for name in ("groupbench.binary.model.binary_pipeline", "groupbench.binary.model._fit_pipeline",
                 "groupbench.binary.model.GroupBenchClassifier._fit_prepared",
                 "groupbench.binary.model.Pipeline.fit", "groupbench.binary.model.SimpleImputer.fit",
                 "groupbench.report.write_payloads"):
        stack.enter_context(patch(name, side_effect=AssertionError(f"No-fit path called {name}")))


class BinaryLabelMetricTests(unittest.TestCase):
    def test_strict_original_labels(self):
        invalid = [[0, True], [False, 1], [False, True], ["0", "1"], [-1, 1], [0, 0.9],
                   [0, np.nan], [0, np.inf], [0, complex(1)], [0, None], [[0], [1]],
                   np.array([0, 1], dtype=object), pd.Series([0, 1], dtype=object), [0, 0], [1, 1],
                   [0, 10**1000], np.array([0, np.nextafter(np.longdouble(1), np.longdouble(2))]),
                   np.array([np.nextafter(np.longdouble(0), np.longdouble(1)), 1])]
        for y in invalid:
            with self.subTest(y=repr(y)), self.assertRaises(ValueError):
                validate_binary_target(y, 2)
        for y in ([0, 1.0], np.array([0, 1]), np.array([0., 1.]), pd.Series([0, 1], dtype="Int64")):
            np.testing.assert_array_equal(validate_binary_target(y, 2), [0, 1])
        np.testing.assert_array_equal(validate_binary_target([0, 0], 2, require_both=False), [0, 0])

    def test_exact_unequal_groups_and_single_class_denominators(self):
        overall, groups = grouped_binary_metrics([0, 0, 0, 1], [0., 0., 0., 0.], ["A", "A", "A", "B"])
        self.assertEqual(overall["brier"], .25)
        self.assertEqual(overall["group_brier"], .5)
        self.assertEqual(overall["worst_group_brier"], 1)
        self.assertEqual(overall["confusion"], {"tn": 3, "fp": 0, "fn": 1, "tp": 0})
        self.assertIsNone(overall["precision_1"])
        self.assertIsNone(groups[0]["recall_1"])
        self.assertIsNone(groups[1]["specificity_0"])
        self.assertTrue(all(g["balanced_accuracy"] is None for g in groups))
        self.assertEqual(overall["probability_zero_count"], 4)
        self.assertEqual(overall["probability_below_log_epsilon_count"], 4)
        json.dumps(overall, allow_nan=False)

    def test_probability_oracles_and_no_mutation(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0., .5, .8, 1.])
        original = p.copy()
        result = binary_metrics(y, p)
        self.assertAlmostEqual(result["brier"], brier_score_loss(y, p, pos_label=1))
        self.assertAlmostEqual(result["log_loss_clipped"], log_loss(y, p, labels=[0, 1]))
        self.assertEqual(result["confusion"], {"tn": 1, "fp": 1, "fn": 0, "tp": 2})
        np.testing.assert_array_equal(p, original)
        self.assertEqual(binary_metrics([0, 1], [0., 1.])["brier"], 0)
        self.assertEqual(binary_metrics([0, 1], [1., 0.])["brier"], 1)
        invalid = [[np.nan, 0], [np.inf, 0], [-1e-10, 1], [0, 1.0000001], [[0], [1]], [],
                   [.5+.25j, .5-.25j]]
        if np.dtype(np.longdouble).itemsize > np.dtype(np.float64).itemsize:
            invalid.append(np.array([0, 1], dtype=np.longdouble))
        for bad in invalid:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                binary_metrics([0, 1], bad)
        good = np.array([[.6, .4], [.5, .5]])
        np.testing.assert_array_equal(checked_probabilities(good, 2, [0, 1]), good)
        for values, classes in ((good, [1, 0]), (good[:, :1], [0, 1]), (good * 2, [0, 1]),
                                (np.array([[np.nan, 1], [0, 1]]), [0, 1])):
            with self.assertRaises(ValueError):
                checked_probabilities(values, 2, classes)

    def test_tiny_positive_losses_fail_only_when_rounded_to_zero(self):
        with self.assertRaisesRegex(ValueError, "scoring resolution"):
            binary_metrics([0], [1e-300])
        self.assertGreater(binary_metrics([0], [1e-160])["brier"], 0)
        smallest = np.nextafter(0., 1.)
        with self.assertRaisesRegex(ValueError, "scoring resolution"):
            _mean([smallest, 0.])
        p = np.sqrt(smallest)
        self.assertEqual(binary_metrics([0], [p])["brier"], smallest)
        # Overall mean retains a tiny positive value but equal-group mean does not.
        with self.assertRaisesRegex(ValueError, "scoring resolution"):
            grouped_binary_metrics([0] * 101, [p] * 100 + [0.], ["A"] * 100 + ["B"])


class BinaryPlanSelectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.new = self.frame.drop(columns=["lot", "fail"]).iloc[:3].copy()

    def test_default_counts_and_no_construction_or_write(self):
        with ExitStack() as stack:
            no_fit(stack)
            a = plan_binary_benchmark(self.frame, target="fail", group="lot")
            b = plan_binary_prediction(self.frame, self.new, target="fail", group="lot")
        self.assertEqual(a["fit_budget"]["total_fits"], 105)
        self.assertEqual(b["fit_budget"]["total_fits"], 21)
        for plan in (a, b):
            self.assertEqual(plan["fits_performed"], 0)
            self.assertEqual(plan["schema_id"], "mlcalc.binary.execution_plan")
            self.assertEqual(plan["label_contract"]["classes"], [0, 1])
            self.assertNotIn("train_groups", json.dumps(plan))
            self.assertNotIn("test_groups", json.dumps(plan))
            self.assertEqual(len(plan["config"]["candidates"]), 6)
        self.assertEqual(a["config"]["candidates"][4]["params"]["max_leaf_nodes"], 256)

    def test_all_schedule_class_guards_before_first_fit(self):
        frame = self.frame.copy()
        frame["fail"] = 0
        frame.loc[frame.lot == "g0", "fail"] = 1
        with ExitStack() as stack:
            no_fit(stack)
            for action in (benchmark_binary, plan_binary_benchmark):
                with self.assertRaisesRegex(ValueError, "lacks class"):
                    action(frame, **options(outer_splits=3, baselines=("dummy",)))
            for action in (fit_predict_binary, plan_binary_prediction):
                with self.assertRaisesRegex(ValueError, "lacks class"):
                    action(frame, self.new, **options(baselines=("dummy",)))
        mono = pd.concat([self.frame] * 4, ignore_index=True)
        mono["lot"] = np.repeat([f"mono-{i}" for i in range(32)], 4)
        mono["fail"] = np.repeat(np.arange(32) % 2, 4)
        result = benchmark_binary(mono, **options(outer_splits=4, baselines=("dummy",)))
        self.assertEqual(len(result.summary["models"]["selected"]["per_group"]), 32)
        self.assertTrue(all(row["balanced_accuracy"] is None for row in result.summary["models"]["selected"]["per_group"]))

    def test_invalid_protocols_before_fit(self):
        cases = [{"outer_splits": 11}, {"outer_splits": True}, {"outer_splits": 1},
                 {"inner_splits": 11}, {"inner_splits": True}, {"random_state": -1},
                 {"baselines": ()}, {"baselines": ("ridge",)}, {"baselines": ("dummy", "dummy")},
                 {"baselines": {"dummy", "logistic"}}, {"baselines": frozenset(["dummy"])},
                 {"baselines": {"dummy": 1}}, {"baselines": iter(["dummy"])}]
        with ExitStack() as stack:
            no_fit(stack)
            for overrides in cases:
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    plan_binary_benchmark(self.frame, **{**options(), **overrides})

    def _instrument(self, action):
        calls = []
        def factory(X, candidate, seed):
            pipeline = binary_pipeline(X, candidate, seed)
            fit = pipeline.fit
            def counted(X_fit, y_fit, **kwargs):
                calls.append((candidate.name, seed, len(X_fit)))
                return fit(X_fit, y_fit, **kwargs)
            pipeline.fit = counted
            return pipeline
        with patch("groupbench.binary.model.binary_pipeline", side_effect=factory):
            result = action()
        return result, calls

    def test_plan_execute_exact_fit_sequences_and_families(self):
        for families in (("dummy",), ("logistic",), ("extra_trees",), ("extra_trees", "dummy", "logistic")):
            with self.subTest(families=families):
                kwargs = options(baselines=families)
                plan = plan_binary_prediction(self.frame, self.new, **kwargs)
                result, calls = self._instrument(lambda: fit_predict_binary(self.frame, self.new, **kwargs))
                expected = [(c["name"], plan["selection"]["inner_seed"], f["train_rows"])
                            for c in plan["config"]["candidates"] for f in plan["selection"]["inner_folds"]]
                expected += [(result.estimator.family_candidates_[family], kwargs["random_state"], len(self.frame)) for family in families]
                self.assertEqual(calls, expected)
                self.assertEqual(len(calls), plan["fit_budget"]["total_fits"])
                self.assertEqual(plan["fit_budget"]["selected_extra_refits"], 0)
        kwargs = options(baselines=("dummy", "logistic"), outer_splits=3)
        plan = plan_binary_benchmark(self.frame, **kwargs)
        result, calls = self._instrument(lambda: benchmark_binary(self.frame, **kwargs))
        expected = []
        for planned, actual in zip(plan["outer_folds"], result.manifest["outer_folds"]):
            self.assertEqual(planned["train_rows"], actual["train_rows"])
            self.assertEqual(actual["overlap_count"], 0)
            expected.extend((c["name"], planned["inner_seed"], f["train_rows"])
                            for c in plan["config"]["candidates"] for f in planned["inner_folds"])
            expected.extend((actual["family_candidates"][family], planned["inner_seed"], planned["train_rows"])
                            for family in kwargs["baselines"])
            for f in actual["inner_folds"]:
                self.assertFalse(set(f["train_groups"]) & set(f["test_groups"]))
        self.assertEqual(calls, expected)
        self.assertEqual(len(calls), plan["fit_budget"]["total_fits"])

    def test_target_blind_assignments_and_outer_test_invariance(self):
        kwargs = options(baselines=("dummy", "logistic", "extra_trees"), outer_splits=3)
        first = benchmark_binary(self.frame, **kwargs)
        altered = self.frame.copy()
        fold = first.predictions.outer_fold.to_numpy() == 0
        altered.loc[fold, "fail"] = 1 - altered.loc[fold, "fail"]
        second = benchmark_binary(altered, **kwargs)
        np.testing.assert_array_equal(first.predictions.outer_fold, second.predictions.outer_fold)
        probability_cols = [c for c in first.predictions if c.startswith("probability_")]
        np.testing.assert_array_equal(first.predictions.loc[fold, probability_cols], second.predictions.loc[fold, probability_cols])
        for key in ("selected_candidate", "family_candidates", "inner_selection"):
            self.assertEqual(first.manifest["outer_folds"][0][key], second.manifest["outer_folds"][0][key])
        self.assertEqual(first.predictions.row_position.tolist(), list(range(len(self.frame))))

    def test_fit_state_failure_no_silent_nonconvergence(self):
        X = self.frame.drop(columns=["lot", "fail"])
        model = GroupBenchClassifier(baselines=("dummy",), inner_splits=2).fit(X, self.frame.fail, self.frame.lot)
        before = model.predict_proba(self.new).copy()
        old_best = model.best_estimator_
        def bad_fit(*args, **kwargs):
            warnings.warn("synthetic nonconvergence", ConvergenceWarning)
        with patch("groupbench.binary.model.Pipeline.fit", side_effect=bad_fit), self.assertRaisesRegex(ValueError, "ConvergenceWarning"):
            model.fit(X, self.frame.fail, self.frame.lot)
        self.assertIs(model.best_estimator_, old_best)
        np.testing.assert_array_equal(model.predict_proba(self.new), before)
        fresh = GroupBenchClassifier(baselines=("dummy",), inner_splits=2)
        with patch("groupbench.binary.model.Pipeline.fit", side_effect=bad_fit), self.assertRaises(ValueError):
            fresh.fit(X, self.frame.fail, self.frame.lot)
        self.assertFalse(hasattr(fresh, "best_estimator_"))

    def test_source_only_preprocessing_and_incoming_invariance(self):
        source = self.frame.copy()
        source.loc[0, "x"] = np.nan
        source.index = np.arange(len(source))[::-1]
        first = fit_predict_binary(source, self.new, **options(baselines=("logistic",)))
        changed = self.new.copy()
        changed["x"] = 10000
        changed["kind"] = "unseen-source-category"
        second = fit_predict_binary(source, changed, **options(baselines=("logistic",)))
        self.assertEqual(first.manifest["inner_selection"], second.manifest["inner_selection"])
        np.testing.assert_array_equal(first.estimator.best_estimator_.named_steps["model"].coef_, second.estimator.best_estimator_.named_steps["model"].coef_)
        transformer = first.estimator.best_estimator_.named_steps["preprocess"]
        self.assertEqual(transformer.named_transformers_["numeric"].named_steps["impute"].statistics_[0], source.x.median())
        categories = transformer.named_transformers_["categorical"].named_steps["encode"].categories_[0]
        self.assertNotIn("unseen-source-category", categories)
        plan = plan_binary_prediction(source, changed, **options(baselines=("logistic",)))
        self.assertEqual(plan["input_support"], second.manifest["input_support"])
        # All probabilities are exactly tied for this prior; explicit threshold chooses 1.
        tied = fit_predict_binary(self.frame, self.new, **options(baselines=("dummy",)))
        self.assertTrue(np.all(tied.estimator.predict(self.new) == 1))
        self.assertEqual(tied.estimator.classes_.tolist(), [0, 1])

    def test_prediction_schema_errors_before_training(self):
        for bad in (self.new.assign(fail=0), self.new.assign(lot="g"), self.new.drop(columns="x"),
                    self.new.assign(x="text"), self.new.assign(kind=3), self.new.assign(x=np.inf), self.new.iloc[:0]):
            with ExitStack() as stack:
                no_fit(stack)
                for action in (fit_predict_binary, plan_binary_prediction):
                    with self.assertRaises(ValueError):
                        action(self.frame, bad, **options())
        ordered = fit_predict_binary(self.frame, self.new[self.new.columns[::-1]], **options(baselines=("dummy",)))
        self.assertEqual(len(ordered.predictions), len(self.new))

    def test_regression_zero_one_keeps_old_contracts(self):
        old = benchmark(self.frame, **options(outer_splits=3, baselines=("dummy",)))
        self.assertEqual(old.summary["schema_version"], "1.0")
        self.assertEqual(set(old.summary), {"schema_version", "models", "selection_objective", "primary_entry", "limitations"})
        self.assertNotIn("task", old.manifest)
        self.assertIn("group_mae", old.summary["models"]["selected"]["overall"])
        self.assertEqual(old.predictions.columns.tolist(), ["row_position", "group", "outer_fold", "observed", "prediction_dummy", "prediction_selected"])
        predicted = fit_predict(self.frame, self.new, **options(baselines=("dummy",)))
        self.assertEqual(predicted.manifest["schema_version"], "1.1")
        self.assertEqual(predicted.predictions.columns.tolist(), ["row_position", "prediction"])


class BinaryArtifactCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frame = fixture()
        self.new = self.frame.drop(columns=["lot", "fail"]).iloc[:2].copy()
        self.train_path, self.new_path = self.root / "train.csv", self.root / "new.csv"
        self.frame.to_csv(self.train_path, index=False)
        self.new.to_csv(self.new_path, index=False)

    def invoke(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main(args)
        return status, out.getvalue(), err.getvalue()

    def test_raw_probability_roundtrip_and_bundle_verification(self):
        evaluation = benchmark_binary(self.frame, **options(outer_splits=3, baselines=("dummy", "logistic")))
        prediction = fit_predict_binary(self.frame, self.new, **options(baselines=("dummy",)))
        for name, result in (("evaluation", evaluation), ("prediction", prediction)):
            destination = self.root / name
            result.write(destination)
            roundtrip = pd.read_csv(destination / "predictions.csv", float_precision="round_trip")
            columns = [c for c in result.predictions if c.startswith("probability_")]
            np.testing.assert_array_equal(roundtrip[columns].to_numpy(), result.predictions[columns].to_numpy())
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["task"], "binary_classification")
            self.assertEqual(manifest["schema_version"], "1.0")
            if _SAFE_OPEN_SUPPORTED:
                self.assertEqual(verify_bundle(destination).bundle_type, name)
            with self.assertRaises(FileExistsError):
                result.write(destination)
        for name, metrics in evaluation.summary["models"].items():
            scores, groups = grouped_binary_metrics(evaluation.predictions.observed_label,
                evaluation.predictions[f"probability_1_{name}"], evaluation.predictions.group)
            self.assertEqual(metrics["overall"], scores)
            self.assertEqual(metrics["per_group"], groups)
        adverse = evaluation.summary["adverse_comparison_to_dummy"]
        self.assertEqual(adverse["groups_better"] + adverse["groups_tied"] + adverse["groups_worse"], 8)

    def test_fresh_output_guard_all_names_before_cli_fit(self):
        for name in ("report.html", "summary.json", "manifest.json", "predictions.csv", "checksums.json"):
            out = self.root / name.replace(".", "-")
            out.mkdir()
            (out / name).write_text("keep")
            with ExitStack() as stack:
                no_fit(stack)
                status, _, error = self.invoke(["binary", "run", str(self.train_path), "--target", "fail", "--group", "lot", "--out", str(out)])
            self.assertEqual(status, 2)
            self.assertIn("fresh", error)
            self.assertEqual((out / name).read_text(), "keep")
        out = self.root / "dangling"
        out.mkdir()
        (out / "summary.json").symlink_to(out / "missing")
        with self.assertRaises(FileExistsError):
            ensure_fresh_binary_output(out)

    def test_plan_cli_no_fit_write_csv_and_allowlist_hash(self):
        schema = self.root / "feature_schema.json"
        schema.write_text(json.dumps({"schema_version": 1, "row_key": ["lot"],
            "model_feature_allowlist": ["x", "kind", "empty"], "metadata_columns": ["lot"],
            "target_columns": ["fail"], "feature_units": {"x": "synthetic", "kind": "category", "empty": "synthetic"}, "warnings": []}))
        snapshot = {p.name: p.read_bytes() for p in self.root.iterdir()}
        for mode in ("run", "predict"):
            args = ["binary", "plan", mode, str(self.train_path)]
            if mode == "predict":
                args.append(str(self.new_path))
            args += ["--target", "fail", "--group", "lot", "--features-file", str(schema)]
            with ExitStack() as stack:
                no_fit(stack)
                status, out, err = self.invoke(args)
            self.assertEqual(status, 0, err)
            result = json.loads(out)
            self.assertEqual(result["source_file_sha256"], hashlib.sha256(self.train_path.read_bytes()).hexdigest())
            self.assertEqual(result["config"]["features"], ["x", "kind", "empty"])
            self.assertEqual(snapshot, {p.name: p.read_bytes() for p in self.root.iterdir()})
            for extra in (["--out", "2"], ["--overwrite"], ["--objective", "brier"], ["--threshold", ".4"]):
                with self.assertRaises(SystemExit):
                    self.invoke(args + extra)
        malformed = self.root / "malformed.csv"
        malformed.write_text("lot,x,fail\ng,1,0,extra\n")
        with ExitStack() as stack:
            no_fit(stack)
            status, out, err = self.invoke(["binary", "plan", "run", str(malformed), "--target", "fail", "--group", "lot"])
        self.assertEqual(status, 2)
        self.assertEqual(out, "")
        self.assertIn("fields", err)

    def test_size_preflight_html_escape_and_no_overwrite_argument(self):
        result = benchmark_binary(self.frame, **options(outer_splits=3, baselines=("dummy",)))
        result.manifest["config"]["target"] = '<script>alert("x")</script>'
        rendered = render_binary_html(result)
        self.assertNotIn('<script>', rendered)
        self.assertIn('&lt;script&gt;', rendered)
        self.assertIn("default-src 'none'", rendered)
        out = self.root / "too-large"
        with patch("groupbench.binary.report.MAX_BINARY_PAYLOAD_BYTES", 10), self.assertRaises(ValueError):
            result.write(out)
        self.assertFalse(out.exists())
        with self.assertRaises(TypeError):
            result.write(out, overwrite=True)
        result.summary["invalid"] = float("nan")
        with self.assertRaises(ValueError):
            result.write(out)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
