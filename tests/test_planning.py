"""Synthetic no-fit planning, shared-protocol parity and no-write regressions."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
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
from groupbench.model import Candidate, candidates_for, make_pipeline


def fixture():
    """Unequal invented groups expose both row counts and seeded assignments."""
    groups = np.repeat(["001", "NA", "c", "d", "e", "f", "g"], [2, 4, 3, 5, 2, 3, 4])
    x = np.arange(len(groups), dtype=float)
    return pd.DataFrame({"batch": groups, "x": x, "kind": np.where(x % 2, "A", "B"),
                         "target": 2 * x + np.sin(x), "unused": x * -3})


def counts(audit):
    return {key: value for key, value in audit.items()
            if key in ("fold", "train_rows", "test_rows", "train_group_count", "test_group_count", "overlap_count")}


def no_fitting(stack):
    for target in ("groupbench.model.GroupBenchRegressor.fit",
                   "groupbench.model.GroupBenchRegressor._fit_prepared",
                   "groupbench.model.make_pipeline", "groupbench.model.Pipeline.fit",
                   "groupbench.model.DummyRegressor.fit", "groupbench.model.Ridge.fit",
                   "groupbench.model.ExtraTreesRegressor.fit",
                   "groupbench.report.write_payloads", "groupbench.report.write_report"):
        stack.enter_context(patch(target, side_effect=AssertionError(f"Plan called {target}")))


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.new = pd.DataFrame({"kind": ["A", "unseen"], "x": [2.5, 8.5]})
        self.options = dict(target="target", group="batch", features=["x", "kind"],
                            inner_splits=2, random_state=11)

    def test_default_budgets_and_candidate_grids_without_fitting(self):
        with ExitStack() as stack:
            no_fitting(stack)
            evaluation = plan_benchmark(self.frame, target="target", group="batch")
            prediction = plan_prediction(self.frame, self.new, target="target", group="batch", features=["x", "kind"])
        self.assertEqual(evaluation["fit_budget"], {
            "unit": "pipeline.fit calls", "inner_selection_fits": 90,
            "outer_family_refits": 15, "full_data_family_refits": 0,
            "selected_extra_refits": 0, "total_fits": 105})
        self.assertEqual(prediction["fit_budget"]["inner_selection_fits"], 18)
        self.assertEqual(prediction["fit_budget"]["full_data_family_refits"], 3)
        self.assertEqual(prediction["fit_budget"]["total_fits"], 21)
        expected = [{"name": c.name, "family": c.family, "params": c.params}
                    for c in candidates_for(("dummy", "ridge", "extra_trees"))]
        self.assertEqual(evaluation["candidates"], expected)
        self.assertEqual(prediction["candidates"], expected)
        self.assertEqual([c["params"] for c in expected], [
            {"strategy": "mean"}, {"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0},
            {"max_depth": 6, "n_estimators": 64, "min_samples_leaf": 2},
            {"max_depth": None, "n_estimators": 64, "min_samples_leaf": 2}])
        for result in (evaluation, prediction):
            self.assertEqual(result["fits_performed"], 0)
            self.assertNotIn("train_groups", json.dumps(result))
            self.assertNotIn("test_groups", json.dumps(result))
            json.dumps(result, allow_nan=False)

    def _count_actual_fits(self, action):
        calls = []
        def factory(X, candidate, random_state):
            pipeline = make_pipeline(X, candidate, random_state)
            original_fit = pipeline.fit
            def counted_fit(X_fit, y_fit, **kwargs):
                calls.append((candidate.name, random_state, len(X_fit)))
                return original_fit(X_fit, y_fit, **kwargs)
            pipeline.fit = counted_fit
            return pipeline
        with patch("groupbench.model.make_pipeline", side_effect=factory):
            result = action()
        return result, calls

    def test_run_plan_matches_every_actual_fit_and_nested_split(self):
        for families in (("dummy",), ("ridge",), ("extra_trees",), ("ridge", "dummy", "extra_trees")):
            for objective in ("group_mae", "mae"):
                with self.subTest(families=families, objective=objective):
                    options = {**self.options, "baselines": families, "outer_splits": 3, "objective": objective}
                    plan = plan_benchmark(self.frame, **options)
                    result, calls = self._count_actual_fits(lambda: benchmark(self.frame, **options))
                    expected = []
                    for planned, actual in zip(plan["outer_folds"], result.manifest["outer_folds"]):
                        self.assertEqual(counts(planned), counts(actual))
                        self.assertEqual(planned["inner_seed"], actual["inner_seed"])
                        self.assertEqual(planned["inner_folds"], [counts(f) for f in actual["inner_folds"]])
                        for candidate in plan["candidates"]:
                            expected.extend((candidate["name"], planned["inner_seed"], f["train_rows"])
                                            for f in planned["inner_folds"])
                        expected.extend((actual["family_candidates"][family], planned["inner_seed"], planned["train_rows"])
                                        for family in planned["refit_families"])
                    self.assertEqual(calls, expected)
                    self.assertEqual(len(calls), plan["fit_budget"]["total_fits"])
                    self.assertEqual(plan["fit_budget"]["outer_family_refits"], len(families) * len(plan["outer_folds"]))
                    self.assertEqual(plan["fit_budget"]["selected_extra_refits"], 0)

    def test_predict_plan_matches_every_actual_fit_and_split(self):
        for families in (("dummy",), ("ridge",), ("extra_trees",), ("extra_trees", "dummy", "ridge")):
            with self.subTest(families=families):
                options = {**self.options, "baselines": families}
                plan = plan_prediction(self.frame, self.new, **options)
                result, calls = self._count_actual_fits(lambda: fit_predict(self.frame, self.new, **options))
                selection = plan["selection"]
                self.assertEqual(selection["inner_folds"], [counts(f) for f in result.manifest["inner_folds"]])
                expected = [(c["name"], selection["inner_seed"], fold["train_rows"])
                            for c in plan["candidates"] for fold in selection["inner_folds"]]
                expected.extend((result.estimator.family_candidates_[family], selection["inner_seed"], len(self.frame))
                                for family in selection["refit_families"])
                self.assertEqual(calls, expected)
                self.assertEqual(len(calls), plan["fit_budget"]["total_fits"])
                self.assertEqual(plan["fit_budget"]["full_data_family_refits"], len(families))
                self.assertEqual(plan["fit_budget"]["outer_family_refits"], 0)

    def test_changed_candidate_grid_changes_plan_and_execution_together(self):
        original = candidates_for
        def expanded(families):
            return original(families) + [Candidate("ridge_alpha_100", "ridge", {"alpha": 100.0})]
        with patch("groupbench.model.candidates_for", side_effect=expanded):
            plan = plan_prediction(self.frame, self.new, **self.options, baselines=("ridge",))
            _, calls = self._count_actual_fits(lambda: fit_predict(self.frame, self.new, **self.options, baselines=("ridge",)))
        self.assertEqual(plan["candidates"][-1]["params"], {"alpha": 100.0})
        self.assertEqual(len(calls), plan["fit_budget"]["total_fits"])
        self.assertEqual(sum(name == "ridge_alpha_100" for name, _, _ in calls[:8]), 2)

    def test_impossible_protocols_are_rejected_without_any_fits(self):
        invalid = [dict(outer_splits=8), dict(outer_splits=7, inner_splits=7),
                   dict(outer_splits=1), dict(inner_splits=1), dict(inner_splits=True),
                   dict(random_state=-1), dict(objective="rmse"), dict(baselines=()),
                   dict(baselines=("ridge", "ridge")), dict(baselines=("unknown",))]
        with ExitStack() as stack:
            no_fitting(stack)
            for override in invalid:
                with self.subTest(override=override), self.assertRaises(ValueError):
                    plan_benchmark(self.frame, **{**self.options, **override})
            for override in (dict(inner_splits=8), dict(inner_splits=1), dict(random_state=-1),
                             dict(objective="rmse"), dict(baselines=("unknown",))):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    plan_prediction(self.frame, self.new, **{**self.options, **override})
            too_few = self.frame[self.frame.batch.isin(["001", "NA", "c"])]
            with self.assertRaisesRegex(ValueError, "distinct groups"):
                plan_benchmark(too_few, **self.options, outer_splits=2)

    def test_data_and_allowlist_errors_are_shared(self):
        bad_target = self.frame.copy()
        bad_target.loc[0, "target"] = np.nan
        bad_group = self.frame.copy()
        bad_group.loc[0, "batch"] = " "
        invalid = [(bad_target, {}), (bad_group, {}),
                   (self.frame, {"features": ["target"]}), (self.frame, {"features": ["batch"]}),
                   (self.frame, {"features": ["x", "x"]}), (self.frame, {"features": ["absent"]}),
                   (self.frame, {"features": []}), (self.frame, {"target": "batch"})]
        with ExitStack() as stack:
            no_fitting(stack)
            for frame, override in invalid:
                options = {**self.options, **override}
                for action in (lambda: plan_benchmark(frame, **options, outer_splits=3),
                               lambda: plan_prediction(frame, self.new, **options)):
                    with self.subTest(override=override), self.assertRaises(ValueError):
                        action()

    def test_new_row_schema_and_types_match_prediction_before_fit(self):
        invalid = [self.new.assign(target=1), self.new.assign(batch="g"), self.new.drop(columns="x"),
                   self.new.assign(x=["bad", "text"]), self.new.assign(kind=[1, 2]),
                   self.new.assign(x=[np.inf, 1]), self.new.iloc[:0]]
        with ExitStack() as stack:
            no_fitting(stack)
            for new in invalid:
                errors = []
                for action in (plan_prediction, fit_predict):
                    with self.subTest(columns=list(new), action=action.__name__), self.assertRaises(ValueError) as exc:
                        action(self.frame, new, **self.options)
                    errors.append(str(exc.exception))
                self.assertEqual(errors[0], errors[1])
            for new in (self.new[self.new.columns[::-1]], self.new.assign(kind=np.nan), self.new.assign(x=np.nan)):
                result = plan_prediction(self.frame, new, **self.options)
                self.assertEqual(result["n_prediction_rows"], 2)

    def test_same_seed_is_deterministic_and_targets_do_not_define_plan(self):
        first = plan_benchmark(self.frame, **self.options, outer_splits=3)
        changed = self.frame.copy()
        changed["target"] += 1000
        self.assertEqual(first, plan_benchmark(changed, **self.options, outer_splits=3))
        self.assertEqual(first, plan_benchmark(self.frame, **self.options, outer_splits=3))


class PlanningCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.training = self.root / "predictions.csv"
        self.new = self.root / "new.csv"
        fixture().to_csv(self.training, index=False)
        pd.DataFrame({"kind": ["A", "unseen"], "x": [2.5, 8.5]}).to_csv(self.new, index=False)
        self.schema = self.root / "feature_schema.json"
        self.schema.write_text(json.dumps({
            "schema_version": 1, "row_key": ["batch"], "model_feature_allowlist": ["x", "kind"],
            "metadata_columns": ["batch", "unused"], "target_columns": ["target"],
            "feature_units": {"x": "arbitrary", "kind": "category"}, "warnings": []}))
        for name in ("manifest.json", "summary.json", "report.html", "checksums.json"):
            (self.root / name).write_text("PRESERVE EXISTING " + name)
        (self.root / "existing-directory").mkdir()
        self.args = ["--target", "target", "--group", "batch", "--inner-splits", "2", "--seed", "11"]

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() if p.is_file() else None
                for p in self.root.rglob("*")}

    def invoke(self, mode, *extra):
        args = ["plan", mode, str(self.training)]
        if mode == "predict":
            args.append(str(self.new))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main([*args, *self.args, *extra])
        return status, out.getvalue(), err.getvalue()

    def test_both_modes_no_fitting_or_output_clobber_and_hash_exact_bytes(self):
        before = self.snapshot()
        with ExitStack() as stack:
            no_fitting(stack)
            for mode in ("run", "predict"):
                status, out, err = self.invoke(mode, "--features-file", str(self.schema))
                self.assertEqual(status, 0, err)
                result = json.loads(out)
                self.assertEqual(result["config"]["features"], ["x", "kind"])
                self.assertEqual(result["source_file_sha256"], hashlib.sha256(self.training.read_bytes()).hexdigest())
                self.assertEqual(result["n_training_groups"], 7)
                if mode == "predict":
                    self.assertEqual(result["prediction_source_file_sha256"], hashlib.sha256(self.new.read_bytes()).hexdigest())
                self.assertEqual(self.snapshot(), before)

    def test_allowlist_variants_and_same_defaults(self):
        for mode in ("run", "predict"):
            status, text, err = self.invoke(mode, "--features", "x,kind")
            self.assertEqual(status, 0, err)
            self.assertEqual(json.loads(text)["config"]["features"], ["x", "kind"])
        status, text, err = self.invoke("run")
        self.assertEqual(status, 0, err)
        self.assertEqual(json.loads(text)["config"]["features"], ["x", "kind", "unused"])

    def test_plan_does_not_accept_any_output_flags(self):
        before = self.snapshot()
        for mode in ("run", "predict"):
            for flag in (["--out", str(self.root)], ["--out", "2"], ["--out=2"], ["--overwrite"]):
                with self.subTest(mode=mode, flag=flag), self.assertRaises(SystemExit) as exc:
                    self.invoke(mode, *flag)
                self.assertEqual(exc.exception.code, 2)
        self.assertEqual(self.snapshot(), before)

    def test_malformed_csv_and_invalid_features_fail_without_output(self):
        with ExitStack() as stack:
            no_fitting(stack)
            for mode in ("run", "predict"):
                for flags in (["--features", "target"], ["--features", "batch"],
                              ["--features", "unknown"], ["--baselines", "bad"],
                              ["--inner-splits", "99"]):
                    before = self.snapshot()
                    status, out, err = self.invoke(mode, *flags)
                    self.assertEqual(status, 2, err)
                    self.assertEqual(out, "")
                    self.assertEqual(self.snapshot(), before)
            self.training.write_text("batch,x,target\n001,1,2,EXTRA\n")
            before = self.snapshot()
            for mode in ("run", "predict"):
                status, out, err = self.invoke(mode)
                self.assertEqual(status, 2)
                self.assertIn("fields", err)
                self.assertEqual(out, "")
                self.assertEqual(self.snapshot(), before)

    def test_invalid_feature_file_and_conflicting_allowlists(self):
        self.schema.write_text('{"schema_version": 2}')
        before = self.snapshot()
        with ExitStack() as stack:
            no_fitting(stack)
            for mode in ("run", "predict"):
                status, out, _ = self.invoke(mode, "--features-file", str(self.schema))
                self.assertEqual(status, 2)
                self.assertEqual(out, "")
                with self.assertRaises(SystemExit) as exc:
                    self.invoke(mode, "--features-file", str(self.schema), "--features", "x")
                self.assertEqual(exc.exception.code, 2)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
