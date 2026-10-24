"""Synthetic contracts for explicit distinct-v2 and unchanged legacy defaults."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from groupbench import (GroupBenchRegressor, GroupBenchClassifier, benchmark, fit_predict,
                        plan_benchmark, plan_prediction, benchmark_binary, fit_predict_binary,
                        plan_binary_benchmark, plan_binary_prediction)
from groupbench.categorical import (LEGACY, DISTINCT, MISSING_MARKER as M,
                                    _distinct_category_keys)
from groupbench.cli import main
from groupbench.data import validate_features
from groupbench.model import candidates_for, make_pipeline
from groupbench.binary.model import binary_candidates, binary_pipeline


def dense(value):
    return value.toarray() if hasattr(value, "toarray") else value


def fixture(literal=True):
    return pd.DataFrame({"g": np.repeat([f"g{i}" for i in range(6)], 4),
                         "kind": [None, M if literal else "special", "plain", "plain"] * 6,
                         "y": [0, 1, 0, 1] * 6})


def new_rows():
    return pd.DataFrame({"kind": [None, M, "plain", "unseen"]})


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in ("created_at_utc", "elapsed_seconds")}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


class CategoryKeyTests(unittest.TestCase):
    def test_injection_and_order_with_unicode_controls_and_repeated_prefixes(self):
        alphabet = ["", "\x00", "\x01", " ", ":", "_", "a", "中文", "😀", "\ud800", "\U0010ffff"]
        values = sorted(set(alphabet + [M + suffix for suffix in alphabet] +
                            [M + ":" * n for n in range(5)] +
                            ["".join(parts) for parts in itertools.product(alphabet[1:7], repeat=2)]))
        array = np.array(values, dtype=object).reshape(-1, 1)
        encoded = _distinct_category_keys(array)[:, 0].tolist()
        self.assertEqual(len(encoded), len(set(encoded)))
        self.assertEqual(encoded, sorted(encoded))
        self.assertNotIn(M, encoded)
        for value, key in zip(values, encoded):
            self.assertEqual(key, M + ":" + value[len(M):] if value.startswith(M) else value)
            if value != M:
                self.assertEqual(value < M, key < M)
        self.assertEqual(array[:, 0].tolist(), values)

    def test_all_missing_normalizations_are_distinct_from_real_marker(self):
        X = validate_features(pd.DataFrame({"kind": [None, np.nan, pd.NA, M, M + ":", "nan", ""]}))
        keys = _distinct_category_keys(X)[:, 0].tolist()
        self.assertEqual(keys[:3], [M] * 3)
        self.assertEqual(keys[3:], [M + ":", M + "::", "nan", ""])

    def test_pipeline_feature_names_and_no_mutation(self):
        X = validate_features(pd.DataFrame({"kind": [None, M, "plain"] * 2}))
        original = X.copy(deep=True)
        for factory, candidate in [(make_pipeline, candidates_for(("ridge",))[0]),
                                   (binary_pipeline, binary_candidates(("logistic",))[0])]:
            pipe = factory(X, candidate, 0, DISTINCT).fit(X, np.arange(len(X)) % 2)
            encoded = pipe.named_steps["preprocess"].transform(X)
            self.assertEqual(len(pipe.named_steps["preprocess"].get_feature_names_out()), encoded.shape[1])
            pd.testing.assert_frame_equal(X, original)

    def test_below_cap_real_marker_and_missing_are_separate_in_both_tasks(self):
        X = validate_features(fixture()[["kind"]])
        for factory, candidates in [(make_pipeline, candidates_for(("ridge", "extra_trees"))),
                                    (binary_pipeline, binary_candidates(("logistic", "extra_trees")))]:
            for candidate in candidates:
                with self.subTest(factory=factory.__name__, candidate=candidate.name):
                    old = factory(X, candidate, 0).fit(X, fixture().y)
                    new = factory(X, candidate, 0, DISTINCT).fit(X, fixture().y)
                    old_rows = dense(old.named_steps["preprocess"].transform(X))
                    new_rows = dense(new.named_steps["preprocess"].transform(X))
                    np.testing.assert_array_equal(old_rows[0], old_rows[1])
                    self.assertFalse(np.array_equal(new_rows[0], new_rows[1]))

    def test_unobserved_literal_and_missing_remain_distinct_from_seen_other(self):
        query = validate_features(new_rows())
        for observed in ([None, "plain"], [M, "plain"], [None, None]):
            X = validate_features(pd.DataFrame({"kind": pd.Series(observed * 3, dtype=object)}))
            pipe = make_pipeline(X, candidates_for(("ridge",))[0], 0, DISTINCT).fit(X, np.arange(len(X)))
            encoded = dense(pipe.named_steps["preprocess"].transform(query))
            self.assertFalse(np.array_equal(encoded[0], encoded[1]))
            self.assertEqual(encoded[1 if observed[0] is None else 0].sum(), 0)

    def test_unknown_and_infrequent_compression_are_not_claimed_collision_free(self):
        X = validate_features(pd.DataFrame({"kind": ["known", "plain"] * 3}))
        pipe = make_pipeline(X, candidates_for(("ridge",))[0], 0, DISTINCT).fit(X, np.arange(len(X)))
        encoded = dense(pipe.named_steps["preprocess"].transform(validate_features(new_rows())))
        np.testing.assert_array_equal(encoded[0], encoded[1])
        self.assertEqual(encoded[0].sum(), 0)
        values = [None, M] + [f"ordinary-{i:02d}" for i in range(63)] * 2
        X = validate_features(pd.DataFrame({"kind": values}))
        pipe = make_pipeline(X, candidates_for(("ridge",))[0], 0, DISTINCT).fit(X, np.arange(len(X)))
        encoded = dense(pipe.named_steps["preprocess"].transform(X))
        self.assertEqual(encoded.shape[1], 64)
        np.testing.assert_array_equal(encoded[0], encoded[1])
        self.assertNotEqual(*_distinct_category_keys(X.iloc[:2])[:, 0])

    def test_ordinary_matrices_and_all_families_preserve_exact_cap_ties(self):
        cases = [[None, "plain", "X"], [None] + [f"lower-{i:02d}" for i in range(64)],
                 [None, M + ":", M + "\x00", M + M, "", "中文", "😀"], [None, None]]
        for values in cases:
            X = validate_features(pd.DataFrame({"kind": pd.Series(values * 2, dtype=object)}))
            for factory, candidates in [(make_pipeline, candidates_for(("dummy", "ridge", "extra_trees"))),
                                        (binary_pipeline, binary_candidates(("dummy", "logistic", "extra_trees")))]:
                for candidate in candidates:
                    with self.subTest(case=values[:2], factory=factory.__name__, candidate=candidate.name):
                        y = np.arange(len(X)) % 2
                        old = factory(X, candidate, 7).fit(X, y)
                        new = factory(X, candidate, 7, DISTINCT).fit(X, y)
                        np.testing.assert_array_equal(dense(old.named_steps["preprocess"].transform(X)),
                                                      dense(new.named_steps["preprocess"].transform(X)))
                        np.testing.assert_array_equal(old.predict(X), new.predict(X))
                        if factory is binary_pipeline:
                            np.testing.assert_array_equal(old.predict_proba(X), new.predict_proba(X))


class EncodingWorkflowTests(unittest.TestCase):
    def test_default_and_explicit_legacy_keep_existing_artifact_contracts(self):
        frame, new = fixture(), new_rows()
        for action in (benchmark, benchmark_binary, fit_predict, fit_predict_binary,
                       plan_benchmark, plan_binary_benchmark, plan_prediction, plan_binary_prediction):
            prediction = action in (fit_predict, fit_predict_binary, plan_prediction, plan_binary_prediction)
            args = (frame, new) if prediction else (frame,)
            opts = dict(target="y", group="g", inner_splits=2, baselines=("dummy",))
            if not prediction:
                opts["outer_splits"] = 3
            with self.subTest(action=action.__name__):
                old, explicit = action(*args, **opts), action(*args, **opts, categorical_encoding=LEGACY)
                if isinstance(old, dict):
                    self.assertEqual(old, explicit)
                    self.assertNotIn("categorical_encoding", old["config"])
                else:
                    self.assertEqual(stable(old.manifest), stable(explicit.manifest))
                    pd.testing.assert_frame_equal(old.predictions, explicit.predictions)
                    self.assertNotIn("categorical_encoding", old.manifest["config"])

    def test_opt_in_plans_match_execution_identity_and_fit_counts(self):
        frame, new = fixture(), new_rows()
        cases = [(benchmark, plan_benchmark, "groupbench.model.make_pipeline", make_pipeline, False),
                 (fit_predict, plan_prediction, "groupbench.model.make_pipeline", make_pipeline, True),
                 (benchmark_binary, plan_binary_benchmark, "groupbench.binary.model.binary_pipeline", binary_pipeline, False),
                 (fit_predict_binary, plan_binary_prediction, "groupbench.binary.model.binary_pipeline", binary_pipeline, True)]
        for action, planner, target, factory, prediction in cases:
            args = (frame, new) if prediction else (frame,)
            opts = dict(target="y", group="g", inner_splits=2, baselines=("dummy",), categorical_encoding=DISTINCT)
            if not prediction:
                opts["outer_splits"] = 3
            with self.subTest(action=action.__name__):
                with patch(target, side_effect=AssertionError("No construction during planning")):
                    plan = planner(*args, **opts)
                with patch(target, wraps=factory) as calls:
                    result = action(*args, **opts)
                self.assertEqual(calls.call_count, plan["fit_budget"]["total_fits"])
                self.assertEqual(plan["protocol"], result.manifest["protocol"])
                self.assertEqual(plan["config"]["categorical_encoding"], DISTINCT)
                self.assertEqual(result.manifest["config"]["categorical_encoding"], DISTINCT)
                for call in calls.call_args_list:
                    self.assertEqual(call.kwargs["categorical_encoding"], DISTINCT)

    def test_invalid_options_fail_without_constructing_a_model(self):
        frame, new = fixture(), new_rows()
        for value in (None, False, 2, "v2", "DISTINCT-V2", [], {}):
            for action in (benchmark, benchmark_binary, fit_predict, fit_predict_binary,
                           plan_benchmark, plan_binary_benchmark, plan_prediction, plan_binary_prediction):
                prediction = action in (fit_predict, fit_predict_binary, plan_prediction, plan_binary_prediction)
                args = (frame, new) if prediction else (frame,)
                options = dict(target="y", group="g", inner_splits=2, baselines=("dummy",), categorical_encoding=value)
                if not prediction:
                    options["outer_splits"] = 3
                with self.subTest(value=value, action=action.__name__), patch.object(Pipeline, "fit") as fitting:
                    with self.assertRaisesRegex(ValueError, "categorical_encoding"):
                        action(*args, **options)
                    fitting.assert_not_called()

    def test_fitted_protocol_survives_parameter_change_and_failed_refit(self):
        frame = fixture()
        X, y, groups = frame[["kind"]], frame.y, frame.g
        for cls, family, prediction in [(GroupBenchRegressor, "ridge", "predict"),
                                        (GroupBenchClassifier, "logistic", "predict_proba")]:
            with self.subTest(estimator=cls.__name__):
                est = clone(cls(baselines=(family,), inner_splits=2, categorical_encoding=DISTINCT)).fit(X, y, groups)
                before = getattr(est, prediction)(new_rows())
                self.assertFalse(np.array_equal(before[0], before[1]))
                est.set_params(categorical_encoding=LEGACY)
                np.testing.assert_array_equal(before, getattr(est, prediction)(new_rows()))
                est.set_params(categorical_encoding="invalid")
                with self.assertRaisesRegex(ValueError, "categorical_encoding"):
                    est.fit(X, y, groups)
                self.assertEqual(est.categorical_encoding_, DISTINCT)
                np.testing.assert_array_equal(before, getattr(est, prediction)(new_rows()))
                est.set_params(categorical_encoding=LEGACY).fit(X, y, groups)
                self.assertEqual(est.categorical_encoding_, LEGACY)
                after = getattr(est, prediction)(new_rows())
                np.testing.assert_array_equal(after[0], after[1])

    def test_ordinary_nested_predictions_unchanged_but_run_keys_distinguish_protocol(self):
        frame = fixture(literal=False)
        for action, baselines in [(benchmark, ("dummy", "ridge", "extra_trees")),
                                   (benchmark_binary, ("dummy", "logistic", "extra_trees"))]:
            opts = dict(target="y", group="g", inner_splits=2, outer_splits=3, baselines=baselines)
            old, new = action(frame, **opts), action(frame, **opts, categorical_encoding=DISTINCT)
            pd.testing.assert_frame_equal(old.predictions, new.predictions)
            self.assertEqual(old.manifest["outer_folds"], new.manifest["outer_folds"])
            self.assertNotEqual(old.manifest["run_key"], new.manifest["run_key"])

    def test_outer_targets_cannot_change_their_v2_predictions_or_selection(self):
        frame = fixture()
        for action, family in [(benchmark, "ridge"), (benchmark_binary, "logistic")]:
            opts = dict(target="y", group="g", inner_splits=2, outer_splits=3, baselines=(family,), categorical_encoding=DISTINCT)
            before = action(frame, **opts)
            rows = before.predictions.outer_fold == 0
            changed = frame.copy()
            changed.loc[rows.to_numpy(), "y"] = 1 - changed.loc[rows.to_numpy(), "y"]
            after = action(changed, **opts)
            self.assertEqual(before.manifest["outer_folds"][0]["inner_selection"], after.manifest["outer_folds"][0]["inner_selection"])
            columns = [c for c in before.predictions if c.startswith(("prediction_", "probability_", "predicted_label_"))]
            pd.testing.assert_frame_equal(before.predictions.loc[rows, columns], after.predictions.loc[rows, columns])

    def test_new_rows_do_not_change_v2_selection(self):
        for action, family in [(fit_predict, "ridge"), (fit_predict_binary, "logistic")]:
            opts = dict(target="y", group="g", inner_splits=2, baselines=(family,), categorical_encoding=DISTINCT)
            before = action(fixture(), new_rows(), **opts)
            changed = pd.DataFrame({"kind": [M + ":", "never-trained", None, M + "\x00"]})
            after = action(fixture(), changed, **opts)
            self.assertEqual(before.manifest["inner_selection"], after.manifest["inner_selection"])
            self.assertEqual(before.manifest["selected_candidate"], after.manifest["selected_candidate"])

    def test_failed_cross_protocol_refit_keeps_previous_fitted_encoding(self):
        frame = fixture()
        original_fit = Pipeline.fit
        for cls, family, method in [(GroupBenchRegressor, "ridge", "predict"),
                                    (GroupBenchClassifier, "logistic", "predict_proba")]:
            est = cls(baselines=(family,), inner_splits=2).fit(frame[["kind"]], frame.y, frame.g)
            before = getattr(est, method)(new_rows())
            est.set_params(categorical_encoding=DISTINCT)
            calls = 0
            def failing(pipeline, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise ValueError("Injected synthetic failure")
                return original_fit(pipeline, *args, **kwargs)
            with patch.object(Pipeline, "fit", new=failing), self.assertRaises(ValueError):
                est.fit(frame[["kind"]], frame.y, frame.g)
            self.assertEqual(calls, 3)
            self.assertEqual(est.categorical_encoding_, LEGACY)
            np.testing.assert_array_equal(before, getattr(est, method)(new_rows()))

    def test_cli_run_predict_and_plans_share_explicit_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, incoming = root / "training.csv", root / "new.csv"
            fixture().to_csv(train, index=False)
            new_rows().to_csv(incoming, index=False)
            for binary in (False, True):
                for prediction in (False, True):
                    mode = "predict" if prediction else "run"
                    prefix = ["binary"] if binary else []
                    files = [str(train)] + ([str(incoming)] if prediction else [])
                    opts = ["--target", "y", "--group", "g", "--inner-splits", "2", "--baselines", "dummy",
                            "--categorical-encoding", DISTINCT]
                    if not prediction:
                        opts += ["--outer-splits", "3"]
                    out = root / f"{binary}-{mode}"
                    with self.subTest(binary=binary, mode=mode):
                        stdout, stderr = io.StringIO(), io.StringIO()
                        with redirect_stdout(stdout), redirect_stderr(stderr), patch.object(Pipeline, "fit") as fitting:
                            code = main(prefix + ["plan", mode] + files + opts)
                        self.assertEqual(code, 0, stderr.getvalue())
                        fitting.assert_not_called()
                        self.assertFalse(out.exists())
                        plan = json.loads(stdout.getvalue())
                        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                            code = main(prefix + [mode] + files + opts + ["--out", str(out)])
                        self.assertEqual(code, 0, stderr.getvalue())
                        manifest = json.loads((out / "manifest.json").read_text())
                        self.assertEqual(plan["protocol"], manifest["protocol"])
                        self.assertEqual(plan["config"]["categorical_encoding"], manifest["config"]["categorical_encoding"])
                        self.assertEqual(manifest["config"]["categorical_encoding"], DISTINCT)
                        self.assertEqual(set(p.name for p in out.iterdir()),
                                         {"manifest.json", "predictions.csv", "checksums.json"} |
                                         (set() if prediction else {"summary.json", "report.html"}))


if __name__ == "__main__":
    unittest.main()
