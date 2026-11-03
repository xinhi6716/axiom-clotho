"""Handwritten software fixtures for fixed, partial group-forward evaluation.

No RNG, external data, search, scientific comparison or output artifact is used.
The tiny fitted fixtures test the declared software contract, not model quality.
"""

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import groupbench.group_time_evaluation as temporal
from groupbench.group_time_evaluation import (
    GroupForwardEvaluationError,
    GroupForwardEvaluationPreview,
    GroupForwardEvaluationResult,
    evaluate_group_forward_baselines,
    preview_group_forward_evaluation,
)
from groupbench.group_time_plan import (
    MAX_TICK, MIN_TICK, GroupForwardWindowPlan, GroupSpan, Membership, TimeWindow,
    plan_group_forward_windows,
)


RESULT_FIELDS = {
    "schema_id", "schema_version", "protocol", "coverage_mode", "bindings", "plan",
    "feature_schema", "metadata_columns", "target_name", "group_name", "recipes",
    "fit_budget", "coverage", "predictions", "metrics", "environment", "limitations",
}
PREDICTION_FIELDS = {
    "row_position", "group", "window_index", "observed", "prediction_dummy_mean",
    "prediction_ridge_alpha_1",
}
GROUPED_METRIC_FIELDS = {
    "n_rows", "mae", "rmse", "r2", "bias", "n_groups", "group_mae", "group_rmse",
    "worst_group_mae",
}
METADATA = ("group", "start", "end", "label_available_at", "row_key", "fold")
RECIPES = ("dummy_mean", "ridge_alpha_1")


def handwritten_fixture():
    """The nine-row acceptance fixture specified in DESIGN.md, without RNG."""
    groups = [1, "1", "a", "warm2", "b", "cross", 1, "a", "late"]
    spans = [
        GroupSpan(1, 0, 1, 1), GroupSpan("warm2", 2, 3, 3),
        GroupSpan("1", 10, 10, 12), GroupSpan("a", 11, 12, 12),
        GroupSpan("b", 20, 21, 21), GroupSpan("cross", 8, 12, 12),
        GroupSpan("late", 40, 41, 41),
    ]
    values = [0., 3., 4., 4., 5., 10., 2., 6., 9.]
    plan = plan_group_forward_windows(
        groups, group_spans=spans, windows=[TimeWindow(10, 15), TimeWindow(20, 25)],
        time_unit="day", clock_id="handwritten-clock", gap=2,
    )
    return dict(
        X=pd.DataFrame({"x": values}), y=pd.Series(values, name="outcome"),
        row_groups=pd.Series(groups, dtype=object, name="group"), plan=plan,
        feature_allowlist=("x",), metadata_columns=METADATA,
    )


def preview(arguments):
    return preview_group_forward_evaluation(**arguments)


_UNSET = object()


def evaluate(arguments, binding=_UNSET):
    if binding is _UNSET:
        binding = preview(arguments).bindings["execution_sha256"]
    return evaluate_group_forward_baselines(**arguments, expected_execution_sha256=binding)


def changed_plan(plan, **changes):
    options = dict(
        row_groups=list(plan.row_groups), group_spans=list(plan.group_spans),
        windows=[fold.window for fold in plan.folds], time_unit=plan.time_unit,
        clock_id=plan.clock_id, gap=plan.gap,
    )
    options.update(changes)
    return plan_group_forward_windows(**options)


def forged(value, name, replacement):
    result = deepcopy(value)
    object.__setattr__(result, name, replacement)
    return result


def first_window_predictions(result):
    return tuple((row["prediction_dummy_mean"], row["prediction_ridge_alpha_1"])
                 for row in result.predictions if row["window_index"] == 0)


class TemporalContractTestCase(unittest.TestCase):
    def assert_rejected_before_pipeline(self, arguments, expected="0" * 64):
        # Invalid inputs may not even construct a model. Exercise both public
        # boundaries rather than relying on the preview's validation alone.
        with patch.object(temporal, "make_pipeline", side_effect=AssertionError("constructed")) as make, \
                patch.object(Pipeline, "fit", side_effect=AssertionError("fitted")) as fit, \
                patch.object(np.random, "default_rng", side_effect=AssertionError("RNG")) as rng:
            for operation in (
                lambda: preview(arguments),
                lambda: evaluate(arguments, expected),
            ):
                with self.assertRaises(GroupForwardEvaluationError):
                    operation()
            make.assert_not_called()
            fit.assert_not_called()
            rng.assert_not_called()

    def assert_stale_before_pipeline(self, arguments, digest):
        with patch.object(temporal, "make_pipeline", side_effect=AssertionError("constructed")) as make, \
                patch.object(Pipeline, "fit", side_effect=AssertionError("fitted")) as fit:
            with self.assertRaises(GroupForwardEvaluationError):
                evaluate(arguments, digest)
            make.assert_not_called()
            fit.assert_not_called()

    @contextmanager
    def traced_pipelines(self, mutate_on_first_construction=None):
        original_make = temporal.make_pipeline
        records = []

        def make(*args, **kwargs):
            pipeline = original_make(*args, **kwargs)
            record = {"pipeline": pipeline, "fits": [], "predictions": []}
            records.append(record)
            if mutate_on_first_construction is not None and len(records) == 1:
                mutate_on_first_construction()
            original_fit, original_predict = pipeline.fit, pipeline.predict

            def fit(X, y, **fit_kwargs):
                record["fits"].append((tuple(X.index), np.asarray(X).copy(), np.asarray(y).copy()))
                fitted = original_fit(X, y, **fit_kwargs)
                numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
                record["imputer"] = numeric.named_steps["impute"]
                record["scaler"] = numeric.named_steps["scale"]
                record["statistics"] = record["imputer"].statistics_.copy()
                record["mean"] = record["scaler"].mean_.copy()
                record["variance"] = record["scaler"].var_.copy()
                return fitted

            def predict(X, **predict_kwargs):
                record["predictions"].append((tuple(X.index), np.asarray(X).copy()))
                return original_predict(X, **predict_kwargs)

            pipeline.fit, pipeline.predict = fit, predict
            return pipeline

        with patch.object(temporal, "make_pipeline", make):
            yield records


class ForwardEvaluationAcceptanceTests(TemporalContractTestCase):
    def test_preview_has_zero_fits_and_is_bound_without_rng_or_writes(self):
        arguments = handwritten_fixture()
        with ExitStack() as stack:
            mocks = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
                     for owner, name in (
                         (temporal, "make_pipeline"), (Pipeline, "fit"),
                         (np.random, "default_rng"), (np.random, "RandomState"),
                         (np.random, "seed"), (pd.DataFrame, "to_csv"),
                         (pd.DataFrame, "to_json"),
                     )]
            result = preview(arguments)
        self.assertIsInstance(result, GroupForwardEvaluationPreview)
        self.assertEqual(result.schema_id, "mlcalc.group_forward_baseline_preview")
        self.assertEqual(type(result.schema_version), int)
        self.assertEqual(result.schema_version, 1)
        self.assertEqual(type(result.fits_performed), int)
        self.assertEqual(result.fits_performed, 0)
        self.assertEqual(result.fit_budget["expected_total"], 4)
        self.assertEqual(result.fit_budget["attempted_total"], 0)
        self.assertEqual(result.fit_budget["completed_total"], 0)
        self.assertEqual(result.windows, (
            {"window_index": 0, "train": {"n_rows": 3, "n_groups": 2},
             "test": {"n_rows": 3, "n_groups": 2}, "unused": {"n_rows": 3, "n_groups": 3}},
            {"window_index": 1, "train": {"n_rows": 7, "n_groups": 5},
             "test": {"n_rows": 1, "n_groups": 1}, "unused": {"n_rows": 1, "n_groups": 1}},
        ))
        self.assertEqual(result.resource_limits["matrix_cell_visits"], 28)
        for name in ("plan_sha256", "input_sha256", "execution_sha256"):
            self.assertRegex(result.bindings[name], r"\A[0-9a-f]{64}\Z")
        self.assertFalse(hasattr(result, "predictions"))
        self.assertFalse(hasattr(result, "metrics"))
        self.assertFalse(hasattr(result, "leaderboard"))
        self.assertEqual(result.bindings, preview(arguments).bindings)
        for mocked in mocks:
            mocked.assert_not_called()

    def test_exact_end_to_end_partial_coverage_and_fixed_recipes(self):
        arguments = handwritten_fixture()
        result = evaluate(arguments)
        self.assertIsInstance(result, GroupForwardEvaluationResult)
        self.assertEqual({field.name for field in fields(result)}, RESULT_FIELDS)
        self.assertEqual(set(result.to_dict()), RESULT_FIELDS)
        self.assertEqual(result.schema_id, "mlcalc.group_forward_baseline_evaluation")
        self.assertEqual(type(result.schema_version), int)
        self.assertEqual(result.schema_version, 1)
        self.assertEqual(result.protocol, "group-forward-fixed-regression-v1")
        self.assertEqual(result.coverage_mode, "partial_predeclared")
        self.assertEqual(result.target_name, "outcome")
        self.assertEqual(result.group_name, "group")
        self.assertEqual(result.metadata_columns, METADATA)
        self.assertFalse(hasattr(result, "leaderboard"))
        self.assertNotIn("selected", result.to_dict())
        self.assertEqual(tuple(row["recipe"] for row in result.metrics), RECIPES)
        self.assertEqual(tuple(row["name"] for row in result.recipes), RECIPES)
        self.assertEqual(result.recipes[0]["parameters"], {"strategy": "mean"})
        self.assertEqual(result.recipes[1]["parameters"], {
            "alpha": 1.0, "solver": "lsqr", "fit_intercept": True, "tol": 1e-8,
            "max_iter": 1000, "copy_X": True, "positive": False,
        })
        for recipe in result.recipes:
            self.assertEqual(recipe["preprocessing"], (
                {"step": "SimpleImputer", "strategy": "median", "keep_empty_features": True},
                {"step": "StandardScaler", "with_mean": True, "with_std": True},
            ))
        for dependency in ("package_version", "python", "numpy", "pandas", "sklearn", "scipy", "threadpoolctl"):
            self.assertIsInstance(result.environment[dependency], str)
            self.assertTrue(result.environment[dependency])
        self.assertEqual(result.environment["package_source"]["status"], "available")
        self.assertEqual(tuple(row["row_position"] for row in result.predictions), (1, 2, 4, 7))
        self.assertEqual(tuple(row["window_index"] for row in result.predictions), (0, 0, 1, 0))
        self.assertEqual(tuple(row["observed"] for row in result.predictions), (3., 4., 5., 6.))
        self.assertEqual(tuple(row["group"] for row in result.predictions), ("1", "a", "b", "a"))
        for row in result.predictions:
            self.assertEqual(set(row), PREDICTION_FIELDS)
            self.assertIs(type(row["row_position"]), int)
            self.assertIs(type(row["window_index"]), int)
        np.testing.assert_allclose(
            [row["prediction_dummy_mean"] for row in result.predictions],
            [2., 2., 29 / 7, 2.], rtol=0, atol=1e-12,
        )
        np.testing.assert_allclose(
            [row["prediction_ridge_alpha_1"] for row in result.predictions],
            [2.75, 3.5, 137 / 28, 5.], rtol=0, atol=1e-10,
        )
        self.assertEqual(result.plan["fits_performed"], 0)
        self.assertEqual(result.plan["tested"]["rows"], (1, 2, 4, 7))
        self.assertEqual(result.plan["untested"]["rows"], (0, 3, 5, 6, 8))
        self.assertEqual(result.plan["folds"][0]["train"]["rows"], (0, 3, 6))
        self.assertEqual(result.plan["folds"][1]["train"]["rows"], (0, 1, 2, 3, 5, 6, 7))
        self.assertEqual(result.plan["total"]["groups"][:2], (1, "1"))
        self.assertIs(type(result.plan["total"]["groups"][0]), int)
        self.assertIs(type(result.plan["total"]["groups"][1]), str)
        self.assertFalse(result.coverage["untested_predictions_present"])
        for name, n_rows, n_groups in (("total", 9, 7), ("tested", 4, 3), ("untested", 5, 4)):
            self.assertEqual(result.coverage[name], result.plan[name])
            self.assertEqual((result.coverage[name]["n_rows"], result.coverage[name]["n_groups"]),
                             (n_rows, n_groups))
        self.assertEqual(result.fit_budget["unit"], "pipeline.fit calls")
        for key in ("expected_total", "attempted_total", "completed_total"):
            self.assertEqual(result.fit_budget[key], 4)
        self.assertEqual(result.fit_budget["per_window"], (
            {"window_index": 0, "expected": 2, "attempted": 2, "completed": 2},
            {"window_index": 1, "expected": 2, "attempted": 2, "completed": 2},
        ))
        for key in ("inner_selection_fits", "full_data_refits", "selected_model_refits", "bootstraps", "retries"):
            self.assertEqual(result.fit_budget[key], 0)

    def test_metrics_are_pooled_rows_and_equal_groups_not_equal_windows(self):
        result = evaluate(handwritten_fixture())
        dummy, ridge = result.metrics
        self.assertEqual(set(dummy), {"recipe", "pooled", "per_window", "per_group"})
        self.assertEqual(set(dummy["pooled"]), GROUPED_METRIC_FIELDS)
        pooled = dummy["pooled"]
        self.assertEqual((pooled["n_rows"], pooled["n_groups"]), (4, 3))
        self.assertAlmostEqual(pooled["mae"], 55 / 28)
        self.assertAlmostEqual(pooled["rmse"], math.sqrt(1065) / 14)
        self.assertAlmostEqual(pooled["bias"], -55 / 28)
        self.assertAlmostEqual(pooled["r2"], -164 / 49)
        self.assertAlmostEqual(pooled["group_mae"], 34 / 21)
        self.assertAlmostEqual(pooled["group_rmse"], (1 + math.sqrt(10) + 6 / 7) / 3)
        self.assertAlmostEqual(pooled["worst_group_mae"], 3)
        self.assertAlmostEqual(ridge["pooled"]["mae"], 13 / 28)
        self.assertAlmostEqual(ridge["pooled"]["group_mae"], 31 / 84)
        first, second = dummy["per_window"]
        self.assertEqual(set(first), {"window_index"} | GROUPED_METRIC_FIELDS)
        self.assertEqual((first["window_index"], second["window_index"]), (0, 1))
        self.assertAlmostEqual(first["group_mae"], 2)
        self.assertAlmostEqual(first["rmse"], math.sqrt(7))
        self.assertAlmostEqual(first["r2"], -3.5)
        self.assertIsNone(second["r2"])
        self.assertNotAlmostEqual(pooled["group_mae"], (first["group_mae"] + second["group_mae"]) / 2)
        self.assertNotAlmostEqual(pooled["mae"], (first["mae"] + second["mae"]) / 2)
        groups = dummy["per_group"]
        self.assertEqual(tuple(row["group"] for row in groups), ("1", "a", "b"))
        self.assertEqual(tuple(row["n_rows"] for row in groups), (1, 2, 1))
        self.assertIsNone(groups[0]["r2"])
        self.assertIsNone(groups[2]["r2"])
        self.assertAlmostEqual(groups[1]["rmse"], math.sqrt(10))

    def test_four_fresh_pipelines_fit_exact_train_rows_inside_single_thread_context(self):
        arguments = handwritten_fixture()
        context_active = [False]
        limits_calls, transform_fits = [], []
        original_transform_fit = ColumnTransformer.fit_transform

        @contextmanager
        def limits(*args, **kwargs):
            self.assertFalse(context_active[0])
            self.assertEqual(args, ())
            self.assertEqual(kwargs, {"limits": 1})
            limits_calls.append(kwargs)
            context_active[0] = True
            try:
                yield
            finally:
                context_active[0] = False

        def transform_fit(transformer, X, y=None, **kwargs):
            self.assertTrue(context_active[0])
            transform_fits.append(tuple(X.index))
            return original_transform_fit(transformer, X, y, **kwargs)

        with self.traced_pipelines() as records, \
                patch.object(temporal, "threadpool_limits", limits), \
                patch.object(ColumnTransformer, "fit_transform", transform_fit):
            result = evaluate(arguments)
        expected_train = [(0, 3, 6), (0, 3, 6), (0, 1, 2, 3, 5, 6, 7), (0, 1, 2, 3, 5, 6, 7)]
        self.assertEqual(len(records), 4)
        self.assertEqual(transform_fits, expected_train)
        self.assertTrue(limits_calls)
        self.assertFalse(context_active[0])
        self.assertEqual(len({id(record["pipeline"]) for record in records}), 4)
        self.assertEqual(len({id(record["imputer"]) for record in records}), 4)
        self.assertEqual(len({id(record["scaler"]) for record in records}), 4)
        for index, record in enumerate(records):
            self.assertEqual(len(record["fits"]), 1)
            positions, X, y = record["fits"][0]
            self.assertEqual(positions, expected_train[index])
            np.testing.assert_array_equal(X[:, 0], arguments["X"].iloc[list(positions), 0])
            np.testing.assert_array_equal(y, arguments["y"].iloc[list(positions)])
            self.assertTrue(record["predictions"])
            for positions, _ in record["predictions"]:
                self.assertEqual(positions, (1, 2, 7) if index < 2 else (4,))
        for index, record in enumerate(records):
            model = record["pipeline"].named_steps["model"]
            if index % 2 == 0:
                self.assertEqual(type(model).__name__, "DummyRegressor")
                self.assertEqual(model.get_params()["strategy"], "mean")
            else:
                self.assertEqual(type(model).__name__, "Ridge")
                expected = {"alpha": 1.0, "solver": "lsqr", "fit_intercept": True,
                            "tol": 1e-8, "max_iter": 1000, "copy_X": True, "positive": False}
                self.assertEqual({key: model.get_params()[key] for key in expected}, expected)
        for record in records[:2]:
            np.testing.assert_array_equal(record["statistics"], [2.])
            np.testing.assert_array_equal(record["mean"], [2.])
            np.testing.assert_allclose(record["variance"], [8 / 3], rtol=0, atol=1e-14)
        self.assertEqual(result.fit_budget["completed_total"], 4)

    def test_earlier_predictions_ignore_nontraining_targets_but_later_training_can_use_them(self):
        arguments = handwritten_fixture()
        original = evaluate(arguments)
        changed = handwritten_fixture()
        changed["y"].iloc[[1, 2, 4, 5, 7, 8]] += 100
        revised = evaluate(changed)
        np.testing.assert_allclose(first_window_predictions(original), first_window_predictions(revised), rtol=0, atol=1e-12)
        old_later = next(row for row in original.predictions if row["window_index"] == 1)
        new_later = next(row for row in revised.predictions if row["window_index"] == 1)
        self.assertNotEqual(old_later["prediction_dummy_mean"], new_later["prediction_dummy_mean"])
        self.assertNotEqual(original.bindings["input_sha256"], revised.bindings["input_sha256"])

    def test_future_feature_changes_do_not_change_earlier_training_preprocessing(self):
        arguments = handwritten_fixture()
        arguments["X"].iloc[0, 0] = np.nan
        with self.traced_pipelines() as baseline:
            old_result = evaluate(arguments)
        arguments["X"].iloc[[1, 2, 4, 5, 7, 8], 0] += 100
        with self.traced_pipelines() as changed:
            new_result = evaluate(arguments)
        for old, new in zip(baseline[:2], changed[:2]):
            for key in ("statistics", "mean", "variance"):
                np.testing.assert_array_equal(old[key], new[key])
            np.testing.assert_array_equal(old["statistics"], [3.])
        self.assertNotEqual(first_window_predictions(old_result)[0][1], first_window_predictions(new_result)[0][1])
        self.assertEqual(first_window_predictions(old_result)[0][0], first_window_predictions(new_result)[0][0])

    def test_all_missing_training_feature_is_retained_and_imputed_to_zero(self):
        arguments = handwritten_fixture()
        arguments["X"].iloc[[0, 3, 6], 0] = np.nan
        with self.traced_pipelines() as records:
            result = evaluate(arguments)
        for record in records[:2]:
            np.testing.assert_array_equal(record["statistics"], [0.])
            np.testing.assert_array_equal(record["mean"], [0.])
        for row in result.predictions:
            for name in ("prediction_dummy_mean", "prediction_ridge_alpha_1"):
                self.assertTrue(math.isfinite(row[name]))
        for row in result.predictions:
            if row["window_index"] == 0:
                self.assertAlmostEqual(row["prediction_ridge_alpha_1"], 2)

    def test_prepared_snapshots_are_detached_from_caller_mutation_during_fitting(self):
        arguments = handwritten_fixture()
        baseline = evaluate(arguments)
        digest = preview(arguments).bindings["execution_sha256"]

        original_span = arguments["plan"].group_spans[0]
        original_window = arguments["plan"].folds[0].window
        prepared_snapshots = []
        original_prepare = temporal._prepare

        def capture_prepare(*args, **kwargs):
            prepared = original_prepare(*args, **kwargs)
            prepared_snapshots.append(prepared)
            return prepared

        def mutate():
            arguments["X"].iloc[:, :] = 999
            arguments["y"].iloc[:] = -999
            arguments["row_groups"].iloc[:] = "changed-after-snapshot"
            # Frozen dataclasses are not ownership barriers against deliberate
            # object.__setattr__. Retained canonical records must be copies.
            object.__setattr__(original_span, "label_available_at", 999)
            object.__setattr__(original_window, "lower", 999)

        with self.traced_pipelines(mutate) as records, \
                patch.object(temporal, "_prepare", capture_prepare):
            result = evaluate(arguments, digest)
        self.assertEqual(len(prepared_snapshots), 1)
        canonical = prepared_snapshots[0].plan
        self.assertIsNot(canonical.group_spans[0], original_span)
        self.assertIsNot(canonical.folds[0].window, original_window)
        self.assertEqual(canonical.group_spans[0].label_available_at, 1)
        self.assertEqual(canonical.folds[0].window.lower, 10)
        self.assertEqual(result.plan["group_spans"][0]["label_available_at"], 1)
        self.assertEqual(result.plan["folds"][0]["window"]["lower"], 10)
        self.assertEqual(result.predictions, baseline.predictions)
        self.assertEqual(result.metrics, baseline.metrics)
        self.assertEqual(result.bindings, baseline.bindings)
        self.assertEqual(records[0]["fits"][0][2].tolist(), [0., 4., 2.])
        self.assertEqual(result.plan["row_groups"], baseline.plan["row_groups"])

    def test_results_and_preview_are_recursively_immutable_and_to_dict_is_detached(self):
        arguments = handwritten_fixture()
        for result in (preview(arguments), evaluate(arguments)):
            with self.subTest(type=type(result).__name__):
                self.assertTrue(is_dataclass(result))
                with self.assertRaises(FrozenInstanceError):
                    result.schema_id = "changed"

                def check(value):
                    self.assertNotIsInstance(value, (dict, list, set, np.ndarray, pd.DataFrame, pd.Series))
                    if isinstance(value, Mapping):
                        with self.assertRaises(TypeError):
                            value["injected"] = True
                        for item in value.values():
                            check(item)
                    elif isinstance(value, tuple):
                        for item in value:
                            check(item)
                    elif is_dataclass(value):
                        for field in fields(value):
                            check(getattr(value, field.name))

                check(result)
                before = result.to_dict()
                detached = result.to_dict()
                detached["bindings"]["execution_sha256"] = "changed"
                detached["metadata_columns"].append("changed")
                if "predictions" in detached:
                    detached["predictions"][0]["observed"] = -123
                    detached["plan"]["tested"]["rows"].append(8)
                self.assertEqual(result.to_dict(), before)


class ForwardEvaluationValidationTests(TemporalContractTestCase):
    def test_noncanonical_indexes_and_lengths_are_never_repaired(self):
        for target in ("X", "y", "row_groups"):
            for label, index in (
                ("named", pd.RangeIndex(9, name="source_row")),
                ("offset", pd.RangeIndex(1, 10)),
                ("step", pd.RangeIndex(0, 18, 2)),
                ("integer_index", pd.Index(np.asarray(list(range(9))))),
                ("reversed", pd.Index(list(reversed(range(9))))),
                ("duplicate", pd.Index([0] * 9)),
            ):
                with self.subTest(target=target, index=label):
                    arguments = handwritten_fixture()
                    arguments[target].index = index
                    self.assert_rejected_before_pipeline(arguments)
            with self.subTest(target=target, length="short"):
                arguments = handwritten_fixture()
                arguments[target] = arguments[target].iloc[:-1]
                self.assert_rejected_before_pipeline(arguments)
        arguments = handwritten_fixture()
        for name in ("X", "y", "row_groups"):
            arguments[name].index = pd.RangeIndex(1, 10)
        self.assert_rejected_before_pipeline(arguments)

    def test_required_input_containers_and_named_series(self):
        cases = (
            ("X", lambda value: value.to_numpy()),
            ("X", lambda value: value.to_dict()),
            ("y", lambda value: value.to_numpy()),
            ("y", lambda value: value.tolist()),
            ("row_groups", lambda value: value.tolist()),
            ("y", lambda value: value.rename(None)),
            ("row_groups", lambda value: value.rename(None)),
            ("y", lambda value: value.rename(" padded ")),
            ("row_groups", lambda value: value.rename(" ")),
        )
        for name, transform in cases:
            with self.subTest(name=name, transform=transform):
                arguments = handwritten_fixture()
                arguments[name] = transform(arguments[name])
                self.assert_rejected_before_pipeline(arguments)

    def test_allowlist_is_explicit_exact_order_and_excludes_all_declared_roles(self):
        for allowlist in ("x", [], (), ("x", "x"), (" x",), ("",), {"x"}, iter(["x"]), (1,)):
            with self.subTest(allowlist=repr(allowlist)):
                arguments = handwritten_fixture()
                arguments["feature_allowlist"] = allowlist
                self.assert_rejected_before_pipeline(arguments)
        for name in ("outcome", "group", "start", "end", "label_available_at", "row_key", "fold"):
            with self.subTest(role=name):
                arguments = handwritten_fixture()
                arguments["X"] = arguments["X"].rename(columns={"x": name})
                arguments["feature_allowlist"] = (name,)
                self.assert_rejected_before_pipeline(arguments)
        arguments = handwritten_fixture()
        arguments["X"]["extra"] = 1.
        self.assert_rejected_before_pipeline(arguments)
        arguments["feature_allowlist"] = ("extra", "x")
        self.assert_rejected_before_pipeline(arguments)
        arguments = handwritten_fixture()
        arguments["X"] = pd.concat([arguments["X"], arguments["X"]], axis=1)
        self.assert_rejected_before_pipeline(arguments)

    def test_metadata_declaration_is_required_distinct_and_role_disjoint(self):
        for metadata in (None, "group", [], ("start",), ("group", "group"),
                         ("group", "outcome"), ("group", "x"), ("group", " padded ")):
            with self.subTest(metadata=metadata):
                arguments = handwritten_fixture()
                arguments["metadata_columns"] = metadata
                self.assert_rejected_before_pipeline(arguments)
        arguments = handwritten_fixture()
        arguments["y"] = arguments["y"].rename("group")
        self.assert_rejected_before_pipeline(arguments)

    def test_non_real_numeric_columns_and_unsafe_precision_reject_before_fit(self):
        columns = {
            "bool": pd.Series([True] * 9),
            "complex": pd.Series([1 + 0j] * 9),
            "numeric_strings": pd.Series(["3"] * 9),
            "object_numbers": pd.Series([3.] * 9, dtype=object),
            "categorical": pd.Series(pd.Categorical([1, 2, 3] * 3)),
            "datetime": pd.Series(pd.date_range("2000-01-01", periods=9)),
            "timedelta": pd.Series(pd.to_timedelta(list(range(9)), unit="day")),
            "positive_inf": pd.Series([np.inf] + [1.] * 8),
            "negative_inf": pd.Series([-np.inf] + [1.] * 8),
            "inexact_integer": pd.Series([2**53 + 1] + [1] * 8, dtype=np.int64),
            "inexact_negative_integer": pd.Series([-(2**53 + 1)] + [1] * 8, dtype=np.int64),
            "uint64_max": pd.Series([2**64 - 1] + [1] * 8, dtype=np.uint64),
            "nullable_float": pd.Series([1.] * 9, dtype="Float64"),
            "nullable_int": pd.Series([1] * 9, dtype="Int64"),
            "masked_extension": pd.Series([1., pd.NA] + [1.] * 7, dtype="Float64"),
        }
        if np.dtype(np.longdouble).itemsize > 8:
            columns["wide_float"] = pd.Series(np.ones(9, dtype=np.longdouble))
        for label, series in columns.items():
            for target in ("X", "y"):
                with self.subTest(dtype=label, target=target):
                    arguments = handwritten_fixture()
                    if target == "X":
                        arguments["X"]["x"] = series
                    else:
                        arguments["y"] = series.rename("outcome")
                    self.assert_rejected_before_pipeline(arguments)

    def test_missing_target_rejects_even_when_row_is_never_tested(self):
        for row in (0, 8, 1):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(row=row, value=value):
                    arguments = handwritten_fixture()
                    arguments["y"].iloc[row] = value
                    self.assert_rejected_before_pipeline(arguments)

    def test_feature_nans_and_exact_large_integer_values_are_valid_for_preview(self):
        for name in ("X", "y"):
            arguments = handwritten_fixture()
            values = pd.Series([2**53 + 2, -(2**53 + 2)] + [0] * 7, dtype=np.int64)
            if name == "X":
                arguments["X"]["x"] = values
            else:
                arguments["y"] = values.rename("outcome")
            self.assertEqual(preview(arguments).fits_performed, 0)
        arguments = handwritten_fixture()
        arguments["X"].iloc[[0, 1, 8], 0] = np.nan
        self.assertEqual(preview(arguments).fits_performed, 0)

    def test_group_dtype_coercion_and_nonexact_scalar_types_reject(self):
        for bad in (True, 1., np.int64(1), np.float64(1), None, "", "   ", 2**63, -(2**63) - 1):
            with self.subTest(value_type=type(bad).__name__):
                arguments = handwritten_fixture()
                arguments["row_groups"].iat[0] = bad
                self.assert_rejected_before_pipeline(arguments)
        for dtype in (str, "string"):
            with self.subTest(dtype=dtype):
                arguments = handwritten_fixture()
                arguments["row_groups"] = arguments["row_groups"].astype(dtype)
                self.assert_rejected_before_pipeline(arguments)
        arguments = handwritten_fixture()
        arguments["row_groups"].iat[0] = "1"
        self.assert_rejected_before_pipeline(arguments)
        arguments = handwritten_fixture()
        arguments["row_groups"].iloc[[0, 1]] = arguments["row_groups"].iloc[[1, 0]].to_numpy()
        self.assert_rejected_before_pipeline(arguments)

    def test_forged_frozen_plan_fields_are_rebuilt_and_compared_type_sensitively(self):
        arguments = handwritten_fixture()
        original = arguments["plan"]
        first, second = original.folds
        bad_firsts = (
            replace(first, index=True),
            replace(first, train=replace(first.train, rows=(False, 3, 6))),
            replace(first, test=replace(first.test, rows=(True, 2, 7))),
            replace(first, test=replace(first.test, rows=(1, 2))),
            replace(first, train=replace(first.train, groups=tuple(reversed(first.train.groups)))),
            replace(first, unused=Membership(first.unused.rows + (1,), first.unused.groups)),
            replace(first, exclusions=(replace(first.exclusions[0], train="not_true"),) + first.exclusions[1:]),
            replace(first, exclusions=first.exclusions[:-1]),
            replace(first, train=replace(first.train, rows=list(first.train.rows))),
            replace(first, exclusions=list(first.exclusions)),
            replace(first, window=replace(first.window, lower=True)),
        )
        bad_plans = [replace(original, folds=(first_change, second)) for first_change in bad_firsts]
        bad_plans += [
            forged(original, "fits_performed", False),
            forged(original, "fits_performed", 1),
            forged(original, "protocol", "other-protocol"),
            forged(original, "coverage_mode", "full_oof"),
            replace(original, gap=True),
            replace(original, row_groups=list(original.row_groups)),
            replace(original, group_spans=list(original.group_spans)),
            replace(original, folds=list(original.folds)),
            replace(original, total=replace(original.total, groups=tuple(reversed(original.total.groups)))),
            replace(original, tested=replace(original.tested, rows=original.tested.rows + (0,))),
            replace(original, untested=replace(original.untested, rows=original.untested.rows[:-1])),
            replace(original, group_spans=(replace(original.group_spans[0], start=False),) + original.group_spans[1:]),
            replace(original, folds=(first, replace(second, test=first.test))),
            replace(original, folds=(first, first)),
        ]
        for index, bad in enumerate(bad_plans):
            with self.subTest(tamper=index):
                altered = {**arguments, "plan": bad}
                self.assert_rejected_before_pipeline(altered)

    def test_exact_dataclasses_with_missing_slots_reject_categorically_before_fit(self):
        arguments = handwritten_fixture()
        original = arguments["plan"]
        no_fields = object.__new__(GroupForwardWindowPlan)
        missing_total = deepcopy(original)
        object.__delattr__(missing_total, "total")
        missing_rows = deepcopy(original.folds[0].train)
        object.__delattr__(missing_rows, "rows")
        missing_exclusions = deepcopy(original.folds[0])
        object.__delattr__(missing_exclusions, "exclusions")
        missing_span_group = deepcopy(original.group_spans[0])
        object.__delattr__(missing_span_group, "group")
        missing_window_lower = deepcopy(original.folds[0].window)
        object.__delattr__(missing_window_lower, "lower")
        bad_plans = (
            no_fields,
            missing_total,
            replace(original, folds=(replace(original.folds[0], train=missing_rows), original.folds[1])),
            replace(original, folds=(missing_exclusions, original.folds[1])),
            replace(original, group_spans=(missing_span_group,) + original.group_spans[1:]),
            replace(original, folds=(replace(original.folds[0], window=missing_window_lower), original.folds[1])),
        )
        for index, bad in enumerate(bad_plans):
            with self.subTest(missing_slot=index):
                self.assert_rejected_before_pipeline({**arguments, "plan": bad})

    def test_plan_and_nested_record_subclasses_are_not_verification_certificates(self):
        class PlanSubclass(GroupForwardWindowPlan):
            pass

        arguments = handwritten_fixture()
        original = arguments["plan"]
        arguments["plan"] = PlanSubclass(**{field.name: getattr(original, field.name)
                                           for field in fields(original) if field.init})
        self.assert_rejected_before_pipeline(arguments)

    def test_each_content_role_and_plan_change_invalidates_old_execution_binding(self):
        original = handwritten_fixture()
        digest = preview(original).bindings["execution_sha256"]
        cases = []
        for column, position in (("X", 0), ("X", 8), ("y", 0), ("y", 8)):
            changed = handwritten_fixture()
            if column == "X":
                changed[column].iloc[position, 0] += 1
            else:
                changed[column].iloc[position] += 1
            cases.append(changed)
        changed = handwritten_fixture()
        changed["metadata_columns"] = tuple(reversed(METADATA))
        cases.append(changed)
        changed = handwritten_fixture()
        changed["y"] = changed["y"].rename("renamed_outcome")
        cases.append(changed)
        changed = handwritten_fixture()
        changed["row_groups"] = changed["row_groups"].rename("renamed_group")
        changed["metadata_columns"] = ("renamed_group",) + METADATA[1:]
        cases.append(changed)
        for options in ({"clock_id": "different-clock"}, {"time_unit": "hour"}, {"gap": 1},
                        {"windows": [TimeWindow(10, 16), TimeWindow(20, 25)]},
                        {"group_spans": [replace(original["plan"].group_spans[0], start=-1)]
                         + list(original["plan"].group_spans[1:])}):
            changed = handwritten_fixture()
            changed["plan"] = changed_plan(changed["plan"], **options)
            cases.append(changed)
        for index, changed in enumerate(cases):
            with self.subTest(change=index):
                self.assertNotEqual(preview(changed).bindings["execution_sha256"], digest)
                self.assert_stale_before_pipeline(changed, digest)

    def test_feature_order_and_within_group_row_swaps_change_input_binding(self):
        arguments = handwritten_fixture()
        arguments["X"]["second"] = [9., 8., 7., 6., 5., 4., 3., 2., 1.]
        arguments["feature_allowlist"] = ("x", "second")
        old = preview(arguments).bindings["execution_sha256"]
        arguments["X"] = arguments["X"][["second", "x"]]
        arguments["feature_allowlist"] = ("second", "x")
        self.assert_stale_before_pipeline(arguments, old)
        arguments = handwritten_fixture()
        old = preview(arguments).bindings["execution_sha256"]
        # Positions zero and six share integer group 1, so row-group binding
        # alone cannot discover this change. The complete input digest must.
        arguments["X"].iloc[[0, 6]] = arguments["X"].iloc[[6, 0]].to_numpy()
        arguments["y"].iloc[[0, 6]] = arguments["y"].iloc[[6, 0]].to_numpy()
        self.assertEqual(arguments["row_groups"].iat[0], arguments["row_groups"].iat[6])
        self.assert_stale_before_pipeline(arguments, old)

    def test_missing_or_malformed_digest_never_constructs_a_pipeline(self):
        arguments = handwritten_fixture()
        for digest in (None, "", "0" * 63, "0" * 65, "z" * 64, 0, b"0" * 64, True):
            with self.subTest(digest_type=type(digest).__name__, digest_length=len(digest) if isinstance(digest, (str, bytes)) else None):
                self.assert_stale_before_pipeline(arguments, digest)
        with patch.object(temporal, "make_pipeline") as make:
            with self.assertRaises(TypeError):
                evaluate_group_forward_baselines(**arguments)
            make.assert_not_called()

    def test_unavailable_package_source_rejects_preview_and_execution_before_fitting(self):
        with patch.object(temporal, "observe_package_source", return_value={
            "schema_version": "1.0", "algorithm": "sha256", "scope": "fixed_groupbench_python_sources_v1",
            "status": "unavailable", "reason": "unsupported_source_access",
        }):
            self.assert_rejected_before_pipeline(handwritten_fixture())

    def test_observed_source_change_invalidates_execution_binding(self):
        arguments = handwritten_fixture()
        old = preview(arguments).bindings["execution_sha256"]
        observed = temporal.observe_package_source()
        self.assertEqual(observed["status"], "available")
        changed = {**observed, "sha256": "0" * 64}
        with patch.object(temporal, "observe_package_source", return_value=changed):
            self.assert_stale_before_pipeline(arguments, old)

    def test_fixed_recipe_declaration_changes_invalidate_old_execution_binding(self):
        arguments = handwritten_fixture()
        old = preview(arguments).bindings["execution_sha256"]
        altered = temporal._recipes()
        altered[1]["parameters"]["alpha"] = 2.0
        with patch.object(temporal, "_recipes", return_value=altered):
            self.assert_stale_before_pipeline(arguments, old)

    def test_binding_preserves_signed_zero_and_canonicalizes_nan_payloads(self):
        arguments = handwritten_fixture()
        original = preview(arguments).bindings["input_sha256"]
        arguments["X"].iat[0, 0] = -0.
        self.assertNotEqual(preview(arguments).bindings["input_sha256"], original)
        arguments["X"].iat[0, 0] = np.asarray([0x7FF8000000000001], dtype=np.uint64).view(np.float64)[0]
        first_nan = preview(arguments).bindings["input_sha256"]
        arguments["X"].iat[0, 0] = np.asarray([0x7FF8000000000012], dtype=np.uint64).view(np.float64)[0]
        self.assertEqual(preview(arguments).bindings["input_sha256"], first_nan)
        arguments["X"].iat[1, 0] = np.nan
        self.assertNotEqual(preview(arguments).bindings["input_sha256"], first_nan)

    def test_errors_do_not_echo_private_invalid_values(self):
        arguments = handwritten_fixture()
        secret = "private-fixture-marker-not-for-error-messages"
        arguments["y"] = pd.Series([secret] * 9, name="outcome")
        with self.assertRaises(GroupForwardEvaluationError) as caught:
            preview(arguments)
        self.assertNotIn(secret, str(caught.exception))
        arguments = handwritten_fixture()
        arguments["row_groups"].iat[0] = secret
        with self.assertRaises(GroupForwardEvaluationError) as caught:
            preview(arguments)
        self.assertNotIn(secret, str(caught.exception))


class ForwardEvaluationIdentityAndFailureTests(TemporalContractTestCase):
    def test_typed_integer_endpoints_whitespace_and_nul_ids_survive_surrogate_metrics(self):
        identities = ["warm", 1, "1", "01", " spaced ", "nul\x00inside", MIN_TICK, MAX_TICK, 1, "1"]
        values = [0., 1., 2., 3., 4., 5., 6., 7., 3., 2.]
        unique = list(dict.fromkeys(identities))
        plan = plan_group_forward_windows(
            identities,
            group_spans=[GroupSpan("warm", 0, 0, 0)] + [GroupSpan(group, 10, 11, 11) for group in unique[1:]],
            windows=[TimeWindow(10, 15)], time_unit="day", clock_id="typed-handwritten",
        )
        arguments = dict(
            X=pd.DataFrame({"x": values}), y=pd.Series(values, name="outcome"),
            row_groups=pd.Series(identities, name="group", dtype=object), plan=plan,
            feature_allowlist=("x",), metadata_columns=("group", "start", "end", "label_available_at"),
        )
        result = evaluate(arguments)
        expected = tuple((type(group), group) for group in unique[1:])
        for recipe in result.metrics:
            self.assertEqual(tuple((type(row["group"]), row["group"]) for row in recipe["per_group"]), expected)
            self.assertEqual(recipe["pooled"]["n_groups"], 7)
            self.assertEqual(recipe["per_group"][0]["n_rows"], 2)
            self.assertEqual(recipe["per_group"][1]["n_rows"], 2)
            self.assertIsNone(recipe["per_group"][1]["r2"])
        self.assertEqual(tuple((type(row["group"]), row["group"]) for row in result.predictions),
                         tuple((type(group), group) for group in identities[1:]))
        self.assertEqual(result.coverage["tested"]["groups"], tuple(unique[1:]))

    def test_temporal_cutoff_equalities_crossings_and_delayed_labels_are_retained(self):
        groups = ["warm", "gap_equal", "label_equal", "delayed_test", "cross_lower", "cross_upper", "late"]
        spans = [GroupSpan("warm", 0, 1, 7), GroupSpan("gap_equal", 0, 1, 8),
                 GroupSpan("label_equal", 0, 1, 10), GroupSpan("delayed_test", 10, 11, 30),
                 GroupSpan("cross_lower", 9, 11, 11), GroupSpan("cross_upper", 11, 15, 15),
                 GroupSpan("late", 40, 41, 41)]
        plan = plan_group_forward_windows(
            groups, group_spans=spans, windows=[TimeWindow(10, 15)],
            time_unit="day", clock_id="edge-handwritten", gap=2,
        )
        result = preview(dict(
            X=pd.DataFrame({"x": [0.] * 7}), y=pd.Series([1.] * 7, name="outcome"),
            row_groups=pd.Series(groups, dtype=object, name="group"), plan=plan,
            feature_allowlist=("x",), metadata_columns=METADATA,
        ))
        fold = result.plan["folds"][0]
        self.assertEqual(fold["train"]["groups"], ("warm",))
        self.assertEqual(fold["test"]["groups"], ("delayed_test",))
        self.assertEqual(fold["unused"]["groups"], ("gap_equal", "label_equal", "cross_lower", "cross_upper", "late"))
        reasons = {row["group"]: row for row in fold["exclusions"]}
        self.assertEqual(reasons["gap_equal"]["train"], "gap_not_elapsed_before_window")
        self.assertEqual(reasons["label_equal"]["train"], "label_not_available_before_window")
        self.assertEqual(reasons["cross_lower"]["test"], "crosses_lower_boundary")
        self.assertEqual(reasons["cross_upper"]["test"], "crosses_upper_boundary")
        self.assertIsNone(reasons["delayed_test"]["test"])
        self.assertEqual(result.coverage["untested"]["n_rows"], 6)

    def test_last_fit_failure_reports_attempted_completed_counts_without_retry_or_partial_result(self):
        arguments = handwritten_fixture()
        digest = preview(arguments).bindings["execution_sha256"]
        original_make = temporal.make_pipeline
        pipelines = []

        def make(*args, **kwargs):
            pipeline = original_make(*args, **kwargs)
            pipelines.append(pipeline)
            if len(pipelines) == 4:
                def fail(*_args, **_kwargs):
                    raise ValueError("handwritten-final-fit-failure")
                pipeline.fit = fail
            return pipeline

        with patch.object(temporal, "make_pipeline", make), \
                patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("unexpected write")) as write:
            with self.assertRaises(GroupForwardEvaluationError) as caught:
                evaluate(arguments, digest)
        self.assertEqual(len(pipelines), 4)
        self.assertEqual(caught.exception.attempted_fits, 4)
        self.assertEqual(caught.exception.completed_fits, 3)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertFalse(hasattr(caught.exception, "result"))
        write.assert_not_called()

    def test_model_construction_failure_does_not_claim_a_pipeline_fit(self):
        arguments = handwritten_fixture()
        digest = preview(arguments).bindings["execution_sha256"]
        with patch.object(temporal, "make_pipeline", side_effect=ValueError("handwritten-construction-failure")) as make:
            with self.assertRaises(GroupForwardEvaluationError) as caught:
                evaluate(arguments, digest)
        self.assertEqual(make.call_count, 1)
        self.assertEqual(caught.exception.attempted_fits, 0)
        self.assertEqual(caught.exception.completed_fits, 0)

    def test_nonfinite_learned_state_and_ridge_iteration_cap_fail_closed(self):
        for label in ("imputer", "scaler", "dummy", "ridge_coef", "ridge_intercept", "ridge_cap"):
            with self.subTest(state=label):
                arguments = handwritten_fixture()
                digest = preview(arguments).bindings["execution_sha256"]
                original_make = temporal.make_pipeline
                calls = []

                def make(*args, **kwargs):
                    pipeline = original_make(*args, **kwargs)
                    calls.append(pipeline)
                    original_fit = pipeline.fit
                    position = len(calls)

                    def fit(X, y, **fit_kwargs):
                        result = original_fit(X, y, **fit_kwargs)
                        numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
                        model = pipeline.named_steps["model"]
                        if position == 1 and label == "imputer":
                            numeric.named_steps["impute"].statistics_[0] = np.inf
                        elif position == 1 and label == "scaler":
                            numeric.named_steps["scale"].mean_[0] = np.inf
                        elif position == 1 and label == "dummy":
                            model.constant_[0, 0] = np.inf
                        elif position == 2 and label == "ridge_coef":
                            model.coef_[0] = np.inf
                        elif position == 2 and label == "ridge_intercept":
                            model.intercept_ = np.inf
                        elif position == 2 and label == "ridge_cap":
                            model.n_iter_ = np.asarray([1000], dtype=np.int64)
                        return result

                    pipeline.fit = fit
                    return pipeline

                with patch.object(temporal, "make_pipeline", make):
                    with self.assertRaises(GroupForwardEvaluationError) as caught:
                        evaluate(arguments, digest)
                expected = 2 if label.startswith("ridge") else 1
                self.assertEqual(len(calls), expected)
                self.assertEqual(caught.exception.attempted_fits, expected)
                self.assertEqual(caught.exception.completed_fits, expected)

    def test_nonfinite_or_wrong_shape_predictions_fail_without_replacement(self):
        for label, replacement in (
            ("nan", lambda n: np.full(n, np.nan)),
            ("inf", lambda n: np.full(n, np.inf)),
            ("short", lambda n: np.zeros(n - 1)),
            ("matrix", lambda n: np.zeros((n, 1))),
            ("complex", lambda n: np.zeros(n, dtype=complex)),
        ):
            with self.subTest(prediction=label):
                arguments = handwritten_fixture()
                digest = preview(arguments).bindings["execution_sha256"]
                original_make = temporal.make_pipeline
                calls = []

                def make(*args, **kwargs):
                    pipeline = original_make(*args, **kwargs)
                    calls.append(pipeline)
                    pipeline.predict = lambda X: replacement(len(X))
                    return pipeline

                with patch.object(temporal, "make_pipeline", make):
                    with self.assertRaises(GroupForwardEvaluationError) as caught:
                        evaluate(arguments, digest)
                self.assertEqual(len(calls), 1)
                self.assertEqual(caught.exception.attempted_fits, 1)
                self.assertEqual(caught.exception.completed_fits, 1)

    def test_metric_overflow_and_underflow_do_not_return_successful_results(self):
        for extreme in (1e308, 1e-200):
            with self.subTest(extreme=extreme):
                # Every training target is zero. The final test-only group's
                # residual squares outside the representable metric range.
                arguments = handwritten_fixture()
                arguments["y"].iloc[:] = 0.
                arguments["y"].iloc[4] = extreme
                digest = preview(arguments).bindings["execution_sha256"]
                with patch.object(temporal, "grouped_metrics", wraps=temporal.grouped_metrics) as metrics:
                    with self.assertRaises(GroupForwardEvaluationError) as caught:
                        evaluate(arguments, digest)
                self.assertGreater(metrics.call_count, 0)
                self.assertGreater(caught.exception.attempted_fits, 0)
                self.assertGreaterEqual(caught.exception.completed_fits, 1)

    def test_handwritten_small_resource_boundaries_reject_before_models(self):
        arguments = handwritten_fixture()
        exact_limits = {
            "MAX_ROWS": 9, "MAX_FEATURES": 1, "MAX_WINDOWS": 2,
            "MAX_WORK": 32, "MAX_MATRIX_CELLS": 28,
        }
        with ExitStack() as stack:
            for name, value in exact_limits.items():
                stack.enter_context(patch.object(temporal, name, value))
            at_boundary = preview(arguments)
        self.assertEqual(at_boundary.fits_performed, 0)
        self.assertEqual(at_boundary.resource_limits["matrix_cell_visits"], 28)
        for name, value in exact_limits.items():
            with self.subTest(limit=name), patch.object(temporal, name, value - 1):
                self.assert_rejected_before_pipeline(arguments)
        with patch.object(temporal, "MAX_PLAN_BYTES", 512):
            self.assert_rejected_before_pipeline(arguments)

    def test_internal_consistency_rejects_changed_coverage_identity_metrics_and_fit_counts(self):
        arguments = handwritten_fixture()
        result = evaluate(arguments)
        prepared = temporal._prepare(arguments["X"], arguments["y"], arguments["row_groups"],
                                     arguments["plan"], arguments["feature_allowlist"], arguments["metadata_columns"])
        temporal._validate_result(result, prepared)
        changes = []
        for label, transform in (
            ("missing_row", lambda rows: rows[:-1]),
            ("duplicate_row", lambda rows: rows + [deepcopy(rows[0])]),
            ("wrong_order", lambda rows: list(reversed(rows))),
            ("bool_position", lambda rows: [{**rows[0], "row_position": True}] + rows[1:]),
            ("untested_position", lambda rows: [{**rows[0], "row_position": 0}] + rows[1:]),
            ("typed_identity", lambda rows: [{**rows[0], "group": 1}] + rows[1:]),
            ("wrong_window", lambda rows: [{**rows[0], "window_index": 1}] + rows[1:]),
            ("bool_window", lambda rows: [{**rows[0], "window_index": False}] + rows[1:]),
            ("wrong_observed", lambda rows: [{**rows[0], "observed": 99.}] + rows[1:]),
            ("nonfinite_prediction", lambda rows: [{**rows[0], "prediction_dummy_mean": np.nan}] + rows[1:]),
            ("missing_recipe", lambda rows: [{key: value for key, value in rows[0].items()
                                              if key != "prediction_ridge_alpha_1"}] + rows[1:]),
        ):
            document = result.to_dict()
            changes.append((label, {"predictions": temporal._freeze(transform(document["predictions"]))}))
        document = result.to_dict()
        document["coverage"]["untested_predictions_present"] = True
        changes.append(("false_coverage", {"coverage": temporal._freeze(document["coverage"])}))
        document = result.to_dict()
        document["coverage"]["tested"]["n_rows"] = 9
        changes.append(("false_coverage_count", {"coverage": temporal._freeze(document["coverage"])}))
        document = result.to_dict()
        document["fit_budget"]["completed_total"] = 3
        changes.append(("false_fit_count", {"fit_budget": temporal._freeze(document["fit_budget"])}))
        document = result.to_dict()
        document["metrics"][0]["pooled"]["group_mae"] += 1
        changes.append(("changed_score", {"metrics": temporal._freeze(document["metrics"])}))
        document = result.to_dict()
        document["plan"]["fits_performed"] = 4
        changes.append(("changed_planner_fits", {"plan": temporal._freeze(document["plan"])}))
        changes.append(("bool_schema_version", {"schema_version": True}))
        for label, mutation in changes:
            with self.subTest(tamper=label):
                with self.assertRaises(GroupForwardEvaluationError):
                    temporal._validate_result(replace(result, **mutation), prepared)


if __name__ == "__main__":
    unittest.main()
