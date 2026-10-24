"""Synthetic estimator lifecycle checks; no external data or model files."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from groupbench import GroupBenchRegressor
import groupbench.model as model_module


class RegressionFitStateTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"x": np.arange(12, dtype=float)})
        self.y = 2 * self.X.x.to_numpy() + 1
        self.groups = np.repeat([f"old-{i}" for i in range(6)], 2)
        self.incoming = self.X.rename(columns={"x": "new_x"})
        self.new_y = self.y + 100
        self.new_groups = np.repeat([f"new-{i}" for i in range(6)], 2)

    def fresh(self):
        return GroupBenchRegressor(baselines=("dummy", "ridge"), inner_splits=2)

    def fitted(self):
        return self.fresh().fit(self.X, self.y, self.groups)

    def snapshot(self, model):
        return {
            "attributes": {key: value for key, value in vars(model).items() if key.endswith("_")},
            "audit": deepcopy(model.inner_audit_),
            "selection": deepcopy(model.selection_results_),
            "family_candidates": dict(model.family_candidates_),
            "family_models": dict(model.family_models_),
            "predictions": model.predict(self.X),
            "baselines": model.predict_baselines(self.X),
        }

    def assert_preserved(self, model, before):
        state = {key: value for key, value in vars(model).items() if key.endswith("_")}
        self.assertEqual(set(state), set(before["attributes"]))
        for key, value in before["attributes"].items():
            self.assertIs(state[key], value, key)
        self.assertEqual(model.inner_audit_, before["audit"])
        self.assertEqual(model.selection_results_, before["selection"])
        self.assertEqual(model.family_candidates_, before["family_candidates"])
        self.assertEqual(model.family_models_, before["family_models"])
        np.testing.assert_array_equal(model.predict(self.X), before["predictions"])
        for key, value in model.predict_baselines(self.X).items():
            np.testing.assert_array_equal(value, before["baselines"][key])

    def failing_call(self, original, position, exception=ValueError):
        calls = []

        def wrapped(*args, **kwargs):
            calls.append(None)
            if len(calls) == position:
                raise exception("synthetic lifecycle interruption")
            return original(*args, **kwargs)

        return wrapped, calls

    def test_every_pipeline_fit_failure_preserves_complete_previous_state(self):
        # Four candidates x two inner folds, then two family refits.
        for position in range(1, 11):
            with self.subTest(fit=position):
                model = self.fitted()
                before = self.snapshot(model)
                fail, calls = self.failing_call(Pipeline.fit, position)
                with patch.object(Pipeline, "fit", fail), self.assertRaisesRegex(ValueError, "synthetic"):
                    model.fit(self.incoming, self.new_y, self.new_groups)
                self.assertEqual(len(calls), position)
                self.assert_preserved(model, before)

    def test_every_first_fit_failure_leaves_no_learned_attributes(self):
        for position in range(1, 11):
            with self.subTest(fit=position):
                model = self.fresh()
                fail, calls = self.failing_call(Pipeline.fit, position)
                with patch.object(Pipeline, "fit", fail), self.assertRaisesRegex(ValueError, "synthetic"):
                    model.fit(self.incoming, self.new_y, self.new_groups)
                self.assertEqual(len(calls), position)
                self.assertFalse(any(key.endswith("_") for key in vars(model)))
                for action in (lambda: check_is_fitted(model),
                               lambda: model.predict(self.X),
                               lambda: model.predict_baselines(self.X)):
                    with self.assertRaises(NotFittedError):
                        action()

    def test_inner_prediction_failure_preserves_previous_state(self):
        for position in range(1, 9):
            with self.subTest(prediction=position):
                model = self.fitted()
                before = self.snapshot(model)
                fail, calls = self.failing_call(Pipeline.predict, position)
                with patch.object(Pipeline, "predict", fail), self.assertRaisesRegex(ValueError, "synthetic"):
                    model.fit(self.incoming, self.new_y, self.new_groups)
                self.assertEqual(len(calls), position)
                self.assert_preserved(model, before)

    def test_inner_or_pooled_metric_failure_preserves_previous_state(self):
        for position in (1, 3, 12):
            with self.subTest(metric=position):
                model = self.fitted()
                before = self.snapshot(model)
                fail, calls = self.failing_call(model_module.grouped_metrics, position)
                with patch.object(model_module, "grouped_metrics", fail), self.assertRaisesRegex(ValueError, "synthetic"):
                    model.fit(self.incoming, self.new_y, self.new_groups)
                self.assertEqual(len(calls), position)
                self.assert_preserved(model, before)

    def test_real_metric_arithmetic_failure_preserves_previous_state(self):
        model = self.fitted()
        before = self.snapshot(model)
        with self.assertRaisesRegex(ValueError, "Metric arithmetic"):
            model.fit(self.incoming, np.tile([0.0, 1e-200], 6), self.new_groups)
        self.assert_preserved(model, before)

    def test_interrupted_last_refit_preserves_previous_state(self):
        model = self.fitted()
        before = self.snapshot(model)
        fail, calls = self.failing_call(Pipeline.fit, 10, KeyboardInterrupt)
        with patch.object(Pipeline, "fit", fail), self.assertRaises(KeyboardInterrupt):
            model.fit(self.incoming, self.new_y, self.new_groups)
        self.assertEqual(len(calls), 10)
        self.assert_preserved(model, before)

    def test_validation_failures_do_not_start_fitting_or_change_state(self):
        cases = [(self.incoming, np.full(12, np.nan), self.new_groups),
                 (self.incoming, self.new_y, ["only-one-group"] * 12)]
        for X, y, groups in cases:
            with self.subTest(groups=groups[0]):
                model = self.fitted()
                before = self.snapshot(model)
                with patch.object(Pipeline, "fit") as fitting, self.assertRaises(ValueError):
                    model.fit(X, y, groups)
                fitting.assert_not_called()
                self.assert_preserved(model, before)

    def test_success_after_failure_replaces_all_state_and_matches_fresh_fit(self):
        model = self.fitted()
        before = self.snapshot(model)
        fail, _ = self.failing_call(Pipeline.fit, 10)
        with patch.object(Pipeline, "fit", fail), self.assertRaises(ValueError):
            model.fit(self.incoming, self.new_y, self.new_groups)
        self.assertIs(model.fit(self.incoming, self.new_y, self.new_groups), model)
        expected = self.fresh().fit(self.incoming, self.new_y, self.new_groups)
        self.assertIsNot(model.best_estimator_, before["attributes"]["best_estimator_"])
        self.assertEqual(model.feature_names_in_.tolist(), ["new_x"])
        self.assertEqual(model.inner_audit_, expected.inner_audit_)
        self.assertEqual(model.selection_results_, expected.selection_results_)
        self.assertEqual(model.family_candidates_, expected.family_candidates_)
        self.assertEqual(model.best_candidate_, expected.best_candidate_)
        self.assertEqual(model.n_groups_seen_, expected.n_groups_seen_)
        for key, value in model.predict_baselines(self.incoming).items():
            np.testing.assert_array_equal(value, expected.predict_baselines(self.incoming)[key])
        with self.assertRaisesRegex(ValueError, "exactly match"):
            model.predict(self.X)


if __name__ == "__main__":
    unittest.main()
