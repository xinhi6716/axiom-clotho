"""Synthetic candidate-order contracts; no predictive-performance claims."""

from collections import UserList
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import (GroupBenchRegressor, benchmark, fit_predict,
                        plan_benchmark, plan_prediction)
from groupbench.model import candidates_for, selection_plan


class RegressionBaselineOrderTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"x": np.arange(12, dtype=float),
                                   "target": np.ones(12),
                                   "batch": np.repeat([f"g{i}" for i in range(6)], 2)})
        self.X = self.frame[["x"]]
        self.groups = self.frame.batch.to_numpy()
        self.options = dict(target="target", group="batch", inner_splits=2, random_state=0)

    def invalid_inputs(self):
        return [None, True, 7, 1.0, "dummy", b"dummy", [], (),
                {"dummy", "ridge"}, frozenset(["dummy", "ridge"]),
                {"dummy": 1, "ridge": 2}, iter(["dummy", "ridge"]),
                np.array(["dummy", "ridge"]), ["dummy", "dummy"],
                ["dummy", None], [True], [1], [["dummy"]], [{"dummy": 1}]]

    def test_unordered_and_malformed_inputs_raise_actionable_value_error(self):
        for baselines in self.invalid_inputs():
            with self.subTest(baselines=repr(baselines)):
                with self.assertRaisesRegex(ValueError, "baselines.*ordered sequence"):
                    candidates_for(baselines)

    def test_unknown_ordered_family_keeps_existing_error(self):
        for baselines in (["unsupported"], ("dummy", "unsupported")):
            with self.subTest(baselines=baselines):
                with self.assertRaisesRegex(ValueError, "Unknown baseline.*dummy.*extra_trees.*ridge"):
                    candidates_for(baselines)

    def test_supported_ordered_sequences_keep_caller_order_and_fixed_grids(self):
        order = ["extra_trees", "dummy", "ridge"]
        expected = ["extra_trees_depth_6", "extra_trees_depth_none", "dummy_mean",
                    "ridge_alpha_0.1", "ridge_alpha_1", "ridge_alpha_10"]
        for container in (list, tuple, UserList):
            with self.subTest(container=container.__name__):
                baselines = container(order)
                before = deepcopy(baselines)
                candidates = candidates_for(baselines)
                self.assertEqual([candidate.name for candidate in candidates], expected)
                self.assertEqual(baselines, before)
                plan = selection_plan(self.X, self.groups, baselines=baselines, inner_splits=2,
                                      random_state=0, objective="group_mae")
                self.assertEqual([candidate.name for candidate in plan.candidates], expected)
                self.assertEqual(plan.refit_families, tuple(order))
                self.assertEqual(plan.fit_counts, {"inner_selection_fits": 12,
                                                  "family_refits": 3, "total_fits": 15})

    def test_exact_ties_respect_declared_order_for_lists_and_tuples(self):
        for container in (list, tuple):
            for order in (("dummy", "ridge"), ("ridge", "dummy")):
                with self.subTest(container=container.__name__, order=order):
                    model = GroupBenchRegressor(baselines=container(order), inner_splits=2)
                    model.fit(self.X, self.frame.target, self.groups)
                    self.assertEqual(model.best_family_, order[0])
                    self.assertEqual(model.best_candidate_, "dummy_mean" if order[0] == "dummy" else "ridge_alpha_0.1")
                    self.assertEqual(list(model.family_models_), list(order))
                    self.assertEqual({record["selection_score"] for record in model.selection_results_}, {0.0})
                    np.testing.assert_array_equal(model.predict(self.X), np.ones(12))

    def test_invalid_inputs_fail_before_fit_in_all_public_workflows(self):
        for baselines in self.invalid_inputs():
            actions = [
                lambda: GroupBenchRegressor(baselines=baselines, inner_splits=2).fit(
                    self.X, self.frame.target, self.groups),
                lambda: benchmark(self.frame, baselines=baselines, outer_splits=2, **self.options),
                lambda: fit_predict(self.frame, self.X, baselines=baselines, **self.options),
                lambda: plan_benchmark(self.frame, baselines=baselines, outer_splits=2, **self.options),
                lambda: plan_prediction(self.frame, self.X, baselines=baselines, **self.options),
            ]
            for index, action in enumerate(actions):
                with self.subTest(baselines=repr(baselines), workflow=index):
                    with patch("groupbench.model.make_pipeline") as factory:
                        with self.assertRaisesRegex(ValueError, "baselines.*ordered sequence"):
                            action()
                        factory.assert_not_called()

    def test_invalid_refit_preserves_complete_previous_fitted_state(self):
        model = GroupBenchRegressor(baselines=("dummy", "ridge"), inner_splits=2)
        model.fit(self.X, self.frame.target, self.groups)
        before = {key: value for key, value in vars(model).items() if key.endswith("_")}
        predictions = model.predict_baselines(self.X)
        model.set_params(baselines={"dummy", "ridge"})
        with patch("groupbench.model.make_pipeline") as factory:
            with self.assertRaisesRegex(ValueError, "ordered sequence"):
                model.fit(self.X.rename(columns={"x": "new_x"}), self.frame.target + 1, self.groups)
            factory.assert_not_called()
        current = {key: value for key, value in vars(model).items() if key.endswith("_")}
        self.assertEqual(set(current), set(before))
        for key, value in before.items():
            self.assertIs(current[key], value, key)
        for family, values in model.predict_baselines(self.X).items():
            np.testing.assert_array_equal(values, predictions[family])


if __name__ == "__main__":
    unittest.main()
