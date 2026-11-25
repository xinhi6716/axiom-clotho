import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench.evaluation import benchmark
from groupbench.model import selection_plan
from groupbench.planning import plan_benchmark


class RegressionSplitPositionBudget(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "x": np.arange(12), "y": np.arange(12) + 1,
            "g": [f"g{i}" for i in range(12)],
            "site": [f"s{i % 3}" for i in range(12)],
        })
        self.options = dict(target="y", group="g", features=["x"],
                            outer_splits=3, inner_splits=2, baselines=("dummy",))

    def test_combined_group_positions_preflight_shared_by_plan_and_run(self):
        # Each outer fold keeps 12 positions; each has 8 training rows,
        # and its two inner folds keep 8 positions apiece. Total = 84.
        with patch("groupbench.model._MAX_RETAINED_SPLIT_POSITIONS", 83, create=True), patch(
            "groupbench.evaluation.GroupKFold.split",
            side_effect=AssertionError("must reject before allocating outer arrays"),
        ):
            for action in (plan_benchmark, benchmark):
                with self.subTest(action=action.__name__):
                    with self.assertRaisesRegex(ValueError, "84 retained split positions"):
                        action(self.frame, **self.options)
        with patch("groupbench.model._MAX_RETAINED_SPLIT_POSITIONS", 84, create=True):
            preview = plan_benchmark(self.frame, **self.options)
            result = benchmark(self.frame, **self.options)
        self.assertEqual(preview["fit_budget"]["total_fits"], 9)
        self.assertEqual(len(result.predictions), len(self.frame))

    def test_domain_counts_come_from_actual_outer_training_slices(self):
        opts = dict(self.options, domain="site", inner_strategy="leave_one_domain_out",
                    inner_splits=None)
        # Freeze a legitimate group-disjoint outer roster where the first
        # outer test removes all of one domain and remaining slices keep all.
        tests = (np.array([0, 3, 6, 9]), np.array([1, 2, 4, 5]), np.array([7, 8, 10, 11]))
        splits = tuple((np.setdiff1d(np.arange(12), test), test) for test in tests)
        with patch("groupbench.evaluation.GroupKFold.split", return_value=iter(splits)), patch(
            "groupbench.model._MAX_RETAINED_SPLIT_POSITIONS", 99, create=True
        ), patch("groupbench.model.domain_splits",
                 side_effect=AssertionError("must reject before allocating inner arrays")):
            with self.assertRaisesRegex(ValueError, "100 retained split positions"):
                plan_benchmark(self.frame, **opts)
        with patch("groupbench.evaluation.GroupKFold.split", return_value=iter(splits)), patch(
            "groupbench.model._MAX_RETAINED_SPLIT_POSITIONS", 100, create=True
        ):
            result = plan_benchmark(self.frame, **opts)
        self.assertEqual(result["config"]["inner_splits_per_outer"], [2, 3, 3])

    def test_standalone_inner_guard_before_group_or_domain_materialization(self):
        X, groups = self.frame[["x"]], self.frame["g"]
        common = dict(baselines=("dummy",), random_state=0, objective="group_mae")
        with patch("groupbench.model._MAX_RETAINED_SPLIT_POSITIONS", 23, create=True), patch(
            "groupbench.model.GroupKFold.split", side_effect=AssertionError("must not allocate")
        ), patch("groupbench.model.domain_splits", side_effect=AssertionError("must not allocate")):
            with self.assertRaisesRegex(ValueError, "24 retained split positions"):
                selection_plan(X, groups, inner_splits=2, **common)
            with self.assertRaisesRegex(ValueError, "36 retained split positions"):
                selection_plan(X, groups, inner_splits=None, **common,
                               inner_strategy="leave_one_domain_out", domains=self.frame["site"])

    def test_real_default_rejects_moderate_individual_but_large_combined_schedule(self):
        # Each independent component would be smaller than the cap but all
        # folds' inner arrays are retained together: 500 * (101 + 100 * 40).
        n = 500
        frame = pd.DataFrame({"x": np.arange(n), "y": np.arange(n), "g": np.arange(n)})
        options = dict(self.options, outer_splits=101, inner_splits=40)
        with patch("groupbench.evaluation.GroupKFold.split",
                   side_effect=AssertionError("must reject before allocating any split")):
            with self.assertRaisesRegex(ValueError, "2,050,500 retained split positions"):
                plan_benchmark(frame, **options)
