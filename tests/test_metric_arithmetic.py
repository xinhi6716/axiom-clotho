"""Invented float64 edge cases; no model fitting or external data needed."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from groupbench.cli import main
from groupbench.metrics import grouped_metrics, regression_metrics


def original_metrics(y, pred):
    """Pre-guard arithmetic for exact ordinary-value compatibility checks."""
    residual = pred - y
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "n_rows": len(y),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / total) if len(y) > 1 and total > 0 else None,
        "bias": float(np.mean(residual)),
    }


class MetricArithmeticTests(unittest.TestCase):
    def test_ordinary_values_keep_exact_scores_and_group_weighting(self):
        rng = np.random.default_rng(107)
        groups = np.array(["a"] * 7 + ["b"] * 3 + ["c"] * 2)
        for scale in (1e-100, 1.0, 1e100):
            for _ in range(5):
                y = rng.normal(size=len(groups)) * scale
                pred = y + rng.normal(size=len(groups)) * scale
                expected = original_metrics(y, pred)
                with self.subTest(scale=scale):
                    self.assertEqual(regression_metrics(y, pred), expected)
                    scores, rows = grouped_metrics(y, pred, groups)
                    expected_rows = [
                        {"group": group, **original_metrics(y[groups == group], pred[groups == group])}
                        for group in ("a", "b", "c")
                    ]
                    self.assertEqual(rows, expected_rows)
                    expected.update({
                        "n_groups": 3,
                        "group_mae": float(np.mean([row["mae"] for row in expected_rows])),
                        "group_rmse": float(np.mean([row["rmse"] for row in expected_rows])),
                        "worst_group_mae": max(row["mae"] for row in expected_rows),
                    })
                    self.assertEqual(scores, expected)
                    json.dumps([scores, rows], allow_nan=False)

    def test_huge_finite_values_fail_before_nonfinite_scores_escape(self):
        cases = {
            "residual": ([-1e308], [1e308]),
            "squared_residual": ([0, 1], [1e200, 1e200]),
            "squared_error_sum": ([0] * 4, [1e154] * 4),
            "target_mean": ([1e308, 9e307], [1e308, 9e307]),
            "target_variance": ([-1e200, 1e200], [-1e200, 1e200]),
            "r2_ratio": ([0, 1e-150], [1e150, 1e150]),
        }
        for name, (y, pred) in cases.items():
            with self.subTest(case=name), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaisesRegex(ValueError, "Metric arithmetic.*Rescale"):
                    regression_metrics(y, pred)

    def test_tiny_finite_values_are_not_silent_zero_or_undefined_scores(self):
        cases = {
            "target_variance": ([0, 1e-200], [0, 0]),
            "squared_residual": ([0, 1], [1e-200, 1]),
            "one_row_residual": ([0], [1e-200]),
            "r2_ratio": ([0, 1e150], [1e-150, 1e150]),
        }
        for name, (y, pred) in cases.items():
            with self.subTest(case=name), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaisesRegex(ValueError, "Metric arithmetic.*Rescale"):
                    regression_metrics(y, pred)

    def test_exact_constant_and_one_row_r2_remain_undefined(self):
        # The old mean could overflow (1e308) or create a tiny false variance
        # through summation roundoff (three identical 0.1 values).
        for value in (0.1, 1e308, -1e308, 1e-200, np.nextafter(0.0, 1.0)):
            for length in (1, 3):
                with self.subTest(value=value, length=length):
                    result = regression_metrics([value] * length, [value] * length)
                    self.assertEqual(result, {
                        "n_rows": length, "mae": 0.0, "rmse": 0.0,
                        "r2": None, "bias": 0.0,
                    })
                    json.dumps(result, allow_nan=False)
        self.assertIsNone(regression_metrics([0.1] * 3, [0.2] * 3)["r2"])
        self.assertIsNone(regression_metrics([1], [2])["r2"])

    def test_group_failure_is_not_dropped_or_replaced(self):
        with self.assertRaisesRegex(ValueError, "Metric arithmetic"):
            grouped_metrics([0, 1e-200, 1, 2], [0, 0, 1, 2], ["a", "a", "b", "b"])

    def test_group_aggregate_is_also_guarded(self):
        # Isolate the aggregation boundary even if upstream metric arithmetic
        # later changes to support a wider range of individually finite scores.
        row = {"n_rows": 1, "mae": 1e308, "rmse": 1e308, "r2": None, "bias": 0.0}
        with patch("groupbench.metrics.regression_metrics", side_effect=lambda *args: row.copy()):
            with self.assertRaisesRegex(ValueError, "Metric arithmetic"):
                grouped_metrics([0, 0], [0, 0], ["a", "b"])

    def test_caller_error_settings_cannot_disable_guard_and_are_restored(self):
        initial = np.geterr()
        with np.errstate(all="ignore"):
            ignored = np.geterr()
            with self.assertRaisesRegex(ValueError, "Metric arithmetic"):
                regression_metrics([0, 1e-200], [0, 0])
            self.assertEqual(np.geterr(), ignored)
            regression_metrics([1, 2], [2, 3])
            self.assertEqual(np.geterr(), ignored)
        self.assertEqual(np.geterr(), initial)

    def test_cli_aborts_inner_selection_without_writing_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "tiny.csv", root / "out"
            pd.DataFrame({
                "x": np.arange(12), "batch": np.repeat(list("abcdef"), 2),
                "target": [0.0, 1e-200] * 6,
            }).to_csv(source, index=False)
            # Exercise the actual selection/CLI/error path, not estimator
            # numerics: the pipeline performs no training and predicts zero.
            with patch("groupbench.model.make_pipeline") as factory:
                factory.return_value.predict.side_effect = lambda X: np.zeros(len(X))
                with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
                    code = main([
                        "run", str(source), "--target", "target", "--group", "batch",
                        "--out", str(output), "--outer-splits", "3", "--inner-splits", "2",
                        "--baselines", "dummy",
                    ])
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(factory.return_value.fit.call_count, 1)
            self.assertEqual(code, 2)
            self.assertIn("Metric arithmetic", stderr.getvalue())
            self.assertIn("Rescale", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
