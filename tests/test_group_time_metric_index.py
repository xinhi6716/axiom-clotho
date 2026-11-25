"""Window metrics use one roster scan and preserve per-window row order."""
import unittest

import numpy as np

from groupbench.group_time_evaluation import _RECIPE_NAMES, _checked_grouped, _metrics
from groupbench.group_time_plan import GroupSpan, TimeWindow, plan_group_forward_windows


class WindowMetricIndexTests(unittest.TestCase):
    def test_interleaved_window_rows_use_one_index_without_changing_metrics(self):
        plan = plan_group_forward_windows(
            ["warm", "b", "a", "b", "a"],
            group_spans=[GroupSpan("warm", 0, 1, 1), GroupSpan("a", 3, 4, 4),
                         GroupSpan("b", 6, 7, 7)],
            windows=[TimeWindow(3, 5), TimeWindow(6, 8)],
            time_unit="tick", clock_id="synthetic",
        )
        reads = []

        class CountWindowReads(dict):
            def __getitem__(self, name):
                if name == "window_index":
                    reads.append(self.get("row_position"))
                return super().__getitem__(name)

        records = [CountWindowReads(row_position=i, group=plan.row_groups[i],
                                   window_index=(1, 0, 1, 0)[i - 1], observed=float(i),
                                   prediction_dummy_mean=1.5, prediction_ridge_alpha_1=i + 0.25)
                   for i in plan.tested.rows]
        ref_windows = ((1, 3), (0, 2))
        result = _metrics(records, plan)
        self.assertEqual(reads, list(plan.tested.rows))
        for recipe, got in zip(_RECIPE_NAMES, result):
            y = np.asarray([row["observed"] for row in records])
            pred = np.asarray([row["prediction_" + recipe] for row in records])
            g = [row["group"] for row in records]
            self.assertEqual((got["pooled"], got["per_group"]), _checked_grouped(y, pred, g))
            for window, rows in enumerate(ref_windows):
                idx = list(rows)
                expected, _ = _checked_grouped(y[idx], pred[idx], [g[i] for i in idx])
                self.assertEqual(got["per_window"][window], {"window_index": window, **expected})
