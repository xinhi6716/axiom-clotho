"""Human leaderboard rendering must not turn small errors into an exact zero."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from groupbench.cli import main


class RegressionMetricDisplayTests(unittest.TestCase):
    def run_dummy(self, outcomes):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw, result = folder / "source.csv", folder / "result"
            raw.write_text("group,x,y\n" + "".join(
                f"g{i % 4},{i},{outcome}\n" for i, outcome in enumerate(outcomes)
            ), encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["run", str(raw), "--target", "y", "--group", "group",
                             "--baselines", "dummy", "--outer-splits", "2",
                             "--inner-splits", "2", "--out", str(result)])
            self.assertEqual(code, 0, stderr.getvalue())
            selected = next(line.split()[1:4] for line in stdout.getvalue().splitlines()
                            if line.split() and line.split()[0] == "selected")
            summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
            return selected, summary["models"]["selected"]["overall"]

    def test_real_small_nonzero_errors_remain_visible(self):
        cells, metrics = self.run_dummy(i * 1e-9 for i in range(8))
        for printed, key in zip(cells, ("group_mae", "mae", "rmse")):
            self.assertGreater(metrics[key], 0.0)
            self.assertEqual(printed, f"{metrics[key]:.4e}")

    def test_exact_zero_retains_existing_display(self):
        cells, metrics = self.run_dummy([0] * 8)
        self.assertEqual(cells, ["0.0000"] * 3)
        self.assertEqual([metrics[k] for k in ("group_mae", "mae", "rmse")], [0.0] * 3)
