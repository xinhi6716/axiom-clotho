from pathlib import Path
from runpy import run_path
from unittest import TestCase

demo = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                    / "shared-shift-loss-order" / "run.py"))["demonstrate"]


class SharedShiftLossOrderTests(TestCase):
    def test_robust_opposite_rankings(self):
        row = demo()["opposite_robust_rankings"]
        self.assertEqual(row["mae_range"], ["1/6", "1/3"])
        self.assertEqual(row["mse_range"], ["-13/6", "-7/6"])

    def test_endpoints_miss_mae_extremum(self):
        row = demo()["interior_mae_extremum"]
        self.assertEqual(row["mae_endpoints"], ["0", "0"])
        self.assertEqual(row["mae_range"], ["0", "1/2"])
        self.assertEqual(row["mse_range"], ["1/4", "1/4"])
