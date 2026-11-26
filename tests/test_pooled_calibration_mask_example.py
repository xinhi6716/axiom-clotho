from pathlib import Path
from runpy import run_path
from unittest import TestCase

demo = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                    / "pooled-calibration-mask" / "run.py"))["demonstrate"]


class PooledCalibrationMaskTests(TestCase):
    def test_case_mix_changes_calibration_without_changing_brier(self):
        cases = demo()
        self.assertEqual(cases["balanced"]["pooled_calibration_residual"], "0")
        self.assertEqual(cases["changed_mix"]["pooled_calibration_residual"], "-1/8")
        for case in cases.values():
            self.assertEqual(case["within_stratum_residuals"], ["-1/4", "1/4"])
            self.assertEqual(case["frozen_pooled_brier"], "1/4")
            self.assertEqual(case["frozen_stratum_brier"], "3/16")
