from pathlib import Path
from runpy import run_path
from unittest import TestCase

case = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                    / "finite-aperture-peak-alias" / "run.py"))["case"]


class FiniteAperturePeakAliasTests(TestCase):
    def test_equal_full_apertures_do_not_determine_peak(self):
        report = case()
        self.assertEqual({v["full_aperture_mean"] for v in report.values()}, {"1"})
        self.assertEqual({v["peak"] for v in report.values()}, {1, 2, 4})

    def test_partially_resolved_apertures_still_hide_threshold_crossing(self):
        a, b, c = (case()[k] for k in ("constant", "sharp", "broad"))
        self.assertEqual(a["half_aperture_means"], ["1", "1"])
        self.assertEqual(b["half_aperture_means"], c["half_aperture_means"])
        self.assertEqual(b["half_aperture_means"], ["2", "0"])
        self.assertEqual((b["fraction_above_two"], c["fraction_above_two"]), ("1/4", "0"))
