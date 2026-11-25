"""Check analytic coverage and calibration/scoring role separation."""

from fractions import Fraction as F
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/calibration-width-selection/run.py"


class CalibrationWidthSelectionTests(unittest.TestCase):
    def test_cli_and_primary_rank(self):
        lines = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True).splitlines()
        self.assertEqual(lines, [
            "19 calibration devices: fixed=19/20, width-selected=37/40",
            "exact example: selected center=0, half-width=49/100, covered=False",
            "prespecified center=1, half-width=69/100, covered=True",
        ])

    def test_formula_small_independent_integral_and_39_devices(self):
        mod = runpy.run_path(str(EXAMPLE))
        # n=1: integrate x over [0,1/2] and (1-x) over [1/2,1].
        self.assertEqual(mod["exact_coverages"](1), (F(1, 2), F(1, 4)))
        self.assertEqual(mod["exact_coverages"](39), (F(39, 40), F(77, 80)))

    def test_calibration_mutation_and_evaluation_mutation_are_distinct(self):
        mod = runpy.run_path(str(EXAMPLE))
        cal, select = mod["calibrate"], mod["width_selected"]
        old = mod["fixture"]()
        mirror = tuple(1 - y for y in old)
        self.assertEqual((select(old).center, select(mirror).center), (0, 1))

        first, second = cal(1, old), cal(1, mirror)
        self.assertEqual((first.center, first.score), (second.center, second.score))
        self.assertEqual((first.half_width, second.half_width), (F(69, 100), F(49, 100)))
        before = first.bounds
        self.assertTrue(mod["covered"](first, F(3, 5)))
        self.assertFalse(mod["covered"](first, F(1, 5)))
        self.assertEqual((first.center, first.score, first.bounds),
                         (1, "max-absolute-residual", before))
