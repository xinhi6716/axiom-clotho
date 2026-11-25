"""Check the documented geometry and training-label nonidentifiability witness."""

from fractions import Fraction as F
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/joint-feature-support/run.py"


class JointFeatureSupportTests(unittest.TestCase):
    def test_cli(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "outside: ranges=True, hull=False, world gap=11",
            "inside: ranges=True, hull=True, world gap=1",
        ])

    def test_triangle_boundary_and_axis_range(self):
        check = runpy.run_path(str(EXAMPLE))["coordinate_checks"]
        self.assertEqual(check((F(1, 2), F(1, 2))), (True, True))
        self.assertEqual(check((1, 1)), (True, False))
        self.assertEqual(check((F(-1, 10), 0)), (False, False))

    def test_worlds_agree_on_all_observed_rows_only(self):
        mod = runpy.run_path(str(EXAMPLE))
        self.assertEqual([mod["worlds"](p) for p in mod["TRAIN"]], [(0, 0)] * 3)
        self.assertEqual(mod["worlds"](mod["OUTSIDE"]), (0, 11))
        self.assertEqual(mod["worlds"](mod["INSIDE"]), (0, 1))
