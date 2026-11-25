"""Test exact observability at accelerated levels and use-level separation."""

from fractions import Fraction as F
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/accelerated-mechanism-alias/run.py"


class AcceleratedMechanismAliasTests(unittest.TestCase):
    def test_cli(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "accelerated 1: single=10, competing=10",
            "accelerated 2: single=12, competing=12",
            "use: single=25/3, competing=9",
        ])

    def test_use_hazard_from_independent_geometric_oracle(self):
        hazard = runpy.run_path(str(EXAMPLE))["hazards"]
        # Equal steps on the log-hazard scale imply h(0)h(2) = h(1)^2.
        single, competing = hazard(0)
        self.assertEqual(single, F(10**2, 12))
        self.assertEqual(competing, 9)
        self.assertNotEqual(competing * 12, 10**2)

    def test_mean_time_separates_without_float_survival(self):
        hazard = runpy.run_path(str(EXAMPLE))["hazards"]
        single, competing = hazard(0)
        self.assertEqual(1 / single - 1 / competing, F(2, 225))
