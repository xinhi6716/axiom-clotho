"""Check shared-control covariance against linear coefficient identities."""

from fractions import Fraction as F
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/plate-shared-control/run.py"


class PlateSharedControlTests(unittest.TestCase):
    def test_cli(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "contrast variances: 1, 1; covariance: 1/2",
            "variance of mean contrast: 3/4; assuming independence: 1/2",
        ])

    def test_orthogonal_outcome_coefficient_oracle(self):
        # Independent unit-variance outcomes in order c, a, b on each plate.
        a = tuple(F(x, 2) for x in (-1, 1, 0) * 2)
        b = tuple(F(x, 2) for x in (-1, 0, 1) * 2)
        dot = lambda left, right: sum(x * y for x, y in zip(left, right))
        mean = tuple((x + y) / 2 for x, y in zip(a, b))
        mod = runpy.run_path(str(EXAMPLE))
        self.assertEqual(mod["moments"](),
                         (dot(a, a), dot(b, b), dot(a, b),
                          dot(mean, mean), (dot(a, a) + dot(b, b)) / 4))
