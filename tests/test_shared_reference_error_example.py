"""Exercise endpoint semantics and the exact covariance contrast witness."""

from fractions import Fraction
from itertools import product
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/shared-reference-error/run.py"


class SharedReferenceErrorTests(unittest.TestCase):
    def test_documented_cli(self):
        output = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(output.splitlines(), [
            "shared reference: 0, 1",
            "separate reference: 2, 1",
            "physical target: 1, 0",
            "shared-reference duplicate disagreement: 0",
        ])

    def test_covariance_oracle(self):
        m = runpy.run_path(str(EXAMPLE))["scores"]()
        differences = {name: pair[0] - pair[1] for name, pair in m.items() if len(pair) == 2}
        for name, es in (("shared reference", lambda b, c: b),
                         ("separate reference", lambda b, c: c)):
            correction = Fraction(-2 * sum(b * es(b, c)
                                           for b, c in product((-1, 1), repeat=2)), 4)
            self.assertEqual(differences[name], differences["physical target"] + correction)

    def test_perfect_duplicate_agreement_coexists_with_nonzero_physical_error(self):
        m = runpy.run_path(str(EXAMPLE))["scores"]()
        self.assertEqual(m["shared-reference duplicate disagreement"], (0,))
        self.assertGreater(m["physical target"][0], m["shared reference"][0])
