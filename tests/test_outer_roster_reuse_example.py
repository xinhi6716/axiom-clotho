"""Verify the exact holdout-reuse witness without external data."""

from fractions import Fraction
from math import comb
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/outer-roster-reuse/run.py"


class OuterRosterReuseTests(unittest.TestCase):
    def test_documented_cli_outputs(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "fixed zero: reused=1/2, fresh=1/2",
            "two constants: reused=5/16, fresh=1/2",
            "all 16 vectors: reused=0, fresh=1/2",
        ])

    def test_binomial_oracle_and_fresh_null_for_every_previous_vector(self):
        mod = runpy.run_path(str(EXAMPLE))
        labels, constants = mod["LABELS"], mod["CONSTANTS"]
        self.assertEqual(len(labels), 16)
        self.assertEqual(len(set(labels)), 16)
        expected = Fraction(sum(comb(4, s) * min(s, 4 - s) for s in range(5)), 64)
        self.assertEqual(mod["expected_losses"](constants), (expected, Fraction(1, 2)))
        for old in labels:
            self.assertEqual(mod["select"](labels, old), old)
            for family in (constants[:1], constants, labels):
                selected = mod["select"](family, old)
                # Count a half of fresh worlds disagreeing at each coordinate,
                # independently of which old outcomes and candidate were used.
                fresh_mismatches = sum(sum(p != y for p, y in zip(selected, fresh))
                                       for fresh in labels)
                self.assertEqual(fresh_mismatches, 32)

    def test_tie_resolution_not_candidate_order(self):
        mod = runpy.run_path(str(EXAMPLE))
        constants = mod["CONSTANTS"]
        self.assertEqual(mod["select"](constants[::-1], (0, 1, 1, 0)), (0, 0, 0, 0))
