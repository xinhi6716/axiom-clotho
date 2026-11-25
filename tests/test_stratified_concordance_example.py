"""Exact tests for the paired rank witness, not a population CI."""

from fractions import Fraction
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/stratified-concordance/run.py"


class StratifiedConcordanceExampleTests(unittest.TestCase):
    def test_documented_commands(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True).splitlines()
        self.assertEqual(out, [
            "candidate: pooled=3/5, within=0, between=1",
            "constant: pooled=1/2, within=1/2, between=1/2",
            "origin-stratum baseline: pooled=4/5, within=1/2, between=1",
        ])

    def test_roster_order_does_not_change_pair_counts(self):
        mod = runpy.run_path(str(EXAMPLE))
        rows = mod["fixture"]((3, 2, 1, 12, 11, 10))
        result = mod["report"](rows)
        self.assertEqual(result, mod["report"](rows[::-1]))
        self.assertEqual(result["pooled"], ((9, 6, 0, 0), Fraction(3, 5)))
        self.assertEqual(result["within"], ((0, 6, 0, 0), Fraction(0)))

    def test_outcome_tie_is_not_a_prediction_tie_or_eligible_pair(self):
        report = runpy.run_path(str(EXAMPLE))["report"]
        got = report(((0, 1, 10), (0, 1, 1), (1, 2, 10)))
        self.assertEqual(got["within"], ((0, 0, 0, 1), None))
        self.assertEqual(got["between"], ((1, 0, 1, 0), Fraction(3, 4)))
