"""Check matched-pair target sensitivity on exact rational shares."""

from fractions import Fraction as F
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/cycle-history-mixture/run.py"


class CycleHistoryMixtureTests(unittest.TestCase):
    def test_command(self):
        got = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(got.splitlines(), [
            "uncycled share=1/2: candidate-minus-baseline=1/2",
            "uncycled share=4/5: candidate-minus-baseline=-2/5",
            "uncycled share=2/3: candidate-minus-baseline=0",
        ])

    def test_independent_equal_mass_expansion(self):
        mod = runpy.run_path(str(EXAMPLE))
        score, rows = mod["estimated_contrast"], mod["ROWS"]
        for total in range(1, 11):
            for count in range(total + 1):
                expanded = rows["uncycled"] * count + rows["cycled"] * (total - count)
                candidate = sum(a for a, _ in expanded)
                baseline = sum(b for _, b in expanded)
                self.assertEqual(score(F(count, total)), F(candidate - baseline, 2 * total))

    def test_target_share_outside_unit_interval_rejects(self):
        score = runpy.run_path(str(EXAMPLE))["estimated_contrast"]
        for p in (F(-1, 5), F(6, 5)):
            with self.assertRaises(ValueError):
                score(p)
