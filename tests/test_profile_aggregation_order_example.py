"""Check the exact CLI and the area invariant without measured data."""

from fractions import Fraction as F
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/profile-aggregation-order/run.py"


class ProfileAggregationOrderTests(unittest.TestCase):
    def test_cli(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "A: peak-of-mean=2, mean-of-peaks=4, area=4",
            "B: peak-of-mean=3, mean-of-peaks=3, area=4",
        ])

    def test_section_integrals_and_reversal(self):
        mod = runpy.run_path(str(EXAMPLE))
        summarize, a, b = mod["summaries"], mod["A"], mod["B"]
        # All endpoints vanish; the independent trapezoid oracle sums interiors.
        areas = [sum(row[1:-1]) for row in a + b]
        self.assertEqual(areas, [4, 4, 4, 4])
        sa, sb = summarize(a[::-1]), summarize(b)
        self.assertEqual((sa[0] - sb[0], sa[1] - sb[1], sa[2] - sb[2]),
                         (F(-1), F(1), F(0)))
