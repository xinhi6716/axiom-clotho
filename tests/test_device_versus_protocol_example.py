"""Run the documented protocol transport example against exact outcomes."""

from fractions import Fraction
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "device-versus-protocol" / "run.py"


class DeviceVersusProtocolExampleTests(unittest.TestCase):
    def test_documented_output(self):
        output = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True, cwd=ROOT)
        self.assertEqual(output.splitlines(), [
            "held devices at seen protocols: seen=20/20, lookup MAE=0, train-mean MAE=1",
            "held protocols: seen=0/20, lookup MAE=4/3, train-mean MAE=4/3",
        ])

    def test_exact_scores_do_not_depend_on_row_order(self):
        compare = runpy.run_path(str(EXAMPLE))["compare"]
        rows = [(p, d, y) for p, y in enumerate((-1, -1, 1, 1)) for d in range(5)]
        self.assertEqual(compare(rows[::-1], 1), (Fraction(0), Fraction(1), 20))
        self.assertEqual(compare(rows[::-1], 0), (Fraction(4, 3), Fraction(4, 3), 0))

    def test_same_physical_device_cannot_be_scored_twice(self):
        compare = runpy.run_path(str(EXAMPLE))["compare"]
        with self.assertRaisesRegex(AssertionError, "evaluated twice"):
            compare([(0, 0, 1), (0, 0, 2), (1, 1, 3)], 0)
