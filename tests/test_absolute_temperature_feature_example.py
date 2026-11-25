"""Check exact affine unit semantics for the documented teaching example."""

from fractions import Fraction as F
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/absolute-temperature-feature/run.py"


class AbsoluteTemperatureTests(unittest.TestCase):
    def test_cli_sign_and_representation(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "converted Celsius: 20000/29594369",
            "declared kelvin: 20000/29594369",
            "raw Celsius anti-pattern: -2/25",
        ])

    def test_absolute_representations_at_zero_and_on_both_sides(self):
        inv = runpy.run_path(str(EXAMPLE))["inverse_temperature"]
        for deg_c in (-25, 0, 25):
            kelvin = F(27315 + 100 * deg_c, 100)
            self.assertEqual(inv(deg_c, "degC", "absolute_temperature"), 1 / kelvin)
            self.assertEqual(inv(kelvin, "K", "absolute_temperature"), 1 / kelvin)

    def test_wrong_quantity_unknown_unit_and_absolute_zero_reject(self):
        inv = runpy.run_path(str(EXAMPLE))["inverse_temperature"]
        for row in ((25, "degC", "increment"), (25, "", "absolute_temperature"),
                    (0, "K", "absolute_temperature"), (-274, "degC", "absolute_temperature")):
            with self.subTest(row=row), self.assertRaises(ValueError):
                inv(*row)
