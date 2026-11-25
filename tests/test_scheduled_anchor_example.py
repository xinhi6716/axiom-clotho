"""Exercise exact nominal selection and observable non-substitution at the CLI."""

from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/scheduled-anchor-vs-first-after/run.py"


class ScheduledAnchorTests(unittest.TestCase):
    def test_documented_cli(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "exact nominal terminal: actual=100011 value=3",
            "first after clock cutoff: nominal=90000 actual=100001 value=10",
            "missing nominal terminal: exact=None",
        ])

    def test_clock_statistic_does_not_fill_missing_nominal_identity(self):
        m = runpy.run_path(str(EXAMPLE))
        rows, exact, clock = m["ROWS"], m["scheduled"], m["first_after"]
        self.assertEqual(exact(rows[::-1]), exact(rows))
        self.assertIsNone(exact(rows[:-1]))
        self.assertEqual(clock(rows[:-1], 100_000), clock(rows, 100_000))
        self.assertEqual((exact(rows)[-1][2] - 7, clock(rows, 100_000)[2] - 7),
                         (-4, 3))

    def test_duplicate_and_inconsistent_actual_time_reject(self):
        m = runpy.run_path(str(EXAMPLE))
        rows, exact = m["ROWS"], m["scheduled"]
        with self.assertRaisesRegex(ValueError, "Duplicate nominal anchor"):
            exact(rows + ((100_000, 100_012, 2),))
        with self.assertRaisesRegex(ValueError, "not strictly increasing"):
            exact(rows[:-1] + ((100_000, 1, 3),))
