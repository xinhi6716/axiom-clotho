"""Check the synthetic dual-status ledger at its public command."""

from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/penetration-is-not-adhesion/run.py"


class PenetrationIsNotAdhesionTests(unittest.TestCase):
    def test_command(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "depth positive/zero/unavailable: (2, 1, 1)",
            "adhesion confirmed/failed/unknown: (2, 1, 1)",
            "confirmed adhesion per attempted track: 1/2",
        ])

    def test_same_depths_do_not_fix_adhesion(self):
        mod = runpy.run_path(str(EXAMPLE))
        rows, report = mod["ROWS"], mod["report"]
        modified = tuple((key, depth, "unknown") for key, depth, _ in rows)
        self.assertEqual(report(rows[::-1]), report(rows))
        self.assertEqual(report(modified), ((2, 1, 1), (0, 0, 4)))
        self.assertEqual(report((("zero", 0, "unknown"), ("missing", None, "unknown"))),
                         ((0, 1, 1), (0, 0, 2)))

    def test_duplicate_and_ambiguous_status_fail(self):
        mod = runpy.run_path(str(EXAMPLE))
        rows, report = mod["ROWS"], mod["report"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report(rows + rows[:1])
        with self.assertRaisesRegex(ValueError, "explicit adhesion"):
            report((("a", 0, None),))
