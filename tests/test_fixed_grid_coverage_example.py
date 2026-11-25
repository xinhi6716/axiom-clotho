"""Pin the fixed-grid example's target and exact coverage counts."""

from pathlib import Path
import runpy
import subprocess
import sys
import unittest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "fixed-grid-coverage" / "run.py"


class FixedGridCoverageExampleTests(unittest.TestCase):
    def test_examples_output_matches_doc(self):
        result = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(result.splitlines(), [
            "four fixed times: 19/20 19/20 19/20 19/20",
            "outcome-selected time with pointwise intervals: 16/20",
            "simultaneous fixed grid: 19/20",
        ])

    def test_reordered_devices_preserve_target_counts(self):
        functions = runpy.run_path(str(EXAMPLE))
        self.assertEqual(functions["audit"](functions["fixture"]()[::-1]), ((19,) * 4, 16, 19))

    def test_incomplete_or_different_grid_is_not_silently_scored(self):
        functions = runpy.run_path(str(EXAMPLE))
        for replacement in (None, 25):
            rows = functions["fixture"]()
            del rows[0][20]
            if replacement is not None:
                rows[0][replacement] = 0
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "complete devices"):
                functions["audit"](rows)
