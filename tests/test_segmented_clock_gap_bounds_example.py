import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples/segmented-clock-gap-bounds/run.py"


class SegmentedClockGapBoundsTests(unittest.TestCase):
    def test_cli(self):
        out = json.loads(subprocess.check_output([sys.executable, str(RUNNER)], cwd=ROOT))
        self.assertEqual(out["unknown_gap"]["elapsed_bounds"], [11, None])
        self.assertEqual(out["bounded_but_crosses"]["elapsed_gate"], "unresolved")
        self.assertEqual(out["bounded_elapsed_no_nominal"]["elapsed_gate"], "inside")
        self.assertFalse(out["bounded_elapsed_no_nominal"]["both_gates"])
        self.assertTrue(out["both_verified"]["both_gates"])
        self.assertEqual(out["outside_proven_even_without_upper"]["elapsed_gate"], "outside")
        self.assertEqual(out["unverified_program_origin"]["elapsed_bounds"], [None, None])

    def test_multiple_bridges(self):
        evaluate = runpy.run_path(str(RUNNER))["evaluate"]
        self.assertEqual(evaluate((1, 2), [(8, (1, 4)), (10, (5, 7))],
                                  3, (28, 34), True)["elapsed_bounds"], [28, 34])
        with self.assertRaisesRegex(ValueError, "gap"):
            evaluate((0, 0), [(8, (-1, 3))], 3, (13, 15), True)
