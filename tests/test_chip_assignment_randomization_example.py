import json
from pathlib import Path
import subprocess
import sys
import unittest


class ChipAssignmentRandomizationTests(unittest.TestCase):
    def test_exact_cli(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "examples/chip-assignment-randomization/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(proc.stdout), {
            "observed_absolute_mean_difference": "1",
            "chip_allocations": 20,
            "wrong_row_allocations": 924,
            "chip_p_value": "1/10",
            "wrong_row_p_value": "1/462",
            "alpha": "1/20",
            "chip_sharp_null_rejection": "0",
            "wrong_sharp_null_rejection_under_chip_design": "1/10",
        })
