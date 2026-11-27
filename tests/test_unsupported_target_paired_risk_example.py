import json
from pathlib import Path
import subprocess
import sys
import unittest


class UnsupportedTargetPairedRiskTests(unittest.TestCase):
    def test_exact_cli(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "examples/unsupported-target-paired-risk/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(proc.stdout), {
            "target_size": 5,
            "labeled_overlap": 3,
            "overlap_brier_gap_a_minus_b": "-1/4",
            "generic_bounds": ["-11/20", "1/4"],
            "paired_bounds": ["-7/20", "1/20"],
            "paired_completions": ["-7/20", "-3/20", "-3/20", "1/20"],
        })
