import json
from pathlib import Path
import subprocess
import sys
import unittest


class MidsectionAndPeakEqualityTests(unittest.TestCase):
    def test_exact_cli(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "examples/midsection-and-peak-equality/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        out = json.loads(proc.stdout)
        self.assertEqual(out["shared_tied_peak"], {
            "missing": [], "midpoint_max": 2, "mean_section_max": "10/3",
            "max_mean_profile": "10/3", "common_argmax": [1],
        })
        self.assertEqual(out["disjoint_peaks"], {
            "missing": [], "midpoint_max": 2, "mean_section_max": "10/3",
            "max_mean_profile": "2", "common_argmax": [],
        })
        self.assertEqual(out["missing_right_section"], {
            "missing": ["right"], "midpoint_max": 2, "mean_section_max": None,
            "max_mean_profile": None, "common_argmax": None,
        })
