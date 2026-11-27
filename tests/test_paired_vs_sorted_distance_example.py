import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples/paired-vs-sorted-distance/run.py"


class PairedVsSortedDistanceTests(unittest.TestCase):
    def test_documented_cli(self):
        proc = subprocess.run(
            [sys.executable, str(RUNNER)], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
        self.assertEqual(json.loads(proc.stdout), {
            "local_error": {"paired_mae": "1/3", "sorted_distance": "1/3",
                            "exhaustive_minimum": "1/3", "optimal_index_permutations": 2},
            "swapped_units": {"paired_mae": "4/3", "sorted_distance": "0",
                              "exhaustive_minimum": "0", "optimal_index_permutations": 2},
        })

    def test_identity_is_not_insertion_order(self):
        compare = runpy.run_path(str(RUNNER))["comparison"]
        self.assertEqual(compare({"y": 0, "z": 2}, {"z": 2, "y": 0})["paired_mae"], "0")
        with self.assertRaisesRegex(ValueError, "roster"):
            compare({"y": 0}, {"not_y": 0})
