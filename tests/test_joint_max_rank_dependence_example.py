import json
from pathlib import Path
import subprocess
import sys
import unittest


class JointMaxRankDependenceTests(unittest.TestCase):
    def test_documented_cli(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "examples/joint-max-rank-dependence/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        out = json.loads(proc.stdout)
        aligned = {"component_ranks": ["1/10", "1/10"], "max_rank": "1/10",
                   "flag_at_alpha": True, "uniform_query_rejection_count_of_20": 2}
        self.assertEqual(out["aligned"], aligned)
        self.assertEqual(out["intact_row_order_control"], aligned)
        self.assertEqual(out["separated"], {
            "component_ranks": ["1/10", "1/10"], "max_rank": "3/20",
            "flag_at_alpha": False, "uniform_query_rejection_count_of_20": 0,
        })
