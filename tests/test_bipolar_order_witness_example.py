import json
from pathlib import Path
import subprocess
import sys
import unittest


class BipolarOrderWitnessTests(unittest.TestCase):
    def test_exact_cli(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "examples/bipolar-order-witness/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        out = json.loads(proc.stdout)
        totals = {"net": 0, "positive": 2, "negative": 2}
        self.assertEqual(out["early_plus"], {"exposure": totals, "state": 0})
        self.assertEqual(out["late_plus"], {"exposure": totals, "state": 2})
        self.assertEqual(out["unknown_polarity"], {
            "exposure": None, "conditional_possible_states": [0, 1],
        })
