import json
from pathlib import Path
import subprocess
import sys
import unittest


class AmbiguousClockUnitsTests(unittest.TestCase):
    def test_exact_cli(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "examples/ambiguous-clock-units/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        out = json.loads(proc.stdout)
        self.assertEqual(out["hypothetical_certified_ms"], {
            "available": 2, "roster_size": 2, "brier_a": "0", "brier_b": "1/16",
            "brier_a_minus_b": "-1/16",
        })
        unusable = {
            "available": 0, "roster_size": 2, "brier_a": "1/4", "brier_b": "1/16",
            "brier_a_minus_b": "3/16",
        }
        self.assertEqual(out["hypothetical_certified_s"], unusable)
        self.assertEqual(out["conflicting_unresolved_metadata"], unusable)
