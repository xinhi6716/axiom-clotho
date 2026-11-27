import json
from pathlib import Path
import subprocess
import sys
import unittest


class EarlyAnchorRateEnvelopeTests(unittest.TestCase):
    def test_exact_cli(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "examples/early-anchor-rate-envelope/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(proc.stdout), {
            "native_early_read_available": False,
            "anchor": 1,
            "linear_back_extrapolation": "59/5",
            "monotone_plus_broad_rate": ["10", "19"],
            "independent_tight_rate": ["59/5", "49/4"],
            "inconsistent_rate_rejected": True,
        })
