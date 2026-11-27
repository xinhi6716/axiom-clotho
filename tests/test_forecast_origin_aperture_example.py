import json
from pathlib import Path
import subprocess
import sys
import unittest


class ForecastOriginApertureTests(unittest.TestCase):
    def test_exact_cli(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "examples/forecast-origin-aperture/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(proc.stdout), {
            "center_passes_timestamp_filter": True,
            "crossing_aperture_usable": False,
            "closed_past_aperture_usable": True,
            "same_past_aperture_late_ready_usable": False,
            "held_outcomes": [0, 1],
            "held_mean_brier": {
                "invalid_centered": "0",
                "guarded_past": "1/2",
                "fixed_neutral": "1/4",
            },
        })
