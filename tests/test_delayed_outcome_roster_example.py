"""Executable checks for the frozen-roster label-arrival example."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "delayed-outcome-roster" / "run.py"


class DelayedOutcomeRosterExampleTests(unittest.TestCase):
    def test_read_only_script_preserves_denominator_and_defers_interval_score(self):
        completed = subprocess.run([sys.executable, str(EXAMPLE)], env={
            **os.environ, "PYTHONPATH": str(ROOT / "src"),
        }, check=True, capture_output=True, text=True, timeout=20)
        early, at_cutoff, complete = json.loads(completed.stdout)
        self.assertEqual([item["scoring_cutoff"] for item in (early, at_cutoff, complete)], [2, 6, 7])
        self.assertEqual(early["pairs"], at_cutoff["pairs"])
        self.assertEqual(early["pairs"], {
            "n_rows": 6, "n_observed": 4, "n_missing": 2,
            "available_mae_b_minus_a": "1",
            "full_mae_b_minus_a_lower": "-1/3", "full_mae_b_minus_a_upper": "5/3",
        })
        self.assertEqual(at_cutoff["paired_intervals"]["status"], "incomplete_roster_not_scored")
        self.assertEqual(complete["pairs"]["n_rows"], 6)
        self.assertEqual(complete["pairs"]["n_missing"], 0)
        self.assertEqual(complete["pairs"]["available_mae_b_minus_a"], "-1/3")
        self.assertEqual(complete["paired_intervals"]["interval_score_b_minus_a_exact"], "0")
        self.assertFalse(completed.stderr)

    def test_late_label_mutation_cannot_affect_earlier_snapshots(self):
        namespace = runpy.run_path(str(EXAMPLE))
        snapshot, labels, forecasts = (namespace[name] for name in ("snapshot", "LABELS", "FORECASTS"))
        before = snapshot(6), snapshot(7), dict(forecasts)
        labels["l0"] = -100
        self.assertEqual(snapshot(6), before[0])
        self.assertNotEqual(snapshot(7), before[1])
        self.assertEqual(forecasts, before[2])
