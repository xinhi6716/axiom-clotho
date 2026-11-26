from pathlib import Path
from runpy import run_path
from unittest import TestCase

report = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                      / "retention-time-measure" / "run.py"))["report"]


class RetentionTimeMeasureTests(TestCase):
    def test_exact_winner_reversal(self):
        self.assertEqual(report(), {
            "log_uniform": {"A": "1", "B": "3/2"},
            "clock_uniform": {"A": "20/11", "B": "3/11"},
        })
