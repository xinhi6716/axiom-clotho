from fractions import Fraction
from pathlib import Path
from runpy import run_path
from unittest import TestCase

fixture = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                       / "delayed-feedback-recency" / "run.py"))


class DelayedFeedbackRecencyTests(TestCase):
    def test_no_at_origin_or_future_feedback(self):
        f, h = fixture["forecast"], fixture["HISTORY"]
        self.assertEqual(f(h, 3, 0), Fraction(1, 2))
        self.assertEqual(f(h, 5, 1), 1)
        self.assertEqual(f(h + ((99, 100, 0),), 6, 0), 1)

    def test_exact_ordering_changes_score(self):
        self.assertEqual(fixture["demonstrate"](), {
            "last_arrival": {"forecasts": ["1", "0"], "brier": "1/2"},
            "last_event_ready": {"forecasts": ["1", "1"], "brier": "0"},
            "frozen_half": {"forecasts": ["1/2", "1/2"], "brier": "1/4"},
        })
