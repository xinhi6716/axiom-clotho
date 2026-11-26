from pathlib import Path
from runpy import run_path
from unittest import TestCase

demo = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                    / "observation-policy-shortcut" / "run.py"))["demonstrate"]


class ObservationPolicyShortcutTests(TestCase):
    def test_frozen_head_advantage_reverses(self):
        self.assertEqual(demo(), {
            "untouched_same_policy": {"schedule_head_brier": "0", "pooled_brier": "1/4"},
            "untouched_independent_policy": {"schedule_head_brier": "1/2", "pooled_brier": "1/4"},
        })
