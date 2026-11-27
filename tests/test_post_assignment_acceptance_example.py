import json
from pathlib import Path
import subprocess
import sys
import unittest


class PostAssignmentAcceptanceTests(unittest.TestCase):
    def test_exact_cli(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "examples/post-assignment-acceptance/run.py"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(proc.stdout), {
            "balanced_assignments": 6,
            "sharp_null_effect": "0",
            "full_roster_expected_gap": "0",
            "acceptance_comparable_assignments": 5,
            "post_filter_empty_arm_assignments": 1,
            "acceptance_only_expected_gap_given_both_arms_nonempty": "-3/5",
            "acceptance_only_negative_gaps": 5,
        })
