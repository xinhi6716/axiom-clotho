import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReadQualityTest(unittest.TestCase):
    def run_audit(self, plan, measured):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, reads_path = (Path(directory) / name for name in ("plan.csv", "reads.csv"))
            plan_path.write_text("pair,expected_range\n" + plan)
            reads_path.write_text("pair,role,status,range\n" + measured)
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[1] /
                 "tools/preflight_rram_read_quality.py"), "--plan", str(plan_path),
                 "--quality", str(reads_path)], text=True, capture_output=True)

    def test_fixed_denominator_and_nonexclusive_flags(self):
        got = self.run_audit(
            "A,R1\nB,R1\nC,R1\nD,R1\nE,R1\n",
            "A,late,settled,R1\nA,early,settled,R1\n"
            "B,early,compliance,R1\nB,late,overrange,R2\n"
            "C,early,unknown,unknown\n"
            "D,early,settling,R2\nD,late,settling,R2\n",
        )
        self.assertEqual(got.returncode, 1, got.stderr)
        self.assertEqual(json.loads(got.stdout), {
            "planned_pairs": 5, "pairs_matching_status_and_range": 1,
            "pairs_by_reason_nonexclusive": {
                "missing_read": 2, "unknown_status": 1, "compliance": 1, "overrange": 1,
                "settling": 1, "unknown_range": 1, "unexpected_range": 2, "range_changed": 1,
            },
        })

    def test_clean_pair_and_invalid_substitutions(self):
        plan = "A,R1\n"
        clean = "A,early,settled,R1\nA,late,settled,R1\n"
        self.assertEqual(self.run_audit(plan, clean).returncode, 0)
        for log in (clean + "A,late,settled,R1\n", clean.replace("A,late", "B,late"),
                    clean.replace("settled", "range-is-fine"),
                    clean.replace("A,late,settled,R1", "A,late,settled,R1,0.002")):
            with self.subTest(log=log):
                got = self.run_audit(plan, log)
                self.assertEqual(got.returncode, 2)
                self.assertEqual(got.stdout, "")
