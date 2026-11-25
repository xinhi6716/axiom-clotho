"""Check exact clock accounting in the invented operation ledger."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


RUN = Path(__file__).resolve().parents[1] / "examples/operation-time-ledger/run.py"


class OperationTimeLedgerTests(unittest.TestCase):
    def test_overlap_and_anchor_are_separate(self):
        receipt = json.loads(subprocess.check_output([sys.executable, str(RUN)], text=True))
        self.assertEqual(receipt["initiated_attempts"], 3)
        a, b, c = (receipt["attempts"][k] for k in "ABC")
        self.assertEqual([a["waveform_work_ticks"], a["busy_ticks_on_declared_clock"]], [40, 35])
        self.assertEqual([a["start_to_terminal_ticks"], b["start_to_terminal_ticks"]], [60, 60])
        self.assertEqual([a["declaration_age_at_reference"], b["declaration_age_at_reference"]], [1000, 1000])
        self.assertEqual([a["write_age_at_reference"], b["write_age_at_reference"]], [1020, 1050])
        self.assertIsNone(c["start_to_terminal_ticks"])
        self.assertIsNone(c["write_age_at_reference"])

    def test_interval_union_is_order_invariant(self):
        union = runpy.run_path(str(RUN))["busy_time"]
        intervals = [(3, 8), (0, 7), (1, 2), (8, 10), (12, 13)]
        self.assertEqual(union(intervals), 11)
        self.assertEqual(union(intervals[::-1]), 11)
        self.assertEqual(union([]), 0)

    def test_impossible_chronology_not_coerced(self):
        describe = runpy.run_path(str(RUN))["describe"]
        for row in ((0, 60, 60, 50, []),
                    (0, 60, 10, 100, [("write", 10, 12)]),
                    (0, None, None, 30, [("write", 31, 32)])):
            with self.subTest(row=row), self.assertRaises(ValueError):
                describe(row)
