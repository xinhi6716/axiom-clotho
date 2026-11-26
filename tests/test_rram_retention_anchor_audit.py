import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "audit_rram_retention_anchors.py"
IDS = ("lot", "wafer", "die", "cell", "episode")
TIMES = ("nominal_us", "clock", "epoch", "tick_us", "error_us")
BASE = ("privateLot", "W1", "D1", "C1", "E1")


def episode(key=BASE, early=1_000_000, final=100_000_000_000, error=0):
    origin = 47
    return [
        [*key, nominal, "counter", "epoch", origin + delta, error]
        for nominal, delta in (
            (0, 0), (1_000_000, early), (100_000_000_000, final)
        )
    ]


class RetentionAnchorAudit(unittest.TestCase):
    def run_capture(self, roster, rows, capture_header=IDS + TIMES):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, header, payload in (
                    ("roster.csv", IDS, roster), ("capture.csv", capture_header, rows)):
                with (folder / name).open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((header, *payload))
            return subprocess.run(
                [sys.executable, str(TOOL), "--frozen-roster", str(folder / "roster.csv"),
                 "--capture", str(folder / "capture.csv")],
                text=True, capture_output=True, check=False,
            )

    def test_complete_interleaved_episodes_use_physical_die_keys(self):
        other = ("privateLot", "W2", "D1", "C1", "E1")
        rows = [row for pair in zip(episode(), episode(other)) for row in pair]
        r = self.run_capture([BASE, other], rows)
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(report["frozen_attempted_dies"], 2)
        self.assertEqual(report["eligible_episodes"], 2)
        self.assertNotIn("privateLot", r.stdout)

    def test_cannot_borrow_late_or_other_episode_measurement(self):
        rows = episode()[:2]
        rows.append([*BASE, 95_000_000_000, "counter", "epoch", 100_001_000_047, 0])
        other = (*BASE[:4], "E2")
        rows.append(episode(other)[-1])
        absent = (*BASE[:4], "E3")
        r = self.run_capture([BASE, absent], rows)
        self.assertEqual(r.returncode, 1, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(report["unexpected_capture_episodes"], 1)
        self.assertEqual(report["nonscheduled_rows_in_frozen_episodes"], 1)
        self.assertEqual(report["frozen_episodes_by_primary_reason"], {
            "missing_capture": 1, "missing_nominal_anchor": 1,
        })

    def test_clock_errors_duplicates_and_numeric_boundaries_remain_distinct(self):
        cases = [
            ("reset", episode(), "different_clock_or_epoch"),
            ("missing_clock", episode(), "unknown_clock_or_time"),
            ("duplicate", episode(), "duplicate_nominal_anchor"),
            ("early_zero", episode(early=0), "early_outside_window"),
            ("early_uncertain", episode(early=1_020_000, error=1),
             "early_boundary_uncertain"),
            ("late_uncertain", episode(final=80_000_000_000, error=1),
             "terminal_boundary_uncertain"),
            ("late_outside", episode(final=79_999_999_999), "terminal_outside_window"),
            ("closed_lower", episode(final=80_000_000_000), "eligible"),
            ("closed_upper", episode(early=1_020_000, final=120_000_000_000), "eligible"),
        ]
        cases[0][1][2][7] = "epoch-reset"
        cases[1][1][0][6] = ""
        cases[2][1].append(cases[2][1][-1])
        for name, rows, reason in cases:
            with self.subTest(name=name):
                r = self.run_capture([BASE], rows)
                self.assertEqual(r.returncode, int(reason != "eligible"), r.stderr)
                self.assertEqual(json.loads(r.stdout)["frozen_episodes_by_primary_reason"],
                                 {reason: 1})

    def test_capture_cannot_supply_labels_or_change_frozen_identity_set(self):
        r = self.run_capture([BASE], [(*row, "sensitive-value") for row in episode()],
                             IDS + TIMES + ("conductance_S",))
        self.assertEqual(r.returncode, 2)
        self.assertFalse(r.stdout)
        self.assertNotIn("sensitive-value", r.stderr)
        r = self.run_capture([BASE, BASE], episode())
        self.assertEqual(r.returncode, 2)
        self.assertIn("duplicate", r.stderr)


if __name__ == "__main__":
    unittest.main()
