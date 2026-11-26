import csv
import hashlib
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
    def run_capture(self, roster, rows, capture_header=IDS + TIMES, *, extra=(), block_map=None):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, header, payload in (
                    ("roster.csv", IDS, roster), ("capture.csv", capture_header, rows)):
                with (folder / name).open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((header, *payload))
            if block_map is not None:
                block_map_path = folder / "blocks.csv"
                with block_map_path.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((IDS + ("acquisition_block",), *block_map))
                extra = (*extra, "--acquisition-block-map", str(block_map_path))
            return subprocess.run(
                [sys.executable, str(TOOL), "--frozen-roster", str(folder / "roster.csv"),
                 "--capture", str(folder / "capture.csv"), *extra],
                text=True, capture_output=True, check=False,
            )

    def test_archived_roster_digest_binds_exact_bytes_before_capture_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            roster = folder / "roster.csv"
            capture = folder / "capture.csv"
            original = (",".join(IDS) + "\r\n" + ",".join(BASE) + "\r\n").encode()
            digest = hashlib.sha256(original).hexdigest()
            roster.write_bytes(original)
            with capture.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows((IDS + TIMES, *episode()))

            def run(*extra):
                return subprocess.run(
                    [sys.executable, str(TOOL), "--frozen-roster", str(roster),
                     "--capture", str(capture), *extra],
                    text=True, capture_output=True, check=False,
                )

            before = json.loads(run().stdout)
            for byte_change in (
                original.replace(b"\r\n", b"\n"),
                b"\xef\xbb\xbf" + original,
                original.replace(b"E1", b"E2"),
                original + b" ",
            ):
                roster.write_bytes(byte_change)
                r = run("--frozen-roster-sha256", digest)
                self.assertEqual(r.returncode, 2)
                self.assertIn("SHA-256 mismatch", r.stderr)
                self.assertFalse(r.stdout)
                self.assertNotIn(digest, r.stderr)
                self.assertNotIn("privateLot", r.stderr)
            roster.write_bytes(original)
            r = run("--frozen-roster-sha256", digest)
            self.assertEqual(r.returncode, 0, r.stderr)
            after = json.loads(r.stdout)
            self.assertIs(after.pop("frozen_roster_byte_digest_verified"), True)
            self.assertEqual(before, after)
            self.assertNotIn(digest, r.stdout)

            capture.unlink()
            r = run("--frozen-roster-sha256", "0" * 64)
            self.assertEqual(r.returncode, 2)
            self.assertIn("SHA-256 mismatch", r.stderr)
            self.assertFalse(r.stdout)
            r = run("--frozen-roster-sha256", digest.upper())
            self.assertEqual(r.returncode, 2)
            self.assertIn("64 lowercase", r.stderr)
            self.assertFalse(r.stdout)

    def test_complete_interleaved_episodes_use_physical_die_keys(self):
        other = ("privateLot", "W2", "D1", "C1", "E1")
        rows = [row for pair in zip(episode(), episode(other)) for row in pair]
        r = self.run_capture([BASE, other], rows)
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(report["frozen_attempted_dies"], 2)
        self.assertEqual(report["eligible_episodes"], 2)
        self.assertNotIn("privateLot", r.stdout)

    def test_diagnostic_counts_die_support_on_frozen_not_captured_denominator(self):
        # Reused die names on different wafers are distinct; different cells
        # and episodes on that same physical die are not additional dies.
        same_die = (*BASE[:3], "C2", "E2")
        another = (BASE[0], "W2", *BASE[2:])
        partial_other = (*another[:4], "E2")
        entirely_absent = (BASE[0], "W3", *BASE[2:])
        extra_die = (BASE[0], "W4", *BASE[2:])
        roster = [BASE, same_die, another, partial_other, entirely_absent]
        capture = episode(BASE) + episode(same_die) + episode(another)
        capture += episode(partial_other, early=0) + episode(extra_die)
        ordinary = self.run_capture(roster, capture)
        diag = self.run_capture(roster, capture, extra=("--diagnose-dies",))
        self.assertEqual((ordinary.returncode, diag.returncode), (1, 1))
        report = json.loads(diag.stdout)
        self.assertEqual(report.pop("declared_die_timing_support"), {
            "dies_with_all_frozen_episodes_eligible": 1,
            "dies_with_some_but_not_all_frozen_episodes_eligible": 1,
            "dies_with_no_frozen_episodes_eligible": 1,
            "max_eligible_episodes_on_one_die": 2,
        })
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertEqual(report["frozen_attempted_dies"], 3)
        self.assertEqual(report["eligible_episodes"], 3)
        self.assertEqual(report["unexpected_capture_episodes"], 1)
        self.assertNotIn("privateLot", diag.stdout)

    def test_explicit_blocks_distinguish_repeated_die_from_shared_session(self):
        repeat = (*BASE[:4], "E2")
        another = (BASE[0], "W2", *BASE[2:])
        absent = (*BASE[:4], "E3")
        roster = [BASE, repeat, another, absent]
        capture = episode(BASE) + episode(repeat) + episode(another)
        block_map = [
            (*another, "privateBlockA"), (*BASE, "privateBlockA"),
            (*absent, "privateBlockC"), (*repeat, "privateBlockB"),
        ]
        ordinary = self.run_capture(roster, capture)
        diag = self.run_capture(roster, capture, block_map=block_map)
        self.assertEqual((ordinary.returncode, diag.returncode), (1, 1))
        report = json.loads(diag.stdout)
        self.assertEqual(report.pop("declared_acquisition_block_timing_support"), {
            "frozen_declared_blocks": 3, "blocks_with_eligible_episodes": 2,
            "blocks_with_eligible_episodes_from_multiple_dies": 1,
            "dies_with_eligible_episodes_in_multiple_blocks": 1,
        })
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertNotIn("privateBlock", diag.stdout)
        self.assertNotIn("privateLot", diag.stdout)

    def test_block_map_cannot_omit_duplicate_or_add_an_episode(self):
        for mapping, diagnostic in [
            ([], "does not cover"),
            ([(*BASE, "block1"), (*BASE, "block1")], "duplicate episode"),
            ([(*BASE, "block1"), (*BASE[:4], "E2", "block2")], "non-roster episode"),
            ([(*BASE, "")], "invalid acquisition block label"),
        ]:
            with self.subTest(diagnostic=diagnostic):
                r = self.run_capture([BASE], episode(), block_map=mapping)
                self.assertEqual(r.returncode, 2)
                self.assertFalse(r.stdout)
                self.assertIn(diagnostic, r.stderr)
                self.assertNotIn("privateLot", r.stderr)

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

    def test_anchor_diagnosis_exposes_secondary_faults_on_the_same_denominator(self):
        off_early = episode(early=0, final=80_000_000_000, error=1)
        off_early[2][7] = "other-epoch"
        other = (*BASE[:4], "E2")
        duplicate_origin = episode(other)
        duplicate_origin.append(duplicate_origin[0])
        duplicate_origin[2][8] = ""
        absent = (*BASE[:4], "E3")
        ok = (*BASE[:4], "E4")
        roster = [BASE, other, absent, ok]
        capture = off_early + duplicate_origin + episode(ok)
        before = self.run_capture(roster, capture)
        richer = self.run_capture(roster, capture, extra=("--diagnose-anchors",))
        self.assertEqual((before.returncode, richer.returncode), (1, 1))
        result = json.loads(richer.stdout)
        counts = result.pop("frozen_nominal_anchor_statuses_by_us")
        self.assertEqual(result, json.loads(before.stdout))
        self.assertEqual(counts, {
            "0": {
                "duplicate_nominal_anchor": 1, "missing_nominal_anchor": 1,
                "usable_timestamp": 2,
            },
            "1000000": {
                "boundary_uncertain": 1, "missing_nominal_anchor": 1,
                "unusable_origin": 1, "within_window": 1,
            },
            "100000000000": {
                "different_clock_or_epoch": 1, "missing_nominal_anchor": 1,
                "unknown_clock_or_time": 1, "within_window": 1,
            },
        })
        self.assertTrue(all(sum(statuses.values()) == len(roster) for statuses in counts.values()))
        self.assertNotIn("other-epoch", richer.stdout)
        self.assertNotIn("privateLot", richer.stdout)

    def test_single_fault_does_not_disable_unrelated_anchor_diagnosis(self):
        rows = episode(early=0)
        richer = self.run_capture([BASE], rows, extra=("--diagnose-anchors",))
        self.assertEqual(richer.returncode, 1, richer.stderr)
        data = json.loads(richer.stdout)
        self.assertEqual(data["frozen_episodes_by_primary_reason"], {"early_outside_window": 1})
        self.assertEqual(data["frozen_nominal_anchor_statuses_by_us"]["100000000000"],
                         {"within_window": 1})

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
