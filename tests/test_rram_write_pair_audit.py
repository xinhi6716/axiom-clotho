import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "audit_rram_write_pairs.py"
HEADER = ("chip", "cell", "episode", "transition", "event_index", "kind",
          "voltage_V", "width_s", "conductance_S", "protocol")


def triplet(chip="chipA", key="t1", start=0):
    identity = [chip, "privateCell", "ep1", key]
    return [
        identity + [start, "pre_read", "0.2", "", "0.00001", "example"],
        identity + [start + 1, "write", "-1.2", "2e-7", "", "example"],
        identity + [start + 2, "post_read", "2e-1", "", "0.000005", "example"],
    ]


class WritePairAudit(unittest.TestCase):
    def run_rows(self, rows, header=HEADER, extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.csv"
            with p.open("w", newline="", encoding="utf-8") as out:
                csv.writer(out).writerows([header, *rows])
            return subprocess.run([sys.executable, str(TOOL), str(p), *extra],
                                  text=True, capture_output=True, check=False)

    def test_interleaved_devices_count_as_declared(self):
        rows = [r for pair in zip(triplet(), triplet("chipB")) for r in pair]
        result = self.run_rows(rows)
        self.assertEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertEqual(report["eligible_transitions"], 2)
        self.assertEqual(report["eligible_declared_cells"], 2)
        self.assertEqual(report["eligible_declared_chips"], 2)
        self.assertNotIn("privateCell", result.stdout)

    def test_eligible_chips_with_disjoint_protocols_have_no_shared_strata(self):
        a, b = triplet(), triplet("chipB")
        for row in b:
            row[9] = "different_protocol"
        report = json.loads(self.run_rows(a + b).stdout)
        self.assertEqual(report["eligible_declared_chips"], 2)
        self.assertEqual(report["declared_read_condition_coverage"], {
            "declared_strata": 2, "multi_chip_strata": 0,
            "eligible_transitions_in_multi_chip_strata": 0,
        })
        self.assertEqual(report["declared_exact_action_coverage"]["multi_chip_strata"], 0)

    def test_shared_protocol_does_not_imply_shared_action(self):
        a, b = triplet(), triplet("chipB")
        b[1][7] = "4e-7"
        report = json.loads(self.run_rows(a + b).stdout)
        self.assertEqual(report["declared_read_condition_coverage"]["multi_chip_strata"], 1)
        self.assertEqual(report["declared_exact_action_coverage"]["multi_chip_strata"], 0)

    def test_exact_declared_action_does_not_merge_distinct_decimal_settings(self):
        for field, distinct, equal in (
            (6, "-1.20000000000000001", "-1.20E0"),
            (7, "2.00000000000000001e-7", "0.000000200"),
        ):
            with self.subTest(field=field):
                a, b = triplet(), triplet("chipB")
                self.assertEqual(float(a[1][field]), float(distinct))
                b[1][field] = distinct
                result = self.run_rows(a + b)
                self.assertEqual(result.returncode, 0)
                report = json.loads(result.stdout)
                self.assertEqual(report["eligible_transitions"], 2)
                self.assertEqual(report["declared_read_condition_coverage"]["multi_chip_strata"], 1)
                self.assertEqual(report["declared_exact_action_coverage"]["multi_chip_strata"], 0)
                self.assertEqual(report["declared_exact_action_coverage"]["declared_strata"], 2)

                b[1][field] = equal
                result = self.run_rows(a + b)
                self.assertEqual(result.returncode, 0)
                exact = json.loads(result.stdout)["declared_exact_action_coverage"]
                self.assertEqual(exact["multi_chip_strata"], 1)
                self.assertEqual(exact["eligible_transitions_in_multi_chip_strata"], 2)

    def test_opt_in_shared_action_gate_keeps_structural_gate_and_report(self):
        gate = ("--require-shared-action",)
        a, b = triplet(), triplet("chipB")
        b[1][7] = "0.000000200"
        shared = self.run_rows(a + b, extra=gate)
        self.assertEqual(shared.returncode, 0, shared.stderr)
        self.assertEqual(json.loads(shared.stdout)["declared_exact_action_coverage"]["multi_chip_strata"], 1)

        invalid = self.run_rows(a + b + triplet("chipC")[1:], extra=gate)
        self.assertEqual(invalid.returncode, 1)
        self.assertEqual(json.loads(invalid.stdout)["eligible_transitions"], 2)

        b[1][7] = "4e-7"
        legacy = self.run_rows(a + b)
        blocked = self.run_rows(a + b, extra=gate)
        self.assertEqual(legacy.returncode, 0)
        self.assertEqual(blocked.returncode, 1)
        self.assertEqual(blocked.stdout, legacy.stdout)
        self.assertEqual(json.loads(blocked.stdout)["declared_exact_action_coverage"]["multi_chip_strata"], 0)

    def test_exact_action_coverage_counts_chips_not_repeats(self):
        a, b, c = triplet(), triplet("chipB"), triplet(key="t2", start=3)
        report = json.loads(self.run_rows(a + b + c).stdout)
        self.assertEqual(report["declared_exact_action_coverage"], {
            "declared_strata": 1, "multi_chip_strata": 1,
            "eligible_transitions_in_multi_chip_strata": 3,
        })
        report = json.loads(self.run_rows(a + c).stdout)
        self.assertEqual(report["declared_exact_action_coverage"]["multi_chip_strata"], 0)

    def test_missing_before_not_inferred(self):
        result = self.run_rows(triplet()[1:])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["rejected_transitions_by_primary_reason"],
                         {"expected_exactly_pre_write_post": 1})

    def test_read_voltage_mismatch_and_nonfinite_write_width(self):
        a, b = triplet(), triplet(key="t2", start=3)
        a[2][6] = "0.3"
        b[1][7] = "1e309"
        result = self.run_rows(a + b)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["rejected_transitions_by_primary_reason"],
                         {"invalid_event_fields": 1, "read_voltage_mismatch": 1})

    def test_protocol_mismatch_is_not_pooled(self):
        rows = triplet()
        rows[2][9] = "different_protocol"
        result = self.run_rows(rows)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["rejected_transitions_by_primary_reason"],
                         {"protocol_mismatch": 1})

    def test_gaps_quarantine_not_reindexed(self):
        a = triplet()
        a[2][4] = 3
        result = self.run_rows(a)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["rejected_transitions_by_primary_reason"],
                         {"noncontiguous_episode_events": 1})

    def test_old_transition_reappears(self):
        a, b = triplet(), triplet(key="t2", start=2)
        a[-1][4] = 5
        result = self.run_rows(a[:2] + b + a[-1:])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["eligible_transitions"], 1)

    def test_duplicate_event_index_is_fatal_and_redacted(self):
        a = triplet()
        a[1][4] = a[0][4]
        result = self.run_rows(a)
        self.assertEqual(result.returncode, 2)
        self.assertIn("indices must increase", result.stderr)
        self.assertNotIn("privateCell", result.stderr)

    def test_header_only_cannot_assert_readiness(self):
        result = self.run_rows([])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["eligible_transitions"], 0)

    def test_repeated_header_field_is_invalid(self):
        result = self.run_rows([], HEADER[:-1] + ("kind",))
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout)


if __name__ == "__main__":
    unittest.main()
