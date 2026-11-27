import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "preflight_rram_two_read_capture.py"
FIELDS = ("addr", "i", "timept", "time", "after_write", "after_relax", "extra")


def group(address=0, elapsed="1.0000", reverse=True):
    return [[address, i, 1, elapsed, 2 + i, 17 - i if reverse else 2 + i, "unused"]
            for i in range(16)]


def capture(path, rows, fields=FIELDS):
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows((fields, *rows))


def cli(*args):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True, check=False)


class TwoReadCapturePreflight(unittest.TestCase):
    def test_complete_interleaved_groups_keep_raw_pairs_and_capture_namespaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = (Path(tmp) / part for part in ("a.csv", "b.csv"))
            rows = group(0) + group(1, "0.999999999999999", False)
            rows[0][3] = "1"
            random.Random(27).shuffle(rows)
            capture(a, rows)
            capture(b, group(0, "1.01", False))
            result = cli("--capture", "c1", "chip1", str(a), "--capture", "c1", "chip2", str(b))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["captures"], [
                {"capture_id": "c1", "chip_id": "chip1", "complete_address_groups": 2,
                 "native_paired_rows": 32, "same_row_persistence_abs_error_sum_adc_codes": 128},
                {"capture_id": "c1", "chip_id": "chip2", "complete_address_groups": 1,
                 "native_paired_rows": 16, "same_row_persistence_abs_error_sum_adc_codes": 0},
            ])
            for key in ("verified_nominal_zero_timestamp", "native_100ks_endpoint", "retention_gate_eligible"):
                self.assertIs(report[key], False)

    def test_missing_duplicate_conflicting_or_invalid_lane_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.csv"
            for column, value, diagnostic in (
                (1, "01", "duplicate native address/lane"), (1, 16, "lanes 0"),
                (3, "1.00000000000000000000001", "do not share one elapsed time"),
                (3, "0.98", "strictly between"), (3, "1.02", "strictly between"),
                (3, "NaN", "strictly between"), (2, "0", "nominal timepoint 1"),
                (4, "65", "ADC codes"), (5, "1", "ADC codes"),
            ):
                with self.subTest(column=column, value=value):
                    rows = group()
                    rows[0][column] = value
                    capture(path, rows)
                    result = cli("--capture", "c", "chip", str(path))
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse(result.stdout)
                    self.assertIn(diagnostic, result.stderr)
            for rows in ([], group()[:-1]):
                capture(path, rows)
                result = cli("--capture", "c", "chip", str(path))
                self.assertEqual(result.returncode, 2)
                self.assertIn("complete 16-lane", result.stderr)
            capture(path, group(), FIELDS[:-1] + ("addr",))
            self.assertIn("distinct header", cli("--capture", "c", "chip", str(path)).stderr)

    def test_cannot_reuse_one_capture_path_under_another_chip_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, alias = (Path(tmp) / part for part in ("raw.csv", "alias.csv"))
            capture(raw, group())
            alias.hardlink_to(raw)
            first = ("--capture", "one", "chipA", str(raw))
            for binding, diagnostic in (
                (("one", "chipA", str(alias)), "duplicate capture/chip"),
                (("two", "chipB", str(alias)), "same input file"),
            ):
                result = cli(*first, "--capture", *binding)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(result.stdout)
                self.assertIn(diagnostic, result.stderr)


    def test_distinct_inodes_with_exact_copied_payload_do_not_support_two_chips(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, copy = (Path(tmp) / part for part in ("raw.csv", "copy.csv"))
            capture(raw, group())
            copy.write_bytes(raw.read_bytes())
            self.assertNotEqual(raw.stat().st_ino, copy.stat().st_ino)
            args = ("--capture", "one", "chipA", str(raw), "--capture", "two", "chipB", str(copy))
            rejected = cli(*args)
            self.assertEqual(rejected.returncode, 2)
            self.assertFalse(rejected.stdout)
            self.assertIn("byte-identical capture payloads", rejected.stderr)
            self.assertNotIn(str(raw), rejected.stderr)

            # Identical coarse time and error summaries do not mean identical payload.
            rows = group()
            for row in rows:
                row[4] += 1
                row[5] += 1
            capture(copy, rows)
            accepted = cli(*args)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            reports = json.loads(accepted.stdout)["captures"]
            self.assertEqual([r["same_row_persistence_abs_error_sum_adc_codes"] for r in reports],
                             [128, 128])
