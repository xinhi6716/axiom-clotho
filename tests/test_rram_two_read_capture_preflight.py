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

    def test_frozen_address_range_rejects_gaps_omissions_and_wrong_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.csv"
            bind = ("--capture", "c", "chip", str(path))
            exact = ("--expected-address-range", "2", "4")
            for addresses in ((2, 4), (2, 3), (2, 3, 5), (2, 3, 4, 5)):
                with self.subTest(addresses=addresses):
                    capture(path, [row for a in addresses for row in group(a)])
                    self.assertEqual(cli(*bind).returncode, 0)
                    rejected = cli(*bind, *exact)
                    self.assertEqual(rejected.returncode, 2)
                    self.assertFalse(rejected.stdout)
                    self.assertIn("frozen address range", rejected.stderr)
            capture(path, group(4) + group(2) + group(3))
            accepted = cli(*bind, *exact)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            report = json.loads(accepted.stdout)
            self.assertEqual(report["matched_declared_address_range"], [2, 4])
            self.assertIs(report["physical_capture_independence_verified"], False)
            for bad in (("4", "2"), ("0", "1000000000"), ("+2", "4")):
                rejected = cli(*bind, "--expected-address-range", *bad)
                self.assertEqual(rejected.returncode, 2)
                self.assertIn("expected address range", rejected.stderr)
            huge = cli(*bind, "--expected-address-range", "0", "999999999")
            self.assertEqual(huge.returncode, 2)
            self.assertIn("frozen address range", huge.stderr)

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


    def test_reformatted_observable_copy_and_exact_elapsed_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, changed = (Path(tmp) / part for part in ("a.csv", "b.csv"))
            elapsed = "1.00000000000000000000000000001"
            rows = group(1, elapsed)
            capture(original, rows)
            # Reorder both axes, use a different newline/quoting, pad integer
            # fields and elapsed values, and relabel non-observable metadata.
            def rewrite(candidate):
                with changed.open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\n")
                    writer.writerow(FIELDS[::-1])
                    for addr, lane, t, time, initial, final, _ in reversed(candidate):
                        writer.writerow(["another metadata", f"0{final}", f"0{initial}",
                                         f"0{time}0", f"0{t}", f"0{lane}", f"0{addr}"])

            args = ("--capture", "captureA", "chipA", str(original),
                    "--capture", "captureB", "chipB", str(changed))
            rewrite(rows)
            self.assertNotEqual(original.read_bytes(), changed.read_bytes())
            rejected = cli(*args)
            self.assertEqual(rejected.returncode, 2)
            self.assertFalse(rejected.stdout)
            self.assertIn("identical required capture observables", rejected.stderr)
            self.assertNotIn(elapsed, rejected.stderr)

            rewrite(group(1, "1.00000000000000000000000000002"))
            accepted = cli(*args)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIs(json.loads(accepted.stdout)["physical_capture_independence_verified"], False)
            # Even with equal error summary and elapsed time, swapping the two
            # ADC vectors at each lane changes the ordered observable.
            swapped = [[a, lane, t, time, final, start, extra]
                       for a, lane, t, time, start, final, extra in rows]
            rewrite(swapped)
            self.assertEqual(cli(*args).returncode, 0)
