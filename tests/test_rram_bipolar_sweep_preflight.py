import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "preflight_rram_bipolar_sweeps.py"


def check(rows, header=("trace_id", "time_s", "voltage_V"), *, local=False):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "geometry.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file).writerows((header, *rows))
        return subprocess.run([sys.executable, str(TOOL), str(path),
                               *(["--confirmed-four-local-segment-clocks"] if local else [])],
                              text=True, capture_output=True, check=False)


class BipolarSweepPreflight(unittest.TestCase):
    def test_distinguishes_measured_middle_zero_plateau_and_bracket(self):
        experiments = (
            (0, 1, 0, -1, 0),
            (0, 1, 1, 0, 0, -1, -1, 0),
            (0, 1, "0.000000000000000000000000000001", "-0.000000000000000000000000000001", -1, 0),
            (0, 1, "0.2", "0.3", 0, -1, 0),
            (0, 1, 0, 0),
        )
        rows = sorted(
            ((f"private{i}", time, bias) for i, v in enumerate(experiments)
             for time, bias in enumerate(v)), key=lambda row: row[1])
        rows += [("clock", "0", 0), ("clock", "1", 1), ("clock", "1.0", -1), ("clock", "2", 0)]
        rows += [("lo", 0, "0.000000000000000000000000000001"), ("lo", 1, 1), ("lo", 2, -1), ("lo", 3, 0)]
        result = check(rows)
        self.assertEqual(result.returncode, 0, result.stderr)
        r = json.loads(result.stdout)
        self.assertEqual(r["traces_by_geometry"], {
            "bipolar_with_unique_measured_middle_zero": 1,
            "bipolar_with_middle_zero_plateau": 1,
            "bipolar_only_brackets_middle_zero": 1,
            "unresolved_or_extra_voltage_reversal": 1,
            "both_voltage_polarities_not_observed": 1,
            "timestamp_order_not_strict": 1,
            "start_or_end_not_exactly_zero": 1,
        })
        self.assertEqual(r["recognized_bipolar_traces_with_repeated_apex"], 1)
        self.assertEqual(r["recognized_bipolar_traces_with_unique_apices"], 2)
        self.assertNotIn("private", result.stdout)

    def test_refuses_current_column_and_nonfinite_or_signed_negative_time(self):
        for rows, header, diagnostic in (
            ([("a", 0, 0, "outcome")], ("trace_id", "time_s", "voltage_V", "current_A"), "omit all current"),
            ([("a", -1, 0)], ("trace_id", "time_s", "voltage_V"), "cannot be negative"),
            ([("a", 0, "NaN")], ("trace_id", "time_s", "voltage_V"), "exact fixed-point"),
        ):
            result = check(rows, header)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(result.stdout)
            self.assertIn(diagnostic, result.stderr)

    def test_confirmed_reset_clocks_preserve_separate_boundary_rows(self):
        readings = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 0), (1, -1), (0, -1), (1, 0)]
        rows = [(key, t, v) for t, v in readings for key in ("privateA", "privateB")]
        ordinary = json.loads(check(rows).stdout)
        self.assertEqual(ordinary["traces_by_geometry"], {"timestamp_order_not_strict": 2})
        self.assertNotIn("clock_interpretation", ordinary)
        result = check(rows, local=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["traces_by_geometry"], {"four_local_clock_segments_with_matched_voltage_edges": 2})
        self.assertEqual(report["recognized_bipolar_traces_with_repeated_apex"], 2)
        self.assertIs(report["physical_time_unit_verified"], False)
        self.assertIs(report["intersegment_elapsed_time_verified"], False)
        self.assertNotIn("privateA", result.stdout)

    def test_local_clock_mode_does_not_repair_invalid_resets_or_voltage_edges(self):
        base = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 0), (1, -1), (0, -1), (1, 0)]
        cases = [
            ([(0.1, 0)] + base[1:], "first_local_clock_not_zero"),
            (base[:2] + [(1, 1)] + base[3:], "tied_local_clock_values"),
            (base[:2] + [("0.000000000000000000000000000001", 1)] + base[3:], "clock_rewind_not_to_exact_zero"),
            (base[:6], "expected_exactly_four_local_clock_segments"),
            (base + [(0, 0), (1, 0)], "expected_exactly_four_local_clock_segments"),
            (base[:-1], "insufficient_rows_in_local_clock_segment"),
            (base[:2] + [(0, "1.000000000000000000000000000001")] + base[3:],
             "voltage_mismatch_at_local_clock_reset"),
            (base[:3] + [(1, "0.1"), (0, "0.1")] + base[5:], "missing_exact_zero_at_sweep_endpoint_or_middle"),
            (base[:1] + [("0.5", 0)] + base[1:], "local_segment_voltage_direction_not_strict"),
        ]
        for points, expected in cases:
            with self.subTest(expected=expected):
                r = check([("trace", t, v) for t, v in points], local=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(json.loads(r.stdout)["traces_by_geometry"], {expected: 1})
        b = check([("trace", 0, 0, "not opened")],
                  ("trace_id", "time_s", "voltage_V", "current_A"), local=True)
        self.assertEqual(b.returncode, 2)
        self.assertIn("omit all current", b.stderr)

    def test_independent_leg_roster_binds_equal_voltage_boundary_rows_to_original_ordinal(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            geometry, roster = (Path(tmp) / name for name in ("input.csv", "roster.csv"))
            points = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 0), (1, -1), (0, -1), (1, 0)]
            with geometry.open("w", newline="") as stream:
                csv.writer(stream).writerows([("trace_id", "time_s", "voltage_V"),
                                               *(("privateA", *point) for point in points),
                                               ("unresolved", 0, 0)])
            expected = hashlib.sha256(geometry.read_bytes()).hexdigest()
            declarations = [("privateA", i + 1, i // 2 + 1) for i in range(8)] + [("unresolved", 9, 1)]

            def run(rows, *, digest=expected):
                with roster.open("w", newline="") as stream:
                    csv.writer(stream).writerows([("trace_id", "capture_record", "protocol_leg"), *rows])
                result = subprocess.run([sys.executable, str(TOOL), str(geometry),
                                         "--confirmed-four-local-segment-clocks", "--segment-roster", str(roster),
                                         "--expected-geometry-sha256", digest], text=True, capture_output=True)
                return result

            # Giving both adjacent equal-AV boundary rows one leg must not
            # pass just because their AV (and potentially another label) match.
            declarations[2] = ("privateA", 3, 1)
            valid = run(list(reversed(declarations)))
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(json.loads(valid.stdout)["declared_segment_roster_support_by_record"], {
                "declared_leg_matches_local_geometry": 7,
                "declared_leg_conflicts_with_local_geometry": 1,
                "local_geometry_unresolved": 1,
            })
            self.assertNotIn("privateA", valid.stdout)
            for malformed, message in (
                (declarations[:-1], "must cover every"),
                (declarations + declarations[:1], "duplicate geometry record"),
                ([("unresolved", 1, 1)] + declarations[1:], "does not identify the exact"),
                ([("privateA", 1, 5)] + declarations[1:], "leg 1"),
            ):
                r = run(malformed)
                self.assertEqual(r.returncode, 2)
                self.assertFalse(r.stdout)
                self.assertIn(message, r.stderr)
            geometry.write_bytes(geometry.read_bytes().replace(b"\r\n", b"\n"))
            mismatch = run(declarations)
            self.assertEqual(mismatch.returncode, 2)
            self.assertIn("geometry input SHA-256 mismatch", mismatch.stderr)
