import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "preflight_rram_bipolar_sweeps.py"


def check(rows, header=("trace_id", "time_s", "voltage_V")):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "geometry.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file).writerows((header, *rows))
        return subprocess.run([sys.executable, str(TOOL), str(path)],
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
