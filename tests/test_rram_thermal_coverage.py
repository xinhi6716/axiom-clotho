import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ThermalCoverageTest(unittest.TestCase):
    def run_audit(self, plan, monitor):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path, monitor_path = (Path(tmp) / name for name in ("plan.csv", "monitor.csv"))
            plan_path.write_text("episode,start_s,end_s,min_c,max_c\n" + plan)
            monitor_path.write_text("episode,start_s,end_s,observed_min_c,observed_max_c\n" + monitor)
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[1] /
                 "tools/preflight_rram_thermal_coverage.py"), "--plan", str(plan_path),
                 "--monitor", str(monitor_path)], capture_output=True, text=True,
            )

    def test_planned_denominator_and_joint_defects(self):
        got = self.run_audit(
            "".join(f"{name},0,1,20,24\n" for name in "ABCDE"),
            "A,0.5,1,21,24\nA,0,0.5,20,23\nB,0,0.5,20,23\n"
            "B,0.500000000000000000000000000001,1,21,23\n"
            "C,0,1,19,23\nD,0,0.5,21,25\n",
        )
        self.assertEqual(got.returncode, 1, got.stderr)
        self.assertEqual(json.loads(got.stdout), {
            "planned_episodes": 5,
            "by_monitor_status": {"complete_in_band": 1, "unobserved_gap": 2,
                                  "out_of_band": 1, "gap_and_out_of_band": 1},
        })

    def test_complete_and_invalid_boundary_overlap_or_episode(self):
        plan = "A,1,2,20,24\n"
        got = self.run_audit(plan, "A,0,1,2,9\nA,1,2,20,24\n")
        self.assertEqual(got.returncode, 0, got.stderr)
        for monitor in ("A,0,1.5,19,24\n",
                        "A,1,1.5,21,22\nA,1.4,2,21,22\n",
                        "B,1,2,20,24\n",
                        "A,1,2,NaN,24\n"):
            with self.subTest(monitor=monitor):
                got = self.run_audit(plan, monitor)
                self.assertEqual(got.returncode, 2)
                self.assertEqual(got.stdout, "")

    def test_io_error_does_not_echo_input_path(self):
        missing = "sensitive-missing-thermal-example.csv"
        got = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] /
             "tools/preflight_rram_thermal_coverage.py"), "--plan", missing,
             "--monitor", missing], capture_output=True, text=True,
        )
        self.assertEqual(got.returncode, 2)
        self.assertEqual(got.stdout, "")
        self.assertEqual(got.stderr, "invalid thermal metadata: check input files and declared intervals\n")
