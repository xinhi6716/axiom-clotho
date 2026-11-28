import json
from pathlib import Path
import subprocess
import sys
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "preflight_rram_classic_linear.py"


class ClassicLinearPreflight(unittest.TestCase):
    def run_plan(self, channels, interval, samples, *, confirmed=True, design=False):
        command = [sys.executable, str(TOOL), "--measurement-channels", str(channels),
                   "--sample-interval-us", str(interval), "--samples-per-channel", str(samples)]
        if confirmed:
            command.append("--confirm-classic-linear-standard-grid")
        if design:
            command.append("--diagnose-joint-anchor-design")
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_displayed_duration_is_not_final_start_and_exact_grid_has_no_spares(self):
        result = self.run_plan(1, 1_000_000, 100_001)
        self.assertEqual(result.returncode, 0, result.stderr)
        p = json.loads(result.stdout)
        self.assertEqual(p["unused_samples_per_channel"], 0)
        self.assertEqual(p["last_planned_start_elapsed_us"], 100_000_000_000)
        self.assertEqual(p["displayed_sampling_duration_us"], 100_001_000_000)
        self.assertTrue(p["exact_1s_on_planned_grid"])
        self.assertTrue(p["exact_100ks_on_planned_grid"])
        p = json.loads(self.run_plan(1, 1_000_000, 100_000).stdout)
        self.assertEqual(p["displayed_sampling_duration_us"], 100_000_000_000)
        self.assertFalse(p["exact_100ks_on_planned_grid"])

    def test_channel_cap_floors_and_extending_interval_loses_early_support(self):
        for channels, cap, latest, minimum in ((2, 50_000, 49_999_000_000, 1_601_000),
                                               (4, 25_000, 24_999_000_000, 3_201_000)):
            p = json.loads(self.run_plan(channels, 1_000_000, cap).stdout)
            self.assertEqual(p["per_channel_sample_cap"], cap)
            self.assertEqual(p["last_planned_start_elapsed_us"], latest)
            self.assertEqual(p["minimum_grid_interval_to_reach_80ks_at_sample_cap_us"], minimum)
            self.assertEqual(p["planned_starts_in_terminal_window"], 0)
            p = json.loads(self.run_plan(channels, minimum, cap).stdout)
            self.assertGreater(p["planned_starts_in_terminal_window"], 0)
            self.assertEqual(p["planned_nonorigin_starts_in_early_window"], 0)
        p = json.loads(self.run_plan(2, 1_020_000, 50_000).stdout)
        self.assertEqual(p["last_planned_start_elapsed_us"], 50_998_980_000)
        self.assertEqual(p["planned_nonorigin_starts_in_early_window"], 1)
        self.assertFalse(p["exact_1s_on_planned_grid"])

    def test_closed_terminal_edges_and_origin_only(self):
        p = json.loads(self.run_plan(1, 2_000_000, 60_001).stdout)
        self.assertEqual(p["planned_starts_in_terminal_window"], 20_001)
        self.assertEqual(p["planned_nonorigin_starts_in_early_window"], 0)
        p = json.loads(self.run_plan(1, 1_000_000, 1).stdout)
        self.assertIsNone(p["first_nonorigin_planned_start_us"])
        self.assertEqual(p["last_planned_start_elapsed_us"], 0)
        self.assertEqual(p["planned_nonorigin_starts_in_early_window"], 0)

    def test_joint_anchor_design_distinguishes_windows_from_exact_nominal_slots(self):
        p = json.loads(self.run_plan(1, 2_000_000, 1, design=True).stdout)["joint_anchor_design"]
        self.assertEqual(p["minimum_samples_window_witness"], {
            "sample_interval_us": 1_020_000, "samples_per_channel": 78_433,
            "witness_terminal_start_us": 80_000_640_000,
        })
        self.assertEqual(p["minimum_samples_exact_nominal_witness"], {
            "sample_interval_us": 1_000_000, "samples_per_channel": 100_001,
            "witness_terminal_start_us": 100_000_000_000,
        })
        self.assertEqual(json.loads(self.run_plan(2, 1_000_000, 1, design=True).stdout)["joint_anchor_design"], {
            "minimum_samples_window_witness": None,
            "minimum_samples_exact_nominal_witness": None,
        })

    def test_unconfirmed_out_of_grid_and_over_budget_fail_without_json(self):
        for result in (self.run_plan(1, 1_000_000, 2, confirmed=False),
                       self.run_plan(2, 1_000_000, 50_001),
                       self.run_plan(1, 1_999, 2),
                       self.run_plan(1, 2_001, 2),
                       self.run_plan(1, 65_536_000, 2),
                       self.run_plan(0, 1_000_000, 2)):
            self.assertEqual(result.returncode, 2)
            self.assertFalse(result.stdout)


if __name__ == "__main__":
    unittest.main()
