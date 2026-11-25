"""Test the distinct estimands in the toy fixed-horizon policy receipt."""

from fractions import Fraction
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


RUN = Path(__file__).resolve().parents[1] / "examples/policy-stop-and-lost-followup/run.py"


class PolicyStopExampleTests(unittest.TestCase):
    def test_readme_values(self):
        output = subprocess.check_output([sys.executable, str(RUN)], text=True)
        self.assertIn("deployed-policy success=1/2; wrongly censoring it gives 2/3", output)
        self.assertIn("success fraction can range from 1/2 to 3/4", output)

    def test_missing_bounds_are_attained_by_observable_completions(self):
        mod = runpy.run_path(str(RUN))
        fn = mod["evaluate"]
        missing = mod["INCOMPLETE"]
        lo, hi, _ = fn(missing[::-1])
        self.assertEqual((lo, hi), (Fraction(1, 2), Fraction(3, 4)))
        self.assertEqual(fn(((5, "event_free_to_horizon"), *missing[1:]))[:2], (lo, lo))
        self.assertEqual(fn(((4, "success"), *missing[1:]))[:2], (hi, hi))

    def test_incomplete_time_not_substituted_for_horizon(self):
        fn = runpy.run_path(str(RUN))["evaluate"]
        with self.assertRaisesRegex(ValueError, "fixed horizon"):
            fn(((2, "event_free_to_horizon"),))
