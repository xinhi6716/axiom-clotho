from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import runpy
import subprocess
import sys
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "audit_rram_retention_stage_bounds.py"
BOUNDS = runpy.run_path(str(TOOL))["identified_bounds"]


def cli(*counts, extra=()):
    flags = ("attempted", "ramp-failures", "ramp-unknown", "later-failures", "later-unknown")
    args = [arg for flag, count in zip(flags, counts) for arg in (f"--{flag}", str(count))]
    return subprocess.run([sys.executable, str(TOOL), *args, *extra], text=True, capture_output=True, check=False)


def endpoints(record):
    return tuple(Fraction(record[f"{name}_numerator"], record["denominator"])
                 for name in ("lower", "upper"))


class RetentionStageBounds(unittest.TestCase):
    def test_cli_reports_both_denominators_and_unestablished_survivor_cohort(self):
        result = cli(128, 16, 8, 14, 7)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["documented_ramp_survivors"], 104)
        self.assertEqual(endpoints(report["attempted_cohort_bounds"]), (Fraction(30, 128), Fraction(45, 128)))
        self.assertEqual(endpoints(report["conditional_on_actual_ramp_survival_bounds"]),
                         (Fraction(14, 112), Fraction(29, 112)))
        unresolved = json.loads(cli(3, 2, 1, 0, 0).stdout)
        self.assertIsNone(unresolved["conditional_on_actual_ramp_survival_bounds"])
        self.assertEqual(endpoints(unresolved["attempted_cohort_bounds"]), (Fraction(2, 3), Fraction(1)))
        for counts, diagnostic in (
            ((0, 0, 0, 0, 0), "attempted must be positive"),
            ((5, -1, 1, 0, 0), "nonnegative integers"),
            ((5, 4, 2, 0, 0), "exceed the attempted cohort"),
            ((5, 2, 2, 1, 1), "exceed documented ramp survivors"),
        ):
            error = cli(*counts)
            self.assertEqual(error.returncode, 2)
            self.assertFalse(error.stdout)
            self.assertIn(diagnostic, error.stderr)

    def test_exhaustive_partial_completions_confirm_sharpness(self):
        # Known state for each attempted unit: ramp fail, ramp unknown,
        # survive/later fail, survive/later unknown, survive/later clear.
        cases = 0
        for n in range(1, 8):
            for ramp, unknown_ramp, late, unknown_late in product(range(n + 1), repeat=4):
                survivors = n - ramp - unknown_ramp
                if survivors <= 0 or late + unknown_late > survivors:
                    continue
                attempted, conditional = [], []
                for completion in product(range(3), repeat=unknown_ramp):
                    # Unknown ramp units can fail in ramp, survive and fail
                    # later, or survive with no later observed failure.
                    for completed_late in product(range(2), repeat=unknown_late):
                        later_count = late + completion.count(1) + sum(completed_late)
                        attempted.append(Fraction(ramp + completion.count(0) + later_count, n))
                        conditional.append(Fraction(later_count, survivors + unknown_ramp - completion.count(0)))
                result = BOUNDS(n, ramp, unknown_ramp, late, unknown_late)
                self.assertEqual(endpoints(result["attempted_cohort_bounds"]), (min(attempted), max(attempted)))
                self.assertEqual(endpoints(result["conditional_on_actual_ramp_survival_bounds"]),
                                 (min(conditional), max(conditional)))
                cases += 1
        self.assertEqual(cases, 756)


    def test_later_failure_does_not_label_survival_through_ramp(self):
        flag = "--later-failures-with-unknown-ramp"
        r = cli(128, 16, 8, 14, 7, extra=(flag, "4"))
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(endpoints(report["attempted_cohort_bounds"]), (Fraction(34, 128), Fraction(45, 128)))
        self.assertEqual(endpoints(report["conditional_on_actual_ramp_survival_bounds"]),
                         (Fraction(14, 108), Fraction(29, 112)))
        self.assertEqual(report["conditional_bound_extremizing_survivor_counts"],
                         {"at_lower_bound_witness": 108, "at_upper_bound_witness": 112})
        self.assertEqual(cli(128, 16, 8, 14, 7, extra=(flag, "9")).returncode, 2)

    def test_extended_bounds_are_sharp_over_unknown_first_stage(self):
        for n in range(1, 7):
            for ramp, unknown, late, censor, observed in product(range(n + 1), repeat=5):
                survivors = n - ramp - unknown
                if survivors <= 0 or late + censor > survivors or observed > unknown:
                    continue
                a, b = [], []
                choices = [(0, 1)] * observed + [(0, 1, 2)] * (unknown - observed)
                for completion in product(*choices):
                    for resolved in range(censor + 1):
                        ramp_new = completion.count(0)
                        late_count = late + completion.count(1) + resolved
                        a.append(Fraction(ramp + ramp_new + late_count, n))
                        b.append(Fraction(late_count, survivors + unknown - ramp_new))
                report = BOUNDS(n, ramp, unknown, late, censor, observed)
                self.assertEqual(endpoints(report["attempted_cohort_bounds"]), (min(a), max(a)))
                self.assertEqual(endpoints(report["conditional_on_actual_ramp_survival_bounds"]),
                                 (min(b), max(b)))
