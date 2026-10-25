"""Exact binary64 oracles for saved-prediction scoring; invented values only."""
import copy
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / "examples/environment-heldout-comparison"
spec = importlib.util.spec_from_file_location("saved_paired_contrast_probe", HERE / "probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def oracle(a, b, target):
    a, b, target = [Fraction.from_float(float(v)) for v in (a, b, target)]
    return abs(b - target) - abs(a - target)


class SavedPairedContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = lambda name: (HERE / "fixtures" / name).read_bytes()
        cls.original = probe.predict(fixture("training.csv"), fixture("future.csv"),
                                     fixture("spec.json"), contract_declared=True)

    def inputs(self, cases):
        body = copy.deepcopy(self.original["body"])
        source = body["predictions"][0]
        body["predictions"] = []
        outcomes = ["device_id,target"]
        for index, (a, b, target) in enumerate(cases):
            row = copy.deepcopy(source)
            row["device_id"] = f"invented_{index}"
            row["predictions"] = {arm: 0.0 for arm in probe.ARMS}
            row["predictions"].update(a=a, b=b)
            body["predictions"].append(row)
            outcomes.append(f"{row['device_id']},{target}")
        body["coverage"].update(future_devices=len(cases), future_dependency_blocks=1)
        return probe.report_bytes(probe.seal(body)), ("\n".join(outcomes) + "\n").encode()

    def score(self, cases):
        prediction, outcomes = self.inputs(cases)
        with patch.object(probe, "make_pipeline", side_effect=AssertionError("scoring must not fit")) as fit:
            result = probe.score(prediction, outcomes)["body"]
        fit.assert_not_called()
        self.assertEqual(prediction, self.inputs(cases)[0])
        return result

    def assert_oracle(self, cases):
        result = self.score(cases)
        expected = [oracle(*case) for case in cases]
        self.assertEqual([row["absolute_error_b_minus_a"] for row in result["rows"]],
                         [float(value) for value in expected])
        self.assertEqual(result["paired_mean_absolute_error_b_minus_a"],
                         float(sum(expected, Fraction()) / len(expected)))
        self.assertEqual(result["coverage"]["omitted_devices"], 0)
        return result

    def test_same_side_positive_negative_targets_and_swap(self):
        for a, b, target, expected in [(1., 2., 1e20, -1.), (2., 1., 1e20, 1.),
                                      (1., 2., -1e20, 1.), (2., 1., -1e20, -1.),
                                      (-2., -1., -1e20, 1.), (-1., -2., -1e20, -1.)]:
            with self.subTest(a=a, b=b, target=target):
                result = self.assert_oracle([(a, b, target)] * 4)
                self.assertEqual(result["paired_mean_absolute_error_b_minus_a"], expected)
                self.assertEqual(result["metrics"]["b"]["mae"] - result["metrics"]["a"]["mae"], 0.)

    def test_opposite_sides_preserve_small_offset_and_exact_ties(self):
        for target in (1., -1., 0.):
            for a, b in [(-1e20, 1e20), (1e20, -1e20)]:
                with self.subTest(a=a, target=target):
                    self.assert_oracle([(a, b, target)] * 4)
        self.assertEqual(self.score([(5., 5., 1.)] * 4)["paired_mean_absolute_error_b_minus_a"], 0.)

    def test_aggregate_uses_unrounded_row_contrasts(self):
        cases = [(1., 1e20, 0.), (1e20, 0., 0.)] * 2
        result = self.assert_oracle(cases)
        self.assertEqual(result["paired_mean_absolute_error_b_minus_a"], -.5)
        self.assertEqual(math.fsum(row["absolute_error_b_minus_a"] for row in result["rows"]), 0.)
        self.assertEqual(self.score(list(reversed(cases)))["paired_mean_absolute_error_b_minus_a"], -.5)
        self.assertEqual(self.score([(b, a, y) for a, b, y in cases])["paired_mean_absolute_error_b_minus_a"], .5)

    def test_near_ties_subnormal_and_extreme_legal_predictions(self):
        for case in [(0., 5e-324, 1.), (5e-324, 0., 1.),
                     (1., math.nextafter(1., math.inf), 1e20),
                     (-1e150, 1e150, 1e100), (1e150, -1e150, -1e100),
                     (1, 2, 1e20), (10**20 + 1, 10**20 + 2, 1e20)]:
            with self.subTest(case=case):
                self.assert_oracle([case] * 4)

    def test_generated_sign_and_magnitude_cases(self):
        rng = random.Random(2718)
        cases = [(rng.uniform(-1, 1) * 10.**rng.randint(-100, 150),
                  rng.uniform(-1, 1) * 10.**rng.randint(-100, 150),
                  rng.choice([-1., 0., 1.]) * 10.**rng.randint(-90, 100))
                 for _ in range(200)]
        self.assert_oracle(cases)

    def test_equal_device_mean_at_maximum_cohort(self):
        cases = [(1., 1e20, 0.), (1e20, 0., 0.)] * (probe.MAX_ROWS // 2)
        result = self.assert_oracle(cases)
        self.assertEqual(result["coverage"]["scored_devices"], 10000)
        self.assertEqual(result["paired_mean_absolute_error_b_minus_a"], -.5)

    def test_nonzero_aggregate_underflow_rejects_whole_comparison(self):
        with self.assertRaises(probe.InputError):
            self.score([(0., 5e-324, 1.)] + [(0., 0., 1.)] * 3)
        for value in (Fraction(1, 10**400), Fraction(10**400)):
            with self.assertRaises(probe.InputError):
                probe.contrast_float(value)

    def test_nonfinite_or_out_of_range_predictions_still_rejected(self):
        for value in (float("inf"), float("nan"), 1e151):
            with self.subTest(value=value), self.assertRaises((ValueError, probe.InputError)):
                self.score([(0., value, 1.)] * 4)

    def test_cli_preserves_exact_contrast_and_rejects_underflow_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction, outcomes = root / "predictions.json", root / "outcomes.csv"
            for index, cases in enumerate(([(1., 2., 1e20)] * 4,
                                           [(0., 5e-324, 1.)] + [(0., 0., 1.)] * 3)):
                p, o = self.inputs(cases)
                prediction.write_bytes(p); outcomes.write_bytes(o)
                output = root / f"result_{index}.json"
                result = subprocess.run([sys.executable, str(HERE / "probe.py"), "score",
                                         str(prediction), str(outcomes), "--output", str(output)],
                                        capture_output=True, timeout=20)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(prediction.read_bytes(), p)
                self.assertEqual(outcomes.read_bytes(), o)
                if index == 0:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    body = probe.strict_json(output.read_bytes())["body"]
                    self.assertEqual(body["paired_mean_absolute_error_b_minus_a"], -1.)
                else:
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertFalse(output.exists())
                    self.assertNotIn(b"Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
