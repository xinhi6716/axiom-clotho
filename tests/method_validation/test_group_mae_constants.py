"""No-fit tests: these do not reproduce the saved estimator execution."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "benchmarks/method_validation/group_mae_constants/run.py"
SPEC = importlib.util.spec_from_file_location("group_mae_constants_companion", PATH)
companion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(companion)


def independent_macro_loss(y, groups, prediction):
    unique = set(groups)
    return sum((sum(abs(Fraction(float(value)) - Fraction(float(prediction)))
                    for value, group in zip(y, groups) if group == identity)
                / sum(group == identity for group in groups) for identity in unique), Fraction()) / len(unique)


class ConstantTests(unittest.TestCase):
    def test_original_fixture_has_no_information_in_feature(self):
        frame = companion.fixture()
        self.assertEqual(len(frame), 156)
        self.assertEqual(frame.group.nunique(), 12)
        self.assertEqual(frame.row_id.nunique(), 156)
        self.assertEqual(frame.constant_feature.unique().tolist(), [1.0])
        self.assertEqual(sorted(frame.groupby("group").size()), [4] * 9 + [40] * 3)
        self.assertEqual(frame.target.value_counts().to_dict(), {10.0: 120, 0.0: 36})

    def test_weighted_median_is_not_median_of_group_medians(self):
        y = [0, 0, 100, 1, 1, 100, 50, 50, 50]
        groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
        value = companion.training_constants(y, groups)["group_median"]
        self.assertEqual(value, 50)
        self.assertEqual(float(np.median([0, 1, 50])), 1)
        self.assertLess(independent_macro_loss(y, groups, value), independent_macro_loss(y, groups, 1))
        for candidate in sorted(set(y)):
            self.assertLessEqual(independent_macro_loss(y, groups, value), independent_macro_loss(y, groups, candidate))

    def test_equal_group_weights_not_equal_rows(self):
        values = companion.training_constants([10] * 40 + [0] * 4 + [0] * 4, ["a"] * 40 + ["b"] * 4 + ["c"] * 4)
        self.assertEqual(values["row_median"], 10)
        self.assertEqual(values["group_median"], 0)
        self.assertEqual(values["group_mean"], 10 / 3)

    def test_repeating_all_rows_of_one_group_preserves_group_constants(self):
        original = companion.training_constants([0, 2, 7, 10], ["a", "a", "b", "c"])
        repeated = companion.training_constants([0, 2] * 5 + [7, 10], ["a"] * 10 + ["b", "c"])
        for key in ("group_mean", "group_median"):
            self.assertEqual(original[key], repeated[key])

    def test_exact_half_uses_lower_observed_target(self):
        values = companion.training_constants([0, 10], ["a", "b"])
        self.assertEqual(values["group_median"], 0)
        self.assertEqual(values["row_median"], 5)
        self.assertEqual(independent_macro_loss([0, 10], ["a", "b"], 0), Fraction(5))

    def test_constant_uses_only_supplied_training_groups(self):
        values = companion.training_constants([1, 1, 5], ["train_a", "train_a", "train_b"])
        self.assertEqual(values["group_mean"], 3)
        self.assertEqual(values["group_median"], 1)

    def test_nonfinite_targets_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                companion.training_constants([value, 1], ["a", "b"])

    def test_output_collision_before_fit(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(companion, "preflight", return_value={}), patch.object(companion, "benchmark") as fitted:
            with self.assertRaises(FileExistsError):
                companion.run(Path(directory))
            fitted.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_runtime_mismatch_before_fit(self):
        with patch.object(companion, "versions", return_value={}), patch.object(companion, "benchmark") as fitted:
            with self.assertRaisesRegex(ValueError, "runtime mismatch"):
                companion.preflight({"required_versions": {"python": "different"}})
            fitted.assert_not_called()


if __name__ == "__main__":
    unittest.main()
