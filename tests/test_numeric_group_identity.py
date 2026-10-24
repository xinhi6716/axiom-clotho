"""Original synthetic numeric-ID regressions; these tests never fit a model."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import GroupKFold, plan_benchmark
from groupbench.binary import plan_binary_benchmark
from groupbench.cli import _read_csv
from groupbench.data import prepare_data, validate_groups
from groupbench.binary.data import prepare_binary_data


def mixed_precision_fixture():
    # Every pair is the SAME exact binary value, represented by two types.
    groups = []
    for value in (0.1, 0.2, 0.3, 0.4):
        low_precision = np.float32(value)
        groups.extend((low_precision, float(low_precision)))
    return pd.DataFrame({"batch": pd.Series(groups, dtype=object),
                         "x": np.arange(8, dtype=float), "target": [0, 1] * 4})


class NumericGroupIdentityTests(unittest.TestCase):
    def test_equal_stored_values_share_identity_across_float_types(self):
        for dtype in (np.float16, np.float32, np.float64, np.longdouble):
            for value in (0.1, -0.2, 0.5):
                first = dtype(value)
                # The widened value is constructed from the exact same value,
                # not by independently parsing the same decimal spelling.
                second = np.longdouble(first)
                with self.subTest(dtype=dtype, value=value):
                    normalized = validate_groups([first, second], 2)
                    self.assertEqual(normalized[0], normalized[1])

    def test_same_short_decimal_display_does_not_merge_distinct_values(self):
        values = [np.float16(0.1), np.float32(0.1), 0.1]
        self.assertEqual(len({str(value) for value in values}), 1)
        self.assertEqual(len({value.as_integer_ratio() for value in values}), 3)
        self.assertEqual(len(set(validate_groups(values, 3))), 3)

    def test_common_float_text_and_integer_equivalence_remain_stable(self):
        values = [0, -0.0, np.float32(-0.0), 1, 1.0, np.float16(1), 0.1, -0.25]
        self.assertEqual(validate_groups(values, len(values)).tolist(),
                         ["0", "0", "0", "1", "1", "1", "0.1", "-0.25"])

    def test_large_integer_identity_never_narrows_to_float(self):
        values = [2**60, 2**60 + 1, 10**400, 10**400 + 1]
        self.assertEqual(validate_groups(values, 4).tolist(), [str(value) for value in values])

    def test_integer_string_limit_does_not_reject_finite_ids(self):
        huge = 1 << 16000
        values = [huge, huge + 1, -huge]
        normalized = validate_groups(values, len(values))
        self.assertEqual(len(set(normalized)), 3)
        if np.finfo(np.longdouble).maxexp > 16000:
            same = np.ldexp(np.longdouble(1), 16000)
            self.assertEqual(validate_groups([same], 1)[0], normalized[0])

    def test_extended_fraction_does_not_round_to_integer_identity(self):
        if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
            self.skipTest("This platform has no wider-than-float64 longdouble.")
        above_one = np.nextafter(np.longdouble(1), np.longdouble(2))
        below_two = np.nextafter(np.longdouble(2), np.longdouble(1))
        values = [1, above_one, below_two, 2]
        self.assertEqual(len(set(validate_groups(values, 4))), 4)

    def test_extended_precision_subnormals_remain_distinct(self):
        if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
            self.skipTest("This platform has no wider-than-float64 longdouble.")
        tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
        values = [0, tiny, tiny * 2, -tiny]
        self.assertEqual(len(set(validate_groups(values, 4))), 4)

    def test_normalization_is_order_independent(self):
        values = [np.float32(0.1), float(np.float32(0.1)), 0.1,
                  np.longdouble("0.1"), 2**60 + 1]
        forward = validate_groups(values, len(values))
        backward = validate_groups(values[::-1], len(values))[::-1]
        np.testing.assert_array_equal(forward, backward)

    def test_text_identity_is_not_numeric_equivalence(self):
        values = ["1", "01", "1.0", "1e0", " 1", "1 ", "0x1/0x2"]
        self.assertEqual(validate_groups(values, len(values)).tolist(), values)

    def test_mixed_text_numbers_and_invalid_numeric_ids_stay_rejected(self):
        for values in ([1, "1"], [np.float32(0.1), "0.1"], [True, 1],
                       [np.bool_(False), 0], [0, np.inf], [0, np.nan]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_groups(values, len(values))

    def test_split_preserves_original_equal_value_groups(self):
        frame = mixed_precision_fixture()
        original_ids = np.repeat(np.arange(4), 2)
        for seed in range(10):
            seen = []
            for train, test in GroupKFold(2, seed).split(groups=frame.batch):
                with self.subTest(seed=seed, test=test.tolist()):
                    self.assertFalse(set(original_ids[train]) & set(original_ids[test]))
                seen.extend(test)
            self.assertEqual(sorted(seen), list(range(len(frame))))

    def test_equal_numeric_representations_do_not_invent_split_capacity(self):
        value = np.float32(0.1)
        with self.assertRaisesRegex(ValueError, "found 1"):
            list(GroupKFold(2).split(groups=[value, float(value)]))

    def test_regression_and_binary_data_apis_share_correct_group_identity(self):
        frame = mixed_precision_fixture()
        for prepare in (prepare_data, prepare_binary_data):
            with self.subTest(prepare=prepare.__name__):
                _, _, groups = prepare(frame, "target", "batch")
                self.assertEqual(len(set(groups)), 4)
                np.testing.assert_array_equal(groups[::2], groups[1::2])

    def test_regression_and_binary_plans_count_true_groups_without_fitting(self):
        frame = mixed_precision_fixture()
        with ExitStack() as stack:
            for target in ("groupbench.model.Pipeline.fit", "groupbench.model.make_pipeline",
                           "groupbench.binary.model.binary_pipeline"):
                stack.enter_context(patch(target, side_effect=AssertionError("No fitting allowed")))
            for planner in (plan_benchmark, plan_binary_benchmark):
                with self.subTest(planner=planner.__name__):
                    result = planner(frame, target="target", group="batch", outer_splits=2,
                                     inner_splits=2, baselines=("dummy",))
                    self.assertEqual(result["n_training_groups"], 4)
                    self.assertEqual(result["fits_performed"], 0)

    def test_csv_cli_group_spelling_remains_literal_text(self):
        values = ["0.1", "0.10000000149011612", "1", "01", "1.0", "1e0"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original_synthetic.csv"
            path.write_text("batch,x,target\n" + "".join(f"{v},{i},{i % 2}\n" for i, v in enumerate(values)))
            frame = _read_csv(path, group="batch")
        _, _, groups = prepare_data(frame, "target", "batch")
        self.assertEqual(groups.tolist(), values)


if __name__ == "__main__":
    unittest.main()
