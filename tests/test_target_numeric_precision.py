"""Typed target values must not lose distinctions before the scorer sees them."""

from contextlib import ExitStack
from decimal import Decimal
from fractions import Fraction
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from groupbench import (GroupBenchRegressor, benchmark, fit_predict,
                        plan_benchmark, plan_prediction)
from groupbench.data import validate_target


class TargetNumericPrecisionTests(unittest.TestCase):
    def test_object_containers_cannot_hide_complex_values_or_emit_cast_warnings(self):
        for scalar_type in (complex, np.complex64, np.complex128, np.clongdouble):
            for imaginary in (0, 2):
                scalar = scalar_type(1 + imaginary * 1j)
                for y in (np.array([scalar, 1.0], dtype=object), pd.Series([scalar, 1.0], dtype=object)):
                    with self.subTest(type=scalar_type.__name__, imaginary=imaginary), warnings.catch_warnings(record=True) as emitted:
                        warnings.simplefilter("always")
                        with self.assertRaisesRegex(ValueError, "Complex target values are not supported"):
                            validate_target(y, 2)
                        self.assertEqual(emitted, [])

    def test_integer_precision_checked_before_mixed_sequence_promotion(self):
        large = 2 ** 53 + 1
        values = ([large, 0.5], (large, 0.5), np.array([large, 0.5], dtype=object),
                  [np.uint64(2 ** 64 - 1), 0.5],
                  pd.Series([large, 0], dtype="Int64"), pd.Series([large, 0], dtype=object))
        for y in values:
            with self.subTest(container=type(y).__name__):
                with self.assertRaisesRegex(ValueError, "exactly representable in float64"):
                    validate_target(y, 2)

    def test_signed_unsigned_boundaries_and_integer_overflow_reject(self):
        values = ([2 ** 53 + 1, 0], [-(2 ** 53 + 1), 0], [2 ** 2000, 0],
                  np.array([2 ** 63 - 1, 0], dtype=np.int64),
                  np.array([2 ** 64 - 1, 0], dtype=np.uint64))
        for y in values:
            with self.subTest(container=type(y).__name__):
                with self.assertRaisesRegex(ValueError, "exactly representable in float64"):
                    validate_target(y, 2)

    def test_exact_integer_boundaries_and_large_powers_of_two_remain_valid(self):
        values = [2 ** 53 - 1, 2 ** 53, 2 ** 53 + 2, -(2 ** 53 - 1),
                  -(2 ** 53), -(2 ** 53 + 2), 2 ** 60, -(2 ** 63), np.uint64(2 ** 63)]
        for y in (values, np.array(values, dtype=object)):
            result = validate_target(y, len(values))
            self.assertEqual([int(value) for value in result], [int(value) for value in values])

    def test_wider_floats_rejected_in_arrays_and_object_containers(self):
        if np.dtype(np.longdouble).itemsize <= 8:
            self.skipTest("This platform has no wider-than-float64 longdouble dtype.")
        wide = np.longdouble(1)
        values = (np.array([wide, np.nextafter(wide, np.longdouble(2))]),
                  [wide, 0.5], np.array([wide, 0.5], dtype=object),
                  pd.Series([wide, wide], dtype=np.longdouble))
        for y in values:
            with self.subTest(container=type(y).__name__):
                with self.assertRaisesRegex(ValueError, "wider than float64"):
                    validate_target(y, 2)

    def test_supported_float_widths_preserve_their_stored_values(self):
        for dtype in (np.float16, np.float32, np.float64):
            values = np.array([0.1, -0.25, 1.0], dtype=dtype)
            for y in (values, values.tolist(), values.astype(object)):
                with self.subTest(dtype=dtype, container=type(y).__name__):
                    np.testing.assert_array_equal(validate_target(y, 3), values.astype(float))

    def test_no_mask_inputs_use_same_precision_rules_without_mutation(self):
        for mask in (False, np.ma.nomask):
            valid = np.ma.array([2 ** 60, -(2 ** 60)], mask=mask)
            np.testing.assert_array_equal(validate_target(valid, 2), valid.data.astype(float))
            invalid = np.ma.array([2 ** 53 + 1, 0], mask=mask)
            before = invalid.data.copy()
            with self.assertRaisesRegex(ValueError, "exactly representable in float64"):
                validate_target(invalid, 2)
            np.testing.assert_array_equal(invalid.data, before)
            self.assertFalse(np.ma.getmaskarray(invalid).any())

    def test_numeric_text_and_boolean_legacy_parsing_remains_unchanged(self):
        examples = ((["1.25", "2.5"], [1.25, 2.5]),
                    ([True, False], [1.0, 0.0]),
                    (np.array([True, False]), [1.0, 0.0]),
                    (pd.Series([True, False], dtype="boolean"), [1.0, 0.0]))
        for y, expected in examples:
            np.testing.assert_array_equal(validate_target(y, 2), expected)
        # Text is still parsed into float64; this is not a lossless text parser.
        np.testing.assert_array_equal(validate_target([str(2 ** 53), str(2 ** 53 + 1)], 2),
                                      [float(2 ** 53), float(2 ** 53)])

    def test_decimal_and_fraction_legacy_conversion_is_not_certified_lossless(self):
        for values in ([Decimal("0.1"), Decimal("0.2")], [Fraction(1, 3), Fraction(2, 3)],
                       [Decimal(2 ** 53), Decimal(2 ** 53 + 1)],
                       [Fraction(2 ** 53, 1), Fraction(2 ** 53 + 1, 1)]):
            np.testing.assert_array_equal(validate_target(values, 2), [float(value) for value in values])

    def test_unknown_and_overflowing_object_conversions_raise_stable_errors(self):
        for values in ([object(), 0], [Fraction(10 ** 400, 1), 0]):
            with self.subTest(type=type(values[0]).__name__):
                with self.assertRaisesRegex(ValueError, "finite numeric vector"):
                    validate_target(values, 2)

    def test_dataframe_entry_points_reject_before_scheduling_fit_or_writes(self):
        frame = pd.DataFrame({"lot": np.repeat(list("abcdef"), 2),
                              "x": np.arange(12, dtype=float),
                              "target": np.tile(np.array([2 ** 53, 2 ** 53 + 1], dtype=np.int64), 6)})
        before = frame.copy(deep=True)
        for workflow, prediction in ((benchmark, False), (plan_benchmark, False),
                                     (fit_predict, True), (plan_prediction, True)):
            with self.subTest(workflow=workflow.__name__), ExitStack() as stack:
                blocked = [stack.enter_context(patch(name, side_effect=AssertionError("unexpected work")))
                           for name in ("groupbench.splitting.pd.factorize", "groupbench.model.make_pipeline",
                                        "groupbench.report.write_payloads")]
                options = dict(target="target", group="lot", inner_splits=2, baselines=("dummy",))
                with self.assertRaisesRegex(ValueError, "exactly representable in float64"):
                    if prediction:
                        workflow(frame, pd.DataFrame({"x": [1.0]}), **options)
                    else:
                        workflow(frame, outer_splits=3, **options)
                for operation in blocked:
                    operation.assert_not_called()
        pd.testing.assert_frame_equal(frame, before)

    def test_direct_estimator_checks_mixed_targets_before_splitting(self):
        X = pd.DataFrame({"x": np.arange(12, dtype=float)})
        y = [2 ** 53 + 1, 0.5] * 6
        with patch("groupbench.splitting.pd.factorize", side_effect=AssertionError("unexpected split")) as factorize:
            with self.assertRaisesRegex(ValueError, "exactly representable in float64"):
                GroupBenchRegressor(inner_splits=2, baselines=("dummy",)).fit(X, y, groups=np.repeat(list("abcdef"), 2))
            factorize.assert_not_called()

    def test_explicit_caller_transform_and_prior_rounding_are_not_reinterpreted(self):
        source = [2 ** 53, 2 ** 53 + 1]
        # Caller-declared fixed reference, not a learned/global target statistic.
        np.testing.assert_array_equal(validate_target([value - 2 ** 53 for value in source], 2), [0.0, 1.0])
        rounded = np.asarray(source, dtype=float)
        np.testing.assert_array_equal(validate_target(rounded, 2), rounded)
        self.assertEqual(rounded[0], rounded[1])  # lost upstream information is unrecoverable


if __name__ == "__main__":
    unittest.main()
