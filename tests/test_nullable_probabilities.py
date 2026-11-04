"""Native pandas nullable numeric probabilities across supported pandas versions."""
import unittest

import numpy as np
import pandas as pd

from groupbench.binary.metrics import binary_metrics, grouped_binary_metrics


DTYPES = ('Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64',
          'Float32', 'Float64')


def containers(values, dtype):
    return (pd.array(values, dtype=dtype), pd.Series(values, dtype=dtype),
            pd.Index(values, dtype=dtype))


class NullableProbabilityTests(unittest.TestCase):
    def test_native_nullable_numeric_containers_match_explicit_numeric_arrays(self):
        y, groups = [0, 1, 0, 1], ['b', 'a', 'b', 'c']
        for dtype in DTYPES:
            values = [0, 1, 0, 1] if not dtype.startswith('Float') else [0., 1., .25, .5]
            expected = binary_metrics(y, values)
            grouped = grouped_binary_metrics(y, values, groups)
            for value in containers(values, dtype):
                before = value.copy()
                with self.subTest(dtype=dtype, container=type(value).__name__):
                    self.assertEqual(binary_metrics(y, value), expected)
                    self.assertEqual(grouped_binary_metrics(y, value, groups), grouped)
                    self.assertEqual(value.dtype, before.dtype)
                    np.testing.assert_array_equal(value.to_numpy(), before.to_numpy())
                    self.assertFalse(value.isna().any())

    def test_missing_values_are_not_filled_or_dropped(self):
        for dtype in DTYPES:
            for value in containers([0, pd.NA], dtype):
                with self.subTest(dtype=dtype, container=type(value).__name__):
                    with self.assertRaises(ValueError):
                        binary_metrics([0, 1], value)
                    with self.assertRaises(ValueError):
                        grouped_binary_metrics([0, 1], value, ['a', 'b'])
                    self.assertEqual(value.isna().tolist(), [False, True])

    def test_out_of_range_native_values_reject_before_probability_narrowing(self):
        cases = [('Int64', [-1, 1]), ('UInt64', [0, 2**64 - 1]),
                 ('Float64', [0., np.nextafter(1., 2.)]), ('Float32', [0., np.inf])]
        for dtype, values in cases:
            for value in containers(values, dtype):
                with self.subTest(dtype=dtype, container=type(value).__name__):
                    with self.assertRaises(ValueError):
                        binary_metrics([0, 1], value)

    def test_booleans_strings_and_generic_objects_are_not_inferred_as_probabilities(self):
        for dtype, values in [('boolean', [False, True]), ('string', ['0', '1']),
                              ('object', [0., 1.]), ('object', [False, 1.])]:
            for value in containers(values, dtype):
                with self.subTest(dtype=dtype, container=type(value).__name__):
                    with self.assertRaises(ValueError):
                        binary_metrics([0, 1], value)

    def test_custom_nullable_subclasses_do_not_enter_native_conversion(self):
        class CustomConversion:
            def to_numpy(self, dtype=None, copy=False, na_value=pd.NA):
                if dtype is not None:
                    raise AssertionError('Custom typed conversion must not run')
                return np.array([0., 1.], dtype=object)

        class CustomFloat(CustomConversion, pd.arrays.FloatingArray):
            pass

        class CustomInteger(CustomConversion, pd.arrays.IntegerArray):
            pass

        for cls, dtype in [(CustomFloat, float), (CustomInteger, int)]:
            value = cls(np.array([0, 1], dtype=dtype), np.array([False, False]))
            with self.subTest(container=cls.__name__):
                with self.assertRaises(ValueError):
                    binary_metrics([0, 1], value)
                with self.assertRaises(ValueError):
                    grouped_binary_metrics([0, 1], value, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
