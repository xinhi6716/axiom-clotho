"""The actual built adapter is required; fixtures are invented and fit no models."""

from concurrent.futures import ThreadPoolExecutor
import json
import struct
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench.data import validate_features
from groupbench.input_support import prediction_input_support
import clotho_numeric_support as adapter


def packed(values):
    return np.array(values, dtype="<f8").tobytes()


class NativeNumericSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = adapter._load_native()  # Missing native code is a failure, not a skip.

    def compare(self, training, query):
        normalized = validate_features(training)
        expected = prediction_input_support(normalized, query)
        actual = adapter.prediction_input_support_numeric(normalized, query)
        self.assertEqual(actual, expected)
        self.assertEqual(json.loads(json.dumps(actual, allow_nan=False)), expected)
        return actual

    def test_numeric_ranges_missingness_and_reordered_columns(self):
        self.compare(pd.DataFrame({"x": [1., 3., np.nan], "constant": [2.] * 3}),
                     pd.DataFrame({"constant": [2., 1., 3., 2., np.nan], "x": [1., 3., 0., 4., np.nan]}))

    def test_all_missing_training_and_query_type_accommodation(self):
        self.compare(pd.DataFrame({"x": [np.nan, np.nan], "y": [1., 2.]}),
                     pd.DataFrame({"y": pd.Series([None, None], dtype=object), "x": [1., np.nan]}))

    def test_extrema_subnormals_signed_zero_and_nan_payloads(self):
        tiny = np.nextafter(0., 1.)
        self.compare(pd.DataFrame({"x": [-np.finfo(float).max, np.finfo(float).max], "tiny": [-0., tiny]}),
                     pd.DataFrame({"x": [0., -np.finfo(float).max, np.nan], "tiny": [-tiny, 0., tiny]}))
        for bits in (0x7FF8000000000001, 0x7FF0000000000001, 0xFFF8000000000001):
            report = self.native.counts((struct.pack("<Q", bits),), (packed([1., np.nan]),))
            self.assertEqual(report[4], [(1, 1, 1, None, None)])

    def test_random_finite_bits_match_without_tolerance(self):
        rng = np.random.default_rng(9173)
        for case in range(96):
            a = rng.integers(0, 2**64, size=(19, 3), dtype=np.uint64).view(np.float64)
            b = rng.integers(0, 2**64, size=(13, 3), dtype=np.uint64).view(np.float64)
            a[~np.isfinite(a)] = np.nan
            b[~np.isfinite(b)] = np.nan
            a[case % 19, case % 3] = np.nan
            self.compare(pd.DataFrame(a, columns=list("abc")), pd.DataFrame(b, columns=list("abc")))

    def test_feature_names_do_not_cross_utf8_boundary(self):
        names = ["", "製程", "line\nbreak", "nul\0name", "high\ud800", "low\udfff"]
        report = self.compare(pd.DataFrame({name: [0., 1.] for name in names}),
                              pd.DataFrame({name: [2., np.nan] for name in reversed(names)}))
        self.assertEqual([feature["name"] for feature in report["features"]], names)

    def test_noncontiguous_reversed_and_big_endian_query_values(self):
        a = np.arange(60., dtype=float).reshape(10, 6)
        training = pd.DataFrame(a[::-1, ::2], columns=list("abc"))
        query = pd.DataFrame(np.asfortranarray(a[::2, ::2]).astype(">f8"), columns=list("abc"))
        self.compare(training, query)

    def test_duplicate_indexes_do_not_change_position_counts_or_mutate_inputs(self):
        training = validate_features(pd.DataFrame({"x": [0., 1.]}, index=[7, 7]))
        training.index = [7, 7]
        query = pd.DataFrame({"x": [2., np.nan]}, index=[-1, -1])
        before, qbefore = training.copy(deep=True), query.copy(deep=True)
        self.assertEqual(adapter.prediction_input_support_numeric(training, query), prediction_input_support(training, query))
        pd.testing.assert_frame_equal(training, before)
        pd.testing.assert_frame_equal(query, qbefore)

    def test_supported_row_and_feature_boundaries(self):
        self.compare(pd.DataFrame({"x": np.zeros(20_000)}), pd.DataFrame({"x": np.ones(20_000)}))
        self.compare(pd.DataFrame({f"x{i}": [0., np.nan] for i in range(100)}),
                     pd.DataFrame({f"x{i}": [1., np.nan] for i in reversed(range(100))}))

    def test_raw_query_rejections_preserve_python_exception_and_message(self):
        training = validate_features(pd.DataFrame({"x": [0., 1.]}))
        bad = [None, pd.DataFrame({"x": []}), pd.DataFrame({"x": [0.] * 20_001}),
               pd.DataFrame({"wrong": [1.]}), pd.DataFrame([[1., 2.]], columns=["x", "x"]),
               pd.DataFrame({"x": [np.inf]}), pd.DataFrame({"x": [1+2j]}),
               pd.DataFrame({"x": pd.to_datetime(["2020-01-01"])}),
               pd.DataFrame({"x": pd.Series([2**53+1], dtype="int64")}),
               pd.DataFrame({"x": ["category"]}), pd.DataFrame({"x": ["bad\0category"]}),
               pd.DataFrame({"x": [True]})]
        if np.dtype(np.longdouble).itemsize > 8:
            bad.append(pd.DataFrame({"x": np.array([1.], dtype=np.longdouble)}))
        for query in bad:
            try:
                prediction_input_support(training, query)
            except Exception as expected:
                with self.subTest(query_type=type(query).__name__), self.assertRaises(type(expected)) as caught:
                    adapter.prediction_input_support_numeric(training, query)
                self.assertEqual(str(caught.exception), str(expected))
            else:
                self.fail("Synthetic negative control did not reject in the Python oracle.")

    def test_training_requires_explicit_plain_float64_normalization(self):
        bad = [None, pd.DataFrame({"x": []}), pd.DataFrame({"x": [0.] * 20_001}),
               pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [True, False]}),
               pd.DataFrame({"x": ["A", "B"]}), pd.DataFrame({"x": pd.Series([1., None], dtype="Float64")}),
               pd.DataFrame({"x": pd.arrays.SparseArray([1., np.nan])}),
               pd.DataFrame({"x": np.array([1.], dtype=np.float32)}),
               pd.DataFrame([[1., 2.]], columns=["x", "x"]), pd.DataFrame({1: [1.]})]
        for source in bad:
            with self.subTest(source_type=type(source).__name__), self.assertRaises((TypeError, ValueError)):
                adapter.prediction_input_support_numeric(source, pd.DataFrame({"x": [0.]}))

    def test_python_categorical_domain_is_not_narrowed_or_silently_encoded(self):
        training = validate_features(pd.DataFrame({"x\ud800": ["v\udfff", "A"]}))
        query = pd.DataFrame({"x\ud800": ["new"]})
        self.assertEqual(prediction_input_support(training, query)["features"][0]["unseen_category_count"], 1)
        with self.assertRaises(ValueError):
            adapter.prediction_input_support_numeric(training, query)

    def test_native_rejects_nonexact_containers_without_callbacks(self):
        class BytesSubclass(bytes):
            pass
        class TupleSubclass(tuple):
            pass
        class Convertible:
            def __bytes__(self):
                raise AssertionError("No conversion callback may run")
            def __iter__(self):
                raise AssertionError("No iterator callback may run")
        valid = (packed([1.]),)
        for bad in ([valid[0]], iter(valid), TupleSubclass(valid), Convertible(), None, "bytes"):
            with self.assertRaises(ValueError):
                self.native.counts(bad, valid)
        for bad in (bytearray(valid[0]), memoryview(valid[0]), BytesSubclass(valid[0]), Convertible(), np.array([1.]), None):
            with self.assertRaises(ValueError):
                self.native.counts((bad,), valid)

    def test_native_shape_limits_and_trailing_bytes_fail_before_a_report(self):
        valid = (packed([1.]),)
        bad = [(), tuple(valid[0] for _ in range(101)), (b"",), (b"x" * 7,), (b"x" * 9,),
               (b"\0" * (20_001 * 8),), (packed([1.]), packed([1., 2.]))]
        for value in bad:
            with self.assertRaises(ValueError):
                self.native.counts(value, valid if len(value) != 2 else valid * 2)
            with self.assertRaises(ValueError):
                self.native.counts(valid if len(value) != 2 else valid * 2, value)
        with self.assertRaises(ValueError):
            self.native.counts(valid, valid * 2)

    def test_infinities_fail_without_value_echo_or_poisoning_later_calls(self):
        valid = (packed([1.]),)
        expected = self.native.counts(valid, valid)
        for value in (np.inf, -np.inf):
            for source, query in (((packed([value]),), valid), (valid, (packed([value]),))):
                with self.assertRaises(ValueError):
                    self.native.counts(source, query)
        self.assertEqual(self.native.counts(valid, valid), expected)

    def test_native_protocol_missing_load_failure_and_no_fallback(self):
        training = validate_features(pd.DataFrame({"x": [0., 1.]}))
        query = pd.DataFrame({"x": [2.]})
        with patch.object(self.native, "PROTOCOL", "future"), self.assertRaises(RuntimeError):
            adapter.prediction_input_support_numeric(training, query)
        with patch.object(adapter._python_support, "NUMERIC_SUPPORT_PROTOCOL", "future"), self.assertRaises(RuntimeError):
            adapter.prediction_input_support_numeric(training, query)
        with patch.object(adapter.importlib, "import_module", side_effect=ImportError("private/path")), self.assertRaises(RuntimeError) as caught:
            adapter.prediction_input_support_numeric(training, query)
        self.assertNotIn("private/path", str(caught.exception))
        with patch.object(self.native, "counts", side_effect=ValueError("synthetic native failure")), \
             patch.object(adapter._python_support, "prediction_input_support", side_effect=AssertionError("no fallback")), \
             self.assertRaisesRegex(ValueError, "synthetic native failure"):
            adapter.prediction_input_support_numeric(training, query)

    def test_unsupported_interpreters_fail_before_native_import(self):
        with patch.object(adapter.sysconfig, "get_config_var", return_value=1), \
             patch.object(adapter.importlib, "import_module", side_effect=AssertionError("must not load")), \
             self.assertRaises(RuntimeError):
            adapter._load_native()

    def test_malformed_native_results_fail_closed(self):
        training = validate_features(pd.DataFrame({"x": [0., 1.]}))
        query = pd.DataFrame({"x": [2., np.nan]})
        valid = self.native.counts((packed([0., 1.]),), (packed([2., np.nan]),))
        bad = [(2, 2, 1, (1, 1, 0, 0), [(0, 1, 0, 0, 1)]), None, list(valid), valid[:-1], (True, *valid[1:]), (3, *valid[1:]),
               (*valid[:2], -1, *valid[3:]), (*valid[:3], (0, 0, 1, 0), valid[4]),
               (*valid[:3], (0, 0, 0, 0), valid[4]), (*valid[:4], []),
               (*valid[:4], [[0, 1, 0, 0, 1]]), (*valid[:4], [(0, 1, 1, 0, 1)]),
               (*valid[:4], [(0, 1, 0, None, None)]), (*valid[:4], [(2, 1, 0, None, None)]),
               (*valid[:4], [(False, 1, 0, 0, 1)]), (*valid[:4], [(0, 1, 0, 2, 1)])]
        for value in bad:
            with self.assertRaisesRegex(RuntimeError, "Malformed native"):
                adapter._report(value, training, query)

    def test_independent_threaded_calls_are_repeatable(self):
        def run(seed):
            rng = np.random.default_rng(seed)
            training = validate_features(pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc")))
            query = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
            return adapter.prediction_input_support_numeric(training, query) == prediction_input_support(training, query)
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertTrue(all(executor.map(run, range(16))))


if __name__ == "__main__":
    unittest.main()
