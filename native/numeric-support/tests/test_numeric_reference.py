"""Reusable reference tests require the real matching extension; all data invented."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import gc
import pickle
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import pandas as pd

from groupbench.data import validate_features
from groupbench.input_support import prediction_input_support
import clotho_numeric_support as adapter
from test_numeric_support import packed


class NumericReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = adapter._load_native()

    def training(self):
        return validate_features(pd.DataFrame({"x": [0., 1., np.nan], "empty": [np.nan] * 3}))

    def test_source_snapshot_does_not_retain_frame_or_mutable_schema(self):
        training = self.training()
        frozen = training.copy(deep=True)
        ref = adapter.NumericSupportReference(training)
        owner = weakref.ref(training)
        training.iloc[:, :] = 1000.
        training.columns = ["changed", "schema"]
        del training
        gc.collect()
        self.assertIsNone(owner())
        query = pd.DataFrame({"empty": [1., np.nan], "x": [-1., 2.]})
        self.assertEqual(ref.compare(query), prediction_input_support(frozen, query))

    def test_immutable_metadata_and_no_serialization(self):
        ref = adapter.NumericSupportReference(self.training())
        for field, value in (("_names", ("fake",)), ("_n_training_rows", 999), ("_reference", None)):
            with self.assertRaises(FrozenInstanceError):
                setattr(ref, field, value)
        with self.assertRaises(FrozenInstanceError):
            del ref._reference
        self.assertFalse(hasattr(ref, "__dict__"))
        self.assertNotIn("1000", repr(ref))
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.assertRaisesRegex(TypeError, "no persistence"):
                pickle.dumps(ref, protocol=protocol)

    def test_reinitialization_is_rejected_without_changing_existing_state(self):
        training = self.training()
        ref = adapter.NumericSupportReference(training)
        expected = ref.compare(training)
        with self.assertRaisesRegex(TypeError, "already initialized"):
            ref.__init__(pd.DataFrame({"different": [1000.]}))
        self.assertEqual(ref.compare(training), expected)

    def test_multiple_batches_and_failed_query_never_expand_reference(self):
        training = self.training()
        ref = adapter.NumericSupportReference(training)
        q = pd.DataFrame({"x": [2., np.nan], "empty": [3., np.nan]})
        expected = prediction_input_support(training, q)
        for x in (1e200, -1e200, 4.):
            ref.compare(pd.DataFrame({"x": [x], "empty": [0.]}))
            with self.assertRaises(ValueError):
                ref.compare(pd.DataFrame({"x": [np.inf], "empty": [0.]}))
            self.assertEqual(ref.compare(q), expected)
        q.iloc[:, :] = 0.
        self.assertEqual(ref.compare(q), prediction_input_support(training, q))

    def test_same_instance_parallel_queries_are_independent(self):
        training = self.training()
        ref = adapter.NumericSupportReference(training)
        queries = [pd.DataFrame({"empty": [float(i), np.nan], "x": [float(i), 0.]}) for i in range(32)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(ref.compare, queries))
        self.assertEqual(actual, [prediction_input_support(training, q) for q in queries])

    def test_reference_rejects_invalid_training_without_partial_instance(self):
        for training in (None, pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [np.inf]}),
                         pd.DataFrame({"x": ["A"]}), pd.DataFrame({"x": []}),
                         pd.DataFrame({"x": pd.Series([1., None], dtype="Float64")})):
            with self.assertRaises((TypeError, ValueError)):
                adapter.NumericSupportReference(training)

    def test_raw_query_exception_type_and_message_equal_python(self):
        training = validate_features(pd.DataFrame({"x": [0., 1.]}))
        ref = adapter.NumericSupportReference(training)
        for query in (None, pd.DataFrame({"x": []}), pd.DataFrame({"wrong": [1.]}),
                      pd.DataFrame({"x": [np.inf]}), pd.DataFrame({"x": [True]}),
                      pd.DataFrame({"x": [1+2j]}), pd.DataFrame({"x": ["A"]}),
                      pd.DataFrame({"x": [2**53+1]}), pd.DataFrame({"x": [0.] * 20_001})):
            try:
                prediction_input_support(training, query)
            except Exception as expected:
                with self.assertRaises(type(expected)) as caught:
                    ref.compare(query)
                self.assertEqual(str(caught.exception), str(expected))
            else:
                self.fail("Negative oracle case unexpectedly accepted")

    def test_reference_protocol_old_extension_fails_without_breaking_one_shot(self):
        training = self.training()
        with patch.object(self.native, "REFERENCE_PROTOCOL", "future"):
            with self.assertRaisesRegex(RuntimeError, "training-reference protocol"):
                adapter.NumericSupportReference(training)
            self.assertEqual(adapter.prediction_input_support_numeric(training, training),
                             prediction_input_support(training, training))
        original = self.native.REFERENCE_PROTOCOL
        try:
            del self.native.REFERENCE_PROTOCOL
            with self.assertRaisesRegex(RuntimeError, "training-reference protocol"):
                adapter.NumericSupportReference(training)
        finally:
            self.native.REFERENCE_PROTOCOL = original

    def test_native_reference_rejects_mutable_or_malformed_bytes(self):
        valid = (packed([0., 1.]),)
        ref = self.native.NumericReference(valid)
        for value in ([valid[0]], (bytearray(valid[0]),), (memoryview(valid[0]),), (),
                      (b"",), (b"x",), (packed([np.inf]),), valid * 101):
            with self.assertRaises(ValueError):
                self.native.NumericReference(value)
            with self.assertRaises(ValueError):
                ref.counts(value)
        with self.assertRaises(ValueError):
            ref.counts(valid * 2)
        self.assertEqual(ref.counts(valid), self.native.counts(valid, valid))

    def test_malformed_reference_output_fails_closed(self):
        class BadReference:
            def counts(self, query):
                return None
        with patch.object(self.native, "NumericReference", return_value=BadReference()):
            ref = adapter.NumericSupportReference(self.training())
        with patch.object(adapter._python_support, "prediction_input_support", side_effect=AssertionError("no fallback")), \
             self.assertRaisesRegex(RuntimeError, "Malformed native"):
            ref.compare(self.training())

    def test_one_shot_error_precedence_is_not_changed(self):
        with self.assertRaisesRegex(ValueError, "complete little-endian"):
            self.native.counts((packed([np.inf]),), (b"x",))
        with self.assertRaisesRegex(ValueError, "finite or NaN"):
            self.native.NumericReference((packed([np.inf]),))


if __name__ == "__main__":
    unittest.main()
