"""Avoid repeated normalization without bypassing public admission checks."""

from contextlib import ExitStack
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import fit_predict, plan_prediction
from groupbench import data, prediction
from groupbench.input_support import prediction_input_support, _prediction_input_support_normalized


OPTIONS = dict(target="target", group="batch", features=["x", "kind"],
               inner_splits=2, baselines=("dummy",))


def fixture():
    return pd.DataFrame({"x": np.arange(12, dtype=float), "kind": ["A", "B"] * 6,
                         "target": np.arange(12, dtype=float), "batch": [f"g{i // 2}" for i in range(12)]})


def queries():
    return {
        "both_missing_opposite_dtypes": pd.DataFrame({"kind": [np.nan, np.nan],
            "x": pd.Series([None, None], dtype=object)}),
        "ordinary_reordered": pd.DataFrame({"kind": ["A", "unknown", None], "x": [1., 13., np.nan]}),
        "categorical_all_missing_numeric_dtype": pd.DataFrame({"x": [1., 2.], "kind": [np.nan, np.nan]}),
        "numeric_all_missing_object_dtype": pd.DataFrame({"x": pd.Series([None, None], dtype=object), "kind": ["A", "B"]}),
    }


# Frozen from main 7edaaabf0da95ac938cb7daf59a1ca6bede48544 with this invented
# fixture and pandas 2.2.3. Hashing is pandas-version dependent, as before.
HISTORICAL_QUERY_HASHES = {
    "both_missing_opposite_dtypes": "1299d43f1a022f4aa49d8877d9325fd315b8295f9d9cf010a77360944a34f46d",
    "ordinary_reordered": "12215d25cadbe0017ba31c6ab3c233b76003c1ede94b01849b1861eb267fbf7b",
    "categorical_all_missing_numeric_dtype": "915c9cd0adee96ba33f9756475e81be06ce273eed760991ec4d64ef9eeccbefd",
    "numeric_all_missing_object_dtype": "cf3ca0eb592c8eb6bd41211010ed2b55812e8e91fea9c7e9f5da1a2a950cc1a1",
}


class PredictionNormalizationTests(unittest.TestCase):
    def test_plan_normalizes_training_and_incoming_once_each(self):
        original = data.validate_features
        with patch.object(data, "validate_features", wraps=original) as training, patch.object(
            prediction, "validate_features", wraps=original
        ) as incoming, patch("groupbench.model.make_pipeline", side_effect=AssertionError("No fits")):
            result = plan_prediction(fixture(), queries()["ordinary_reordered"], **OPTIONS)
        self.assertEqual(training.call_count, 1)
        self.assertEqual(incoming.call_count, 1)
        self.assertEqual(result["fits_performed"], 0)

    def test_fitting_keeps_estimator_prediction_admission(self):
        original = data.validate_features
        with patch.object(data, "validate_features", wraps=original) as general, patch.object(
            prediction, "validate_features", wraps=original
        ) as incoming:
            result = fit_predict(fixture(), queries()["ordinary_reordered"], **OPTIONS)
        # Training, plus the public estimator.predict boundary, remain validated.
        self.assertEqual(general.call_count, 2)
        self.assertEqual(incoming.call_count, 1)
        np.testing.assert_array_equal(result.predictions.prediction, [5.5] * 3)

    def test_public_validator_and_diagnostic_still_validate_raw_inputs(self):
        original = data.validate_features
        source = original(fixture()[["x", "kind"]])
        query = queries()["ordinary_reordered"]
        with patch.object(data, "validate_features", wraps=original) as validate:
            data.validate_prediction_features(query, source.columns, [str(source[c].dtype) for c in source])
            self.assertEqual(validate.call_count, 1)
            prediction_input_support(source, query)
            self.assertEqual(validate.call_count, 2)

    def test_all_missing_alignment_does_not_mutate_fingerprint_input(self):
        source = data.validate_features(fixture()[["x", "kind"]])
        query = data.validate_features(queries()["both_missing_opposite_dtypes"])
        before = query.copy(deep=True)
        aligned = data._align_normalized_prediction_features(
            query, source.columns, [str(source[c].dtype) for c in source])
        self.assertEqual([str(aligned[c].dtype) for c in aligned], ["float64", "object"])
        self.assertEqual([str(query[c].dtype) for c in query], ["float64", "object"])
        self.assertEqual(list(query), ["kind", "x"])
        pd.testing.assert_frame_equal(query, before)
        aligned.loc[0, "x"] = 7.
        pd.testing.assert_frame_equal(query, before)

    def test_normalized_diagnostic_matches_public_boundary_without_mutation(self):
        source = data.validate_features(fixture()[["x", "kind"]])
        for name, raw in queries().items():
            normalized = data.validate_features(raw)
            before_source, before_query = source.copy(deep=True), normalized.copy(deep=True)
            with self.subTest(case=name):
                expected = prediction_input_support(source, raw)
                self.assertEqual(_prediction_input_support_normalized(source, normalized), expected)
                pd.testing.assert_frame_equal(source, before_source)
                pd.testing.assert_frame_equal(normalized, before_query)

    def test_historical_query_fingerprint_and_predictions_are_preserved(self):
        for name, query in queries().items():
            before = query.copy(deep=True)
            normalized = data.validate_features(query)[["x", "kind"]]
            expected = hashlib.sha256(pd.util.hash_pandas_object(normalized, index=False).values.tobytes()).hexdigest()
            with self.subTest(case=name):
                result = fit_predict(fixture(), query, **OPTIONS)
                self.assertEqual(result.manifest["new_features_sha256"], expected)
                # The independent recipe is the portable invariant; frozen
                # hashes additionally protect the environment they came from.
                if pd.__version__ == "2.2.3":
                    self.assertEqual(expected, HISTORICAL_QUERY_HASHES[name])
                np.testing.assert_array_equal(result.predictions.prediction, [5.5] * len(query))
                self.assertEqual(result.manifest["input_support"], plan_prediction(fixture(), query, **OPTIONS)["input_support"])
                pd.testing.assert_frame_equal(query, before)

    def test_invalid_incoming_values_fail_before_scheduling_or_fitting(self):
        good = queries()["ordinary_reordered"].iloc[:2].copy()
        variants = [good.assign(x=[np.inf, 1.]), good.assign(x=np.array([2**53, 2**53 + 1], dtype=np.int64)),
                    good.assign(x=pd.Series([2**53 + 1, None], dtype="Int64")),
                    good.assign(x=pd.date_range("2025-01-01", periods=2)), good.assign(kind=["bad\0value", "A"]),
                    good.assign(x=["not numeric", "still not numeric"]), good.assign(extra=1), good.drop(columns="x")]
        if np.dtype(np.longdouble).itemsize > 8:
            variants.append(good.assign(x=np.array([1., 2.], dtype=np.longdouble)))
        for function in (plan_prediction, fit_predict):
            for number, bad in enumerate(variants):
                before = bad.copy(deep=True)
                with self.subTest(function=function.__name__, invalid=number), ExitStack() as stack:
                    guards = [stack.enter_context(patch(path, side_effect=AssertionError(path))) for path in (
                        "groupbench.prediction.selection_plan", "groupbench.model.make_pipeline",
                        "groupbench.prediction._prediction_input_support_normalized",
                        "groupbench.planning._prediction_input_support_normalized")]
                    with self.assertRaises(ValueError):
                        function(fixture(), bad, **OPTIONS)
                    for guard in guards:
                        guard.assert_not_called()
                pd.testing.assert_frame_equal(bad, before)

    def test_public_raw_boundaries_do_not_treat_normalized_attributes_as_trust(self):
        source = data.validate_features(fixture()[["x", "kind"]])
        bad = pd.DataFrame({"x": [1.], "kind": ["bad\0value"]})
        bad.attrs["normalized"] = True
        for operation in (
            lambda: data.validate_prediction_features(bad, source.columns, [str(source[c].dtype) for c in source]),
            lambda: prediction_input_support(source, bad),
            lambda: plan_prediction(fixture(), bad, **OPTIONS),
        ):
            with self.assertRaisesRegex(ValueError, "NUL"):
                operation()


if __name__ == "__main__":
    unittest.main()
