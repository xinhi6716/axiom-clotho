"""Reject ambiguous observed categorical values before modeling/provenance."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from groupbench import (GroupBenchRegressor, GroupBenchClassifier, benchmark, benchmark_binary,
                        fit_predict, fit_predict_binary, plan_benchmark, plan_binary_benchmark,
                        plan_prediction, plan_binary_prediction)
from groupbench.data import prepare_data, validate_features, validate_prediction_features

BAD_VALUES = ("A\0B", "A\0C", "A\0", "A\0\0", "\0A", "\0", "批次\0甲", "__GROUPBENCH_MISSING__\0")


def fixture():
    return pd.DataFrame({"g": np.repeat(["a", "b", "c", "d", "e", "f"], 4),
                         "x": np.arange(24, dtype=float), "kind": ["A", "B"] * 12,
                         "y": [0, 1] * 12})


class CategoricalNULTests(unittest.TestCase):
    def test_observed_nul_rejected_across_dtypes_without_mutation(self):
        for value in BAD_VALUES:
            for dtype in (object, "string", "category"):
                frame = pd.DataFrame({"kind": pd.Series(["clean", value, None], dtype=dtype)})
                before = frame.copy(deep=True)
                with self.subTest(value=repr(value), dtype=dtype), self.assertRaisesRegex(
                    ValueError, "Categorical feature values.*NUL \\(U\\+0000\\)"
                ):
                    validate_features(frame)
                pd.testing.assert_frame_equal(frame, before)

    def test_error_does_not_disclose_feature_name_or_cell(self):
        with self.assertRaises(ValueError) as caught:
            validate_features(pd.DataFrame({"PRIVATE_COLUMN": ["PRIVATE_VALUE\0secret"]}))
        for fragment in ("PRIVATE", "secret"):
            self.assertNotIn(fragment, str(caught.exception))

    def test_normalization_then_validation_not_raw_source_type_only(self):
        class TextValue:
            def __str__(self):
                return "normalized\0value"
        with self.assertRaisesRegex(ValueError, "Categorical feature values.*NUL"):
            validate_features(pd.DataFrame({"kind": [TextValue()]}))

    def test_unused_nul_category_metadata_does_not_reject_clean_rows(self):
        for values in (["A", "B", None], [None, None, None]):
            frame = pd.DataFrame({"kind": pd.Categorical(values, categories=["A", "B", "unused\0category"])})
            before = frame.copy(deep=True)
            result = validate_features(frame)
            expected = pd.DataFrame({"kind": pd.Series([np.nan if value is None else value for value in values], dtype=object)})
            pd.testing.assert_frame_equal(result, expected)
            pd.testing.assert_frame_equal(frame, before)

    def test_ordinary_unicode_missing_literals_and_numeric_values_unchanged(self):
        values = ["A", "A\\0", "A\\x00", "A␀", "批次", "é", "e\u0301", "🙂", None]
        frame = pd.DataFrame({"kind": values, "x": np.arange(len(values)),
                              "flag": [True, False, None] * 3})
        result = validate_features(frame)
        self.assertEqual(result.kind.iloc[:-1].tolist(), values[:-1])
        self.assertTrue(pd.isna(result.kind.iloc[-1]))
        self.assertEqual(result.x.tolist(), np.arange(len(values), dtype=float).tolist())
        self.assertEqual(result.flag.iloc[:2].tolist(), ["True", "False"])
        self.assertTrue(pd.isna(result.flag.iloc[2]))

    def test_selected_features_only_unused_columns_are_not_newly_rejected(self):
        frame = fixture().assign(unused="not_selected\0value")
        X, _, groups = prepare_data(frame, "y", "g", features=["x", "kind"])
        self.assertEqual(list(X), ["x", "kind"])
        self.assertEqual(len(set(groups)), 6)

    def test_prediction_type_accommodation_does_not_bypass_nul_check(self):
        for value in BAD_VALUES:
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, "Categorical feature values.*NUL"):
                validate_prediction_features(pd.DataFrame({"kind": [value]}), ["kind"], ["object"])
        actual = validate_prediction_features(pd.DataFrame({"kind": [np.nan]}), ["kind"], ["object"])
        self.assertEqual(str(actual.kind.dtype), "object")
        self.assertTrue(actual.kind.isna().all())

    def test_all_workflow_training_and_incoming_paths_reject_before_work(self):
        workflows = ((plan_benchmark, False), (benchmark, False),
                     (plan_prediction, True), (fit_predict, True),
                     (plan_binary_benchmark, False), (benchmark_binary, False),
                     (plan_binary_prediction, True), (fit_predict_binary, True))
        blocked = ("groupbench.model.make_pipeline", "groupbench.binary.model.binary_pipeline",
                   "groupbench.splitting.pd.factorize", "groupbench.prediction.prediction_input_support",
                   "groupbench.planning.prediction_input_support", "groupbench.binary.workflow.prediction_input_support",
                   "groupbench.prediction.data_fingerprint", "groupbench.evaluation.data_fingerprint",
                   "groupbench.binary.workflow.data_fingerprint", "groupbench.report.write_payloads")
        for encoding in ("legacy-v1", "distinct-v2"):
            for workflow, prediction in workflows:
                for source in (("training", "incoming") if prediction else ("training",)):
                    frame = fixture()
                    new = pd.DataFrame({"x": [1., 2.], "kind": ["A", "B"]})
                    (frame if source == "training" else new).loc[0, "kind"] = "A\0B"
                    before, before_new = frame.copy(deep=True), new.copy(deep=True)
                    with self.subTest(encoding=encoding, workflow=workflow.__name__, source=source), ExitStack() as stack:
                        mocks = [stack.enter_context(patch(path, side_effect=AssertionError(path))) for path in blocked]
                        options = dict(target="y", group="g", inner_splits=2, baselines=("dummy",),
                                       categorical_encoding=encoding)
                        if not prediction:
                            options["outer_splits"] = 3
                        with self.assertRaisesRegex(ValueError, "Categorical feature values.*NUL"):
                            workflow(frame, new, **options) if prediction else workflow(frame, **options)
                        for mocked in mocks:
                            mocked.assert_not_called()
                    pd.testing.assert_frame_equal(frame, before)
                    pd.testing.assert_frame_equal(new, before_new)

    def test_direct_estimators_reject_before_fit_or_predict_and_preserve_state(self):
        frame = fixture()
        X = frame[["x", "kind"]].copy()
        X.loc[0, "kind"] = "A\0B"
        for estimator_type in (GroupBenchRegressor, GroupBenchClassifier):
            estimator = estimator_type(baselines=("dummy",), inner_splits=2)
            sentinel = object()
            estimator.best_estimator_ = sentinel
            estimator.feature_names_in_ = np.array(["x", "kind"], dtype=object)
            estimator.feature_types_ = ["float64", "object"]
            with self.subTest(estimator=estimator_type.__name__), patch.object(
                estimator, "_fit_prepared", side_effect=AssertionError("Must not fit")
            ) as fitting:
                with self.assertRaisesRegex(ValueError, "Categorical feature values.*NUL"):
                    estimator.fit(X, frame.y, groups=frame.g)
                fitting.assert_not_called()
                methods = [estimator.predict]
                methods += ([estimator.predict_proba, estimator.predict_proba_baselines]
                            if estimator_type is GroupBenchClassifier else [estimator.predict_baselines])
                for method in methods:
                    with self.assertRaisesRegex(ValueError, "Categorical feature values.*NUL"):
                        method(X)
                self.assertIs(estimator.best_estimator_, sentinel)


if __name__ == "__main__":
    unittest.main()
