"""Nested categorical cells cannot be silently reinterpreted as literal text."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import fit_predict, fit_predict_binary, plan_prediction, plan_binary_prediction
from groupbench.data import validate_features


class CategoricalContainerCellTests(unittest.TestCase):
    def setUp(self):
        self.training = pd.DataFrame({
            "state": ["['A']", "A", "B", "C"] * 4,
            "group": np.repeat(list("abcd"), 4),
            "target": [0, 1, 0, 1] * 4,
        })
        self.query = pd.DataFrame({"state": ["A", "['A']"]})

    def test_regression_and_binary_admission_reject_before_model_work(self):
        for workflow in (plan_prediction, fit_predict, plan_binary_prediction, fit_predict_binary):
            for encoding in ("legacy-v1", "distinct-v2"):
                for which in ("train", "query"):
                    with (self.subTest(workflow=workflow.__name__, encoding=encoding, which=which),
                          patch("groupbench.model.make_pipeline", side_effect=AssertionError("unexpected fit")),
                          patch("groupbench.binary.model.binary_pipeline", side_effect=AssertionError("unexpected fit"))):
                        training, query = self.training.copy(), self.query.copy()
                        (training if which == "train" else query).at[1, "state"] = ["A"]
                        with self.assertRaisesRegex(ValueError, "Categorical feature cells.*nested"):
                            workflow(training, query, target="target", group="group",
                                     inner_splits=2, baselines=("dummy",), categorical_encoding=encoding)

    def test_container_forms_reject_but_unused_category_is_not_examined(self):
        values = ([], ["A"], ["A", "B"], ("A",), {"state": "A"}, {"A"},
                  np.array("A"), np.array(["A"]), pd.Series(["A"]))
        for value in values:
            with self.subTest(kind=type(value)), self.assertRaisesRegex(
                    ValueError, "Categorical feature cells.*nested"):
                validate_features(pd.DataFrame({"state": pd.Series([value], dtype=object)}))
        nested = ("unused", "tuple")
        unused = pd.Categorical(["A", None], categories=["A", nested])
        actual = validate_features(pd.DataFrame({"state": unused}))
        self.assertEqual(actual.at[0, "state"], "A")
        self.assertTrue(pd.isna(actual.at[1, "state"]))
        used = pd.Categorical([nested], categories=["A", nested])
        with self.assertRaisesRegex(ValueError, "Categorical feature cells.*nested"):
            validate_features(pd.DataFrame({"state": used}))

    def test_explicit_text_remains_distinct_from_scalar_category(self):
        result = fit_predict(self.training, self.query, target="target", group="group",
                             inner_splits=2, baselines=("ridge",), categorical_encoding="distinct-v2")
        predictions = result.predictions["prediction"]
        self.assertGreater(predictions.iloc[0], predictions.iloc[1])
        self.assertEqual(validate_features(self.query)["state"].tolist(), ["A", "['A']"])


if __name__ == "__main__":
    unittest.main()
