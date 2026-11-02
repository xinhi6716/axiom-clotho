"""Synthetic numeric-overflow lifecycle checks, not performance evidence."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from groupbench import GroupBenchRegressor, benchmark, fit_predict
from groupbench.cli import main
import groupbench.model as model_module


NUMERIC_ARRAYS = (("impute", "statistics_"), ("scale", "mean_"),
                  ("scale", "var_"), ("scale", "scale_"))


def synthetic_frame(magnitude=1.0):
    return pd.DataFrame({"x": np.tile([-magnitude, magnitude], 8),
                         "group": np.repeat([f"g{i}" for i in range(8)], 2),
                         "target": np.ones(16)})


class RegressionPreprocessingStateTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()
        self.X = self.frame[["x"]]
        self.y = self.frame.target.to_numpy()
        self.groups = self.frame.group.to_numpy()
        self.options = dict(target="target", group="group", inner_splits=2,
                            baselines=("dummy", "ridge"), random_state=0)

    def fresh(self, baselines=("dummy", "ridge")):
        return GroupBenchRegressor(baselines=baselines, inner_splits=2, random_state=0)

    def assert_unfitted(self, model):
        self.assertFalse(any(key.endswith("_") for key in vars(model)))
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

    def test_real_inner_overflow_rejected_before_prediction(self):
        # Squared deviations overflow despite finite inputs and constant targets.
        for family in ("dummy", "ridge"):
            with self.subTest(family=family), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model = self.fresh((family,))
                with patch.object(Pipeline, "predict") as predict:
                    with self.assertRaisesRegex(ValueError, r"numeric state.*scale.var_.*Rescale"):
                        model.fit(synthetic_frame(float(2**600))[["x"]], self.y, self.groups)
                    predict.assert_not_called()
                self.assert_unfitted(model)

    def test_real_final_only_overflow_rejected_before_state_publication(self):
        # Eight-row inner partitions have a finite variance accumulator;
        # sixteen-row family refits overflow it. Both still predict finite 1s.
        for family, expected_calls in (("dummy", 2), ("ridge", 6)):
            with self.subTest(family=family), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model = self.fresh((family,))
                original = Pipeline.predict
                calls = []

                def predict(pipeline, X, *args, **kwargs):
                    result = original(pipeline, X, *args, **kwargs)
                    calls.append(result)
                    return result

                with patch.object(Pipeline, "predict", predict):
                    with self.assertRaisesRegex(ValueError, r"numeric state.*scale.var_"):
                        model.fit(synthetic_frame(float(2**510))[["x"]], self.y, self.groups)
                self.assertEqual(len(calls), expected_calls)
                self.assertTrue(all(np.isfinite(values).all() for values in calls))
                self.assert_unfitted(model)

    def test_every_learned_numeric_array_rejects_nan_and_both_infinities(self):
        model = self.fresh().fit(self.X, self.y, self.groups)
        pipeline = model.best_estimator_
        numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        for step, name in NUMERIC_ARRAYS:
            array = getattr(numeric.named_steps[step], name)
            original = array.flat[0]
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(step=step, attribute=name, value=value):
                    array.flat[0] = value
                    try:
                        with self.assertRaisesRegex(ValueError, f"numeric state.*{step}.{name}"):
                            model_module._validate_fitted_numeric_state(pipeline)
                    finally:
                        array.flat[0] = original
        model_module._validate_fitted_numeric_state(pipeline)

    def test_every_inner_and_family_refit_is_checked(self):
        # Four candidates times two folds, then both family refits.
        original = Pipeline.fit
        for position in range(1, 11):
            with self.subTest(fit=position):
                calls = []

                def fit(pipeline, X, y, *args, **kwargs):
                    result = original(pipeline, X, y, *args, **kwargs)
                    calls.append(pipeline)
                    if len(calls) == position:
                        numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
                        numeric.named_steps["scale"].var_.flat[0] = np.inf
                    return result

                model = self.fresh()
                with patch.object(Pipeline, "fit", fit):
                    with self.assertRaisesRegex(ValueError, "non-finite fitted numeric state"):
                        model.fit(self.X, self.y, self.groups)
                self.assertEqual(len(calls), position)
                self.assert_unfitted(model)

    def test_failed_refit_preserves_previous_state_and_can_recover(self):
        model = self.fresh().fit(self.X, self.y, self.groups)
        before = {key: value for key, value in vars(model).items() if key.endswith("_")}
        selection = deepcopy(model.selection_results_)
        predictions = model.predict_baselines(self.X)
        for magnitude in (float(2**510), float(2**600)):
            with self.subTest(magnitude=magnitude), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                incoming = synthetic_frame(magnitude)[["x"]].rename(columns={"x": "new_x"})
                with self.assertRaisesRegex(ValueError, "numeric state"):
                    model.fit(incoming, self.y + 1, self.groups)
                current = {key: value for key, value in vars(model).items() if key.endswith("_")}
                self.assertEqual(set(current), set(before))
                for key, value in before.items():
                    self.assertIs(current[key], value, key)
                self.assertEqual(model.selection_results_, selection)
                for family, values in model.predict_baselines(self.X).items():
                    np.testing.assert_array_equal(values, predictions[family])
        self.assertIs(model.fit(self.X, self.y + 1, self.groups), model)
        np.testing.assert_array_equal(model.predict(self.X), np.full(16, 2.0))

    def test_valid_constant_empty_missing_and_categorical_features_remain_supported(self):
        frames = [pd.DataFrame({"constant": np.ones(16)}),
                  pd.DataFrame({"empty": np.full(16, np.nan)}),
                  pd.DataFrame({"x": np.tile([1.0, np.nan], 8),
                                "material": [f"m{i % 8}" for i in range(16)]}),
                  pd.DataFrame({"material": [f"m{i % 8}" for i in range(16)]})]
        for X in frames:
            with self.subTest(columns=list(X)):
                model = self.fresh(("dummy", "ridge", "extra_trees")).fit(X, self.y, self.groups)
                for values in model.predict_baselines(X).values():
                    np.testing.assert_array_equal(values, np.ones(16))
                self.assertEqual(model.best_candidate_, "dummy_mean")

    def test_valid_predictions_selection_and_audits_are_unchanged(self):
        X = self.X.assign(material=np.tile(["a", "b", "c", None], 4))
        y = np.arange(16, dtype=float) + self.X.x.to_numpy()
        options = ("dummy", "ridge", "extra_trees")
        with patch.object(model_module, "_validate_fitted_numeric_state"):
            reference = self.fresh(options).fit(X, y, self.groups)
        actual = self.fresh(options).fit(X, y, self.groups)
        self.assertEqual(actual.selection_results_, reference.selection_results_)
        self.assertEqual(actual.family_candidates_, reference.family_candidates_)
        self.assertEqual(actual.inner_audit_, reference.inner_audit_)
        for family, values in actual.predict_baselines(X).items():
            np.testing.assert_array_equal(values, reference.predict_baselines(X)[family])

    def test_public_evaluation_and_prediction_reject_real_overflow(self):
        frame = synthetic_frame(float(2**600))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "numeric state"):
                benchmark(frame, outer_splits=2, **self.options)
            with self.assertRaisesRegex(ValueError, "numeric state"):
                fit_predict(frame, self.X, **self.options)

    def test_cli_failure_does_not_create_or_replace_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, new_source = root / "synthetic.csv", root / "new.csv"
            synthetic_frame(float(2**600)).to_csv(source, index=False)
            self.X.to_csv(new_source, index=False)
            for mode in ("run", "predict"):
                for existing in (False, True):
                    with self.subTest(mode=mode, existing=existing):
                        output = root / f"{mode}-{existing}"
                        args = [mode, str(source)]
                        args.extend([str(new_source)] if mode == "predict" else ["--outer-splits", "2"])
                        args.extend(["--target", "target", "--group", "group", "--out", str(output),
                                     "--inner-splits", "2", "--baselines", "dummy,ridge"])
                        if existing:
                            output.mkdir()
                            (output / "predictions.csv").write_bytes(b"existing synthetic output\n")
                            args.append("--overwrite")
                        with warnings.catch_warnings(), redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
                            warnings.simplefilter("ignore", RuntimeWarning)
                            code = main(args)
                        self.assertEqual(code, 2)
                        self.assertIn("non-finite fitted numeric state", stderr.getvalue())
                        self.assertIn("Rescale", stderr.getvalue())
                        self.assertNotIn("Traceback", stderr.getvalue())
                        self.assertEqual(stdout.getvalue(), "")
                        if existing:
                            self.assertEqual(list(output.iterdir()), [output / "predictions.csv"])
                            self.assertEqual((output / "predictions.csv").read_bytes(), b"existing synthetic output\n")
                        else:
                            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
