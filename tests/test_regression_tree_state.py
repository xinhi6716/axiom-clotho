"""Wholly synthetic software tests, not real-data predictive-performance evidence."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from groupbench import GroupBenchRegressor, benchmark
from groupbench.cli import main
from groupbench.prediction import fit_predict
import groupbench.model as model_module


TREE_ARRAYS = ("threshold", "value", "impurity", "weighted_n_node_samples")


def synthetic_frame(target=1.0):
    # Exact audited fixture: 16 invented rows, eight invented groups of two.
    return pd.DataFrame({
        "x": np.arange(16, dtype=float),
        "group": np.repeat([f"g{i}" for i in range(8)], 2),
        "target": np.full(16, target),
    })


class RegressionTreeStateTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()
        self.X = self.frame[["x"]]
        self.y = self.frame.target.to_numpy()
        self.groups = self.frame.group.to_numpy()
        self.new = pd.DataFrame({"x": [0.0, 7.0, 15.0]})
        self.options = dict(target="target", group="group", inner_splits=2,
                            baselines=("extra_trees",), random_state=0)

    def fresh(self, **options):
        return GroupBenchRegressor(
            baselines=options.get("baselines", ("extra_trees",)),
            inner_splits=2, random_state=0)

    def assert_unfitted(self, model):
        self.assertFalse(any(key.endswith("_") for key in vars(model)))
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

    def corrupt_fit(self, position, name="impurity", value=np.nan):
        original = Pipeline.fit
        calls = []

        def fit(pipeline, X, y, *args, **kwargs):
            result = original(pipeline, X, y, *args, **kwargs)
            calls.append(pipeline)
            if len(calls) == position:
                # Corrupt the last tree, not merely the first. Never used as data.
                tree = pipeline.named_steps["model"].estimators_[-1].tree_
                getattr(tree, name).flat[0] = value
            return result

        return fit, calls

    def test_exact_extreme_constant_fit_rejected_before_inner_prediction(self):
        model = self.fresh()
        with patch.object(Pipeline, "predict") as predict:
            with self.assertRaisesRegex(ValueError, "non-finite fitted tree state.*impurity"):
                model.fit(self.X, synthetic_frame(float(2**600)).target, self.groups)
            predict.assert_not_called()
        self.assert_unfitted(model)

    def test_exact_extreme_constant_fit_predict_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite fitted tree state"):
            fit_predict(synthetic_frame(float(2**600)), self.new, **self.options)

    def test_exact_extreme_constant_benchmark_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite fitted tree state"):
            benchmark(synthetic_frame(float(2**600)), outer_splits=2, **self.options)

    def test_ordinary_constant_fit_predict_and_benchmark_remain_exact(self):
        prediction = fit_predict(self.frame, self.new, **self.options)
        np.testing.assert_array_equal(prediction.predictions.prediction, np.ones(3))
        self.assertEqual(prediction.estimator.best_candidate_, "extra_trees_depth_6")
        for estimator in prediction.estimator.best_estimator_.named_steps["model"].estimators_:
            for name in TREE_ARRAYS:
                self.assertTrue(np.isfinite(getattr(estimator.tree_, name)).all())
        result = benchmark(self.frame, outer_splits=2, **self.options)
        for column in ("prediction_extra_trees", "prediction_selected"):
            np.testing.assert_array_equal(result.predictions[column], np.ones(16))
        scores = result.summary["models"]["selected"]["overall"]
        self.assertEqual(scores["mae"], 0.0)
        self.assertEqual(scores["rmse"], 0.0)
        self.assertIsNone(scores["r2"])
        json.dumps([prediction.manifest, result.summary, result.manifest], allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(prediction.write(Path(directory) / "prediction").exists())
            self.assertTrue(result.write(Path(directory) / "benchmark").exists())

    def test_each_floating_tree_array_rejects_nan_and_both_infinities(self):
        model = self.fresh().fit(self.X, self.y, self.groups)
        pipeline = model.best_estimator_
        tree = pipeline.named_steps["model"].estimators_[-1].tree_
        for name in TREE_ARRAYS:
            array = getattr(tree, name)
            original = array.flat[0]
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(array=name, value=value):
                    array.flat[0] = value
                    try:
                        with self.assertRaisesRegex(ValueError, f"tree state.*{name}.*Rescale"):
                            model_module._validate_fitted_tree_state(pipeline)
                    finally:
                        array.flat[0] = original
        model_module._validate_fitted_tree_state(pipeline)

    def test_every_inner_and_final_fit_is_checked_before_state_publication(self):
        # Two depth candidates x two folds, then the winning family refit.
        for position in range(1, 6):
            with self.subTest(fit=position):
                model = self.fresh()
                corrupt, calls = self.corrupt_fit(position)
                with patch.object(Pipeline, "fit", corrupt):
                    with self.assertRaisesRegex(ValueError, "tree state.*impurity"):
                        model.fit(self.X, self.y, self.groups)
                self.assertEqual(len(calls), position)
                self.assert_unfitted(model)

    def test_both_fit_boundaries_preserve_previous_complete_state_for_each_array(self):
        model = self.fresh().fit(self.X, self.y, self.groups)
        before = {key: value for key, value in vars(model).items() if key.endswith("_")}
        audit, selection = deepcopy(model.inner_audit_), deepcopy(model.selection_results_)
        predictions = model.predict(self.X)
        baselines = model.predict_baselines(self.X)
        for name in TREE_ARRAYS:
            for position in (1, 5):
                with self.subTest(array=name, fit=position):
                    corrupt, calls = self.corrupt_fit(position, name)
                    with patch.object(Pipeline, "fit", corrupt):
                        with self.assertRaisesRegex(ValueError, f"tree state.*{name}"):
                            model.fit(self.X.rename(columns={"x": "new_x"}), self.y + 1, self.groups)
                    self.assertEqual(len(calls), position)
                    current = {key: value for key, value in vars(model).items() if key.endswith("_")}
                    self.assertEqual(set(current), set(before))
                    for key, value in before.items():
                        self.assertIs(current[key], value, key)
                    self.assertEqual(model.inner_audit_, audit)
                    self.assertEqual(model.selection_results_, selection)
                    np.testing.assert_array_equal(model.predict(self.X), predictions)
                    for family, values in model.predict_baselines(self.X).items():
                        np.testing.assert_array_equal(values, baselines[family])

    def test_unselected_tree_family_refit_is_checked(self):
        # The dummy family wins the exact tie; an invalid unselected tree
        # baseline still fails the whole fit, without skipping that family.
        model = self.fresh(baselines=("dummy", "extra_trees"))
        corrupt, calls = self.corrupt_fit(8)
        with patch.object(Pipeline, "fit", corrupt):
            with self.assertRaisesRegex(ValueError, "tree state.*impurity"):
                model.fit(self.X, self.y, self.groups)
        self.assertEqual(len(calls), 8)
        self.assert_unfitted(model)

    def test_cli_extreme_constant_writes_no_evaluation_or_prediction_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, new_source = root / "synthetic.csv", root / "new.csv"
            synthetic_frame(float(2**600)).to_csv(source, index=False)
            self.new.to_csv(new_source, index=False)
            for mode in ("run", "predict"):
                for existing in (False, True):
                    with self.subTest(mode=mode, existing=existing):
                        output = root / f"{mode}-{existing}"
                        args = [mode, str(source)]
                        if mode == "predict":
                            args.append(str(new_source))
                        else:
                            args.extend(["--outer-splits", "2"])
                        args.extend(["--target", "target", "--group", "group", "--out", str(output),
                                     "--inner-splits", "2", "--baselines", "extra_trees"])
                        if existing:
                            output.mkdir()
                            (output / "predictions.csv").write_bytes(b"existing synthetic output\n")
                            args.append("--overwrite")
                        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
                            code = main(args)
                        self.assertEqual(code, 2)
                        self.assertIn("non-finite fitted tree state", stderr.getvalue())
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
