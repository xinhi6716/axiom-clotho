"""Example contract tests: no estimator is fitted by this suite."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/numeric-unit-equivalence"
spec = importlib.util.spec_from_file_location("numeric_unit_example", EXAMPLE / "run.py")
example = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(EXAMPLE))
try:
    spec.loader.exec_module(example)
finally:
    sys.path.pop(0)


class FakeEstimator:
    """Counts calls and returns zeros; never learns or calls sklearn.fit."""
    def __init__(self, family, arm, X_train, records):
        self.family, self.arm, self.records = family, arm, records
        self.X_train = X_train.copy()
        self.estimators_ = [SimpleNamespace(tree_=SimpleNamespace(node_count=1)) for _ in range(64)]
        if arm == "calculator_scaled":
            scaler = SimpleNamespace(mean_=X_train.mean().to_numpy(),
                                     scale_=X_train.std(ddof=0).to_numpy(), n_samples_seen_=256)
            numeric = SimpleNamespace(named_steps={"scale": scaler})
            preprocess = SimpleNamespace(named_transformers_={"numeric": numeric})
            self.named_steps = {"preprocess": preprocess, "model": self}

    def fit(self, X, y):
        pd.testing.assert_frame_equal(X, self.X_train)
        assert list(X.columns) == ["current"] and len(X) == 256 and len(y) == 256
        self.records.append((self.family, self.arm, X.copy(), np.array(y)))
        return self

    def predict(self, X):
        assert list(X.columns) == ["current"] and len(X) == 128
        return np.zeros(len(X))

    def get_params(self, deep=False):
        return {"mock": True}


class NumericUnitExampleTests(unittest.TestCase):
    def setUp(self):
        self.frame = example.generate()

    def predictions(self):
        return {f"{f}__{a}__{u}": np.zeros(128)
                for f in example.FAMILIES for a in example.ARMS for u in example.UNITS}

    def diagnostics(self):
        return {name: {"tree_node_counts": [1] * 64} for name in self.predictions()}

    def test_generator_is_deterministic_and_original_draw_order(self):
        pd.testing.assert_frame_equal(self.frame, example.generate())
        rng = np.random.default_rng(20261026)
        u = rng.uniform(0, 1, 384)
        offsets = rng.normal(0, 0.10, 12)
        noise = rng.normal(0, 0.05, 384)
        groups = np.repeat(np.arange(12), 32)
        expected_A = 1e-9 * (1 + 9 * u)
        np.testing.assert_array_equal(self.frame.current_A, expected_A)
        np.testing.assert_array_equal(self.frame.current_nA, expected_A * 1e9)
        np.testing.assert_array_equal(self.frame.y, 2 + 3 * u + 0.7 * np.sin(6 * np.pi * u) + offsets[groups] + noise)

    def test_split_is_fixed_group_disjoint(self):
        train, test = example.validate_fixture(self.frame)
        self.assertEqual((sum(train), sum(test)), (256, 128))
        self.assertEqual(set(self.frame.loc[train, "group"]), set(range(8)))
        self.assertEqual(set(self.frame.loc[test, "group"]), set(range(8, 12)))

    def test_bad_schema_split_groups_units_and_nonfinite_rejected(self):
        for column, value in [("group", 8), ("split", "test"), ("row_id", 999),
                              ("current_nA", 0), ("current_A", np.nan), ("y", np.inf)]:
            with self.subTest(column=column):
                altered = self.frame.copy()
                altered.loc[0, column] = value
                with self.assertRaises(ValueError):
                    example.validate_fixture(altered)
        with self.assertRaises(ValueError):
            example.validate_fixture(self.frame.drop(columns="y"))

    def test_pipeline_construction_keeps_train_only_numeric_steps(self):
        for family in example.FAMILIES:
            model = example.build_model(family, "calculator_scaled", pd.DataFrame({"current": [1e-9, 2e-9]}))
            preprocessing = model.named_steps["preprocess"]
            numeric = next(transformer for name, transformer, _ in preprocessing.transformers if name == "numeric")
            self.assertEqual(list(numeric.named_steps), ["impute", "scale"])
            self.assertFalse(hasattr(numeric.named_steps["scale"], "mean_"))

    def test_mocked_schedule_has_twelve_calls_and_same_training_targets(self):
        records = []
        factory = lambda family, arm, X: FakeEstimator(family, arm, X, records)
        with patch.object(example, "score_predictions", side_effect=AssertionError("premature scoring")):
            predictions, diagnostics, count = example.train_predict(self.frame, factory=factory)
        self.assertEqual((count, len(records), len(predictions), len(diagnostics)), (12, 12, 12, 12))
        expected_y = self.frame.loc[self.frame.group < 8, "y"].to_numpy()
        for _, _, X, y in records:
            np.testing.assert_array_equal(y, expected_y)
            self.assertEqual(X.index.tolist(), list(range(256)))

    def test_partial_predictions_cannot_be_scored(self):
        with self.assertRaises(ValueError):
            example.score_predictions(self.frame, {}, {})

    def test_score_math_and_training_only_threshold(self):
        summary = example.score_predictions(self.frame, self.predictions(), self.diagnostics())
        expected = self.frame.loc[self.frame.group >= 8, "y"].abs().mean()
        self.assertAlmostEqual(summary["scores"]["dummy__raw__A"]["group_mae"], expected)
        self.assertFalse(summary["checks"]["raw_tree_gap_and_stumps"])
        self.assertTrue(summary["checks"]["all_scaled_pairs_equivalent"])
        changed = self.frame.copy()
        changed.loc[changed.group >= 8, "y"] += 100
        after = example.score_predictions(changed, self.predictions(), self.diagnostics())
        self.assertEqual(after["prediction_equivalence_threshold"], summary["prediction_equivalence_threshold"])

    def test_nonfinite_predictions_rejected(self):
        predictions = self.predictions()
        predictions["dummy__raw__A"][0] = np.nan
        with self.assertRaises(ValueError):
            example.score_predictions(self.frame, predictions, self.diagnostics())

    def test_existing_output_refused_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep")
            with patch.object(example, "generate", side_effect=AssertionError("must not generate")):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    example.run(output)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_wrong_version_refused_without_output_or_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch.object(example.sklearn, "__version__", "0.invalid"), \
                 patch.object(example, "train_predict", side_effect=AssertionError("must not fit")):
                with self.assertRaisesRegex(ValueError, "requires scikit-learn"):
                    example.run(output)
            self.assertFalse(output.exists())

    def test_wrong_pipeline_refused_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch.object(example, "MODEL_SOURCE_SHA256", "0" * 64):
                with self.assertRaisesRegex(ValueError, "pipeline source differs"):
                    example.run(output)
            self.assertFalse(output.exists())

    def test_json_and_fixture_writers_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "keep"
            output.write_text("sentinel")
            with self.assertRaises(FileExistsError):
                example.write_json(output, {"anything": 1})
            with self.assertRaises(FileExistsError):
                example.write_fixture(self.frame, output)
            self.assertEqual(output.read_text(), "sentinel")


if __name__ == "__main__":
    unittest.main()
