"""Prediction-mode and artifact-safety regression tests."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import benchmark, fit_predict, make_materials_fixture
from groupbench.cli import _read_csv, main, protect_input_files
import groupbench.report as report_module


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = make_materials_fixture(6, 4, 7)
        cls.new = make_materials_fixture(2, 3, 19).drop(columns=["response", "batch"])
        cls.result = fit_predict(cls.training, cls.new, target="response", group="batch", inner_splits=2)

    def test_predictions_are_refit_not_evaluation(self):
        result = self.result
        np.testing.assert_array_equal(result.predictions.prediction, result.estimator.predict(self.new))
        self.assertEqual(result.predictions.row_position.tolist(), list(range(len(self.new))))
        self.assertEqual(result.manifest["mode"], "fit_predict")
        self.assertEqual(result.manifest["n_training_rows"], len(self.training))
        self.assertEqual(result.manifest["selected_candidate"], result.estimator.best_candidate_)
        self.assertNotIn("metrics", result.manifest)
        self.assertIn("not held-out evaluation", " ".join(result.manifest["limitations"]))

    def test_new_rows_cannot_affect_selection(self):
        new = self.new.copy()
        new["temperature_c"] += 1e6
        changed = fit_predict(self.training, new, target="response", group="batch", inner_splits=2)
        self.assertEqual(changed.manifest["selected_candidate"], self.result.manifest["selected_candidate"])
        self.assertEqual(changed.manifest["inner_selection"], self.result.manifest["inner_selection"])
        self.assertNotEqual(changed.manifest["new_features_sha256"], self.result.manifest["new_features_sha256"])

    def test_extra_target_and_group_columns_rejected_before_fit(self):
        for column in ("response", "batch"):
            changed = self.new.copy()
            changed[column] = 1
            with patch("groupbench.prediction.GroupBenchRegressor.fit") as fitting:
                with self.assertRaisesRegex(ValueError, "exactly the training feature"):
                    fit_predict(self.training, changed, target="response", group="batch", inner_splits=2)
                fitting.assert_not_called()

    def test_all_missing_new_categorical_is_valid(self):
        changed = self.new.copy()
        changed["material_family"] = np.nan  # CSV inference produces float here.
        predictions = self.result.estimator.predict(changed)
        self.assertTrue(np.isfinite(predictions).all())

    def test_prediction_cannot_reuse_evaluation_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "report.html").write_text("OLD REPORT")
            with self.assertRaisesRegex(ValueError, "separate output directory"):
                self.result.write(out, overwrite=True)
            self.assertEqual((out / "report.html").read_text(), "OLD REPORT")
            self.assertFalse((out / "manifest.json").exists())
            with redirect_stderr(io.StringIO()), patch("groupbench.cli.fit_predict") as fitting:
                code = main(["predict", "unused_train.csv", "unused_new.csv", "--target", "y", "--group", "g", "--out", str(out), "--overwrite"])
            self.assertEqual(code, 2)
            fitting.assert_not_called()

    def test_prediction_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            self.result.write(out)
            self.assertEqual({p.name for p in out.iterdir()}, {"predictions.csv", "manifest.json", "checksums.json"})
            checks = json.loads((out / "checksums.json").read_text())
            for name, digest in checks["files"].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                self.result.write(out)

    def test_predict_cli_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, new, out = root / "train.csv", root / "new.csv", root / "out"
            self.training.to_csv(train, index=False)
            self.new.to_csv(new, index=False)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                code = main(["predict", str(train), str(new), "--target", "response", "--group", "batch", "--out", str(out), "--inner-splits", "2"])
            self.assertEqual(code, 0, err.getvalue())
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["source_file_sha256"], hashlib.sha256(train.read_bytes()).hexdigest())
            self.assertEqual(manifest["prediction_source_file_sha256"], hashlib.sha256(new.read_bytes()).hexdigest())
            self.assertEqual(len(pd.read_csv(out / "predictions.csv")), len(self.new))


class ArtifactSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = benchmark(make_materials_fixture(6, 3), target="response", group="batch", outer_splits=3, inner_splits=2, baselines=("dummy",))

    def test_racing_file_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp, "summary.json")
            real_link = os.link
            def race(source, target, *args, **kwargs):
                if Path(target) == destination:
                    destination.write_text("RACING WRITER")
                return real_link(source, target, *args, **kwargs)
            with patch.object(report_module.os, "link", side_effect=race):
                with self.assertRaises(FileExistsError):
                    self.result.write(temp)
            self.assertEqual(destination.read_text(), "RACING WRITER")

    def test_overwrite_replaces_symlink_not_its_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "precious"
            outside.write_text("KEEP")
            out = root / "out"
            out.mkdir()
            (out / "summary.json").symlink_to(outside)
            self.result.write(out, overwrite=True)
            self.assertEqual(outside.read_text(), "KEEP")
            self.assertFalse((out / "summary.json").is_symlink())

    def test_input_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            source.write_text("KEEP")
            for kind in ("direct", "symlink", "hardlink"):
                out = root / kind
                out.mkdir()
                dest = out / "predictions.csv"
                if kind == "direct":
                    current = dest
                    current.write_text("KEEP")
                elif kind == "symlink":
                    dest.symlink_to(source)
                    current = source
                else:
                    os.link(source, dest)
                    current = source
                with self.assertRaisesRegex(ValueError, "overwrite an input"):
                    protect_input_files([current], out)
                self.assertEqual(current.read_text(), "KEEP")

    def test_predict_cannot_overwrite_new_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, new = root / "train.csv", root / "predictions.csv"
            data = make_materials_fixture(6, 3)
            data.to_csv(train, index=False)
            data.drop(columns=["response", "batch"]).to_csv(new, index=False)
            original = new.read_bytes()
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                code = main(["predict", str(train), str(new), "--target", "response", "--group", "batch", "--out", str(root), "--overwrite"])
            self.assertEqual(code, 2)
            self.assertEqual(new.read_bytes(), original)

    def test_csv_preserves_group_identifier_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "groups.csv")
            path.write_text("batch,x,y\n001,1,2\n1,2,3\nNA,3,4\n")
            frame = _read_csv(path, group="batch")
            self.assertEqual(frame.batch.tolist(), ["001", "1", "NA"])

    def test_oversize_csv_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "large.csv")
            path.write_text("batch,x,y\na,1,2\nb,2,3\nc,3,4\n")
            with patch("groupbench.cli.MAX_ROWS", 2):
                with self.assertRaisesRegex(ValueError, "exceeds the MVP limit"):
                    _read_csv(path, group="batch")


class NumericAndWriterBoundaryTests(unittest.TestCase):
    def test_nonfinite_prediction_rejected(self):
        frame = pd.DataFrame({"g": np.repeat(np.arange(6), 3), "x": np.arange(18, dtype=float)})
        frame["y"] = 1000 * frame["x"]
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(ValueError, "non-finite predictions"):
                fit_predict(frame, pd.DataFrame({"x": [1e308]}), target="y", group="g", inner_splits=2, baselines=("ridge",))

    def test_generic_writer_rejects_path_escape_and_reserved_names(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            for name in ("../escape", "/absolute", "sub/file", "sub\\file", "checksums.json", "", ".."):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    report_module.write_payloads({name: "bad"}, out)
            self.assertFalse(out.exists())


class IntegralProvenanceTests(unittest.TestCase):
    def test_numpy_integers_have_json_safe_manifests(self):
        frame = make_materials_fixture(6, 3)
        result = benchmark(frame, target="response", group="batch", outer_splits=np.int64(3), inner_splits=np.int64(2), random_state=np.int64(7), baselines=("dummy",))
        json.dumps(result.manifest, allow_nan=False)
        new = frame.drop(columns=["response", "batch"]).iloc[:2]
        prediction = fit_predict(frame, new, target="response", group="batch", inner_splits=np.int64(2), random_state=np.int64(7), baselines=("dummy",))
        json.dumps(prediction.manifest, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
