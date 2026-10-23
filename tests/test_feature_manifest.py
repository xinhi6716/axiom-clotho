"""Original synthetic checks for the offline feature-schema CLI bridge."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from groupbench import make_materials_fixture
from groupbench.cli import main
from groupbench.feature_manifest import MAX_MANIFEST_BYTES, read_feature_manifest


def schema_for(features=None):
    """Use the upstream example's exact v1 field structure, not its physics."""
    features = ["pressure_mpa", "temperature_c"] if features is None else features
    return {"schema_version": 1, "row_key": ["device"],
            "model_feature_allowlist": features,
            "metadata_columns": ["device", "batch", "operator_note"],
            "target_columns": [], "feature_units": {name: "arbitrary" for name in features},
            "warnings": ["Original synthetic software fixture, not scientific validation."]}


class FeatureManifestTests(unittest.TestCase):
    def read_raw(self, raw):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp, "feature_schema.json")
            source.write_bytes(raw)
            return read_feature_manifest(source)

    def read_schema(self, schema):
        return self.read_raw(json.dumps(schema, ensure_ascii=False).encode("utf-8"))

    def test_exact_schema_shape_returns_only_declared_names_in_original_order(self):
        schema = schema_for(["温度", "a,b"])
        self.assertEqual(self.read_schema(schema), ["温度", "a,b"])
        self.assertEqual(schema["metadata_columns"], ["device", "batch", "operator_note"])

    def test_malformed_utf8_json_and_nonfinite_constants_fail_cleanly(self):
        for raw in (b"", b"\xff", b"{} trailing", b'{"x":}', b"[1,]", b"[" * 2000 + b"]" * 2000,
                    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'):
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                self.read_raw(raw)

    def test_duplicate_object_keys_at_any_depth_are_rejected(self):
        raw = json.dumps(schema_for()).encode("utf-8")
        for duplicate in (raw.replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1'),
                          raw.replace(b'"feature_units": {', b'"feature_units": {"pressure_mpa": "arbitrary",')):
            with self.subTest(raw=duplicate), self.assertRaisesRegex(ValueError, "duplicate JSON object keys"):
                self.read_raw(duplicate)

    def test_schema_fields_version_types_and_role_conflicts_are_rejected(self):
        invalid = [None, [], {}, ["temperature_c"]]
        for key in schema_for():
            missing = schema_for()
            del missing[key]
            invalid.append(missing)
        extra = schema_for()
        extra["load_module"] = "not_executed.py"
        invalid.append(extra)
        for version in (True, False, 1.0, 2, "1", None):
            invalid.append({**schema_for(), "schema_version": version})
        for field in ("row_key", "model_feature_allowlist", "metadata_columns", "target_columns"):
            for names in (None, "temperature_c", [""], [" "], [" padded"], [None], [1], [False], [["x"]], ["x", "x"]):
                invalid.append({**schema_for(), field: names})
        for field in ("row_key", "model_feature_allowlist"):
            invalid.append({**schema_for(), field: []})
        for field in ("metadata_columns", "target_columns", "row_key"):
            invalid.append({**schema_for(), field: ["temperature_c"]})
        invalid.extend([
            {**schema_for(), "row_key": ["not_metadata"]},
            {**schema_for(), "target_columns": ["batch"]},
            {**schema_for(), "feature_units": []},
            {**schema_for(), "feature_units": {}},
            {**schema_for(), "feature_units": {"temperature_c": "degC", "pressure_mpa": 1}},
            {**schema_for(), "feature_units": {"temperature_c": "degC", "pressure_mpa": " "}},
            {**schema_for(), "feature_units": {"temperature_c": "degC", "pressure_mpa": "MPa", "device": "ID"}},
            {**schema_for(), "warnings": "not a list"},
            {**schema_for(), "warnings": [None]},
            {**schema_for(), "warnings": [" "]},
            schema_for([f"feature_{i}" for i in range(101)]),
        ])
        for schema in invalid:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                self.read_schema(schema)

    def test_byte_limit_before_open_and_during_stream_read(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp, "feature_schema.json")
            source.write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
            with patch("groupbench.feature_manifest.os.open") as opening:
                with self.assertRaisesRegex(ValueError, "byte limit"):
                    read_feature_manifest(source)
                opening.assert_not_called()
            source.write_text("{}")
            sizes = []
            def growing_read(_fd, count):
                sizes.append(count)
                return b" " * count
            with patch("groupbench.feature_manifest.os.read", side_effect=growing_read):
                with self.assertRaisesRegex(ValueError, "byte limit"):
                    read_feature_manifest(source)
            self.assertEqual(sum(sizes), MAX_MANIFEST_BYTES + 1)
            self.assertTrue(all(0 < size <= 8192 for size in sizes))

    def test_valid_manifest_at_exact_byte_limit_is_accepted(self):
        raw = json.dumps(schema_for()).encode("utf-8")
        raw += b" " * (MAX_MANIFEST_BYTES - len(raw))
        self.assertEqual(self.read_raw(raw), schema_for()["model_feature_allowlist"])

    def test_newly_opened_size_is_checked_before_reading(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp, "feature_schema.json")
            source.write_text("{}")
            actual_open = os.open
            def grow_before_open(*args, **kwargs):
                source.write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
                return actual_open(*args, **kwargs)
            with patch("groupbench.feature_manifest.os.open", side_effect=grow_before_open), \
                    patch("groupbench.feature_manifest.os.read") as reading:
                with self.assertRaisesRegex(ValueError, "byte limit"):
                    read_feature_manifest(source)
                reading.assert_not_called()

    def test_directories_and_fifos_are_rejected_before_opening(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root]
            if hasattr(os, "mkfifo"):
                fifo = root / "named-pipe"
                os.mkfifo(fifo)
                paths.append(fifo)
            for source in paths:
                with self.subTest(source=source), patch("groupbench.feature_manifest.os.open") as opening:
                    with self.assertRaisesRegex(ValueError, "regular local file"):
                        read_feature_manifest(source)
                    opening.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX descriptor and symlink test")
    def test_opened_descriptor_is_checked_and_closed_if_source_was_swapped(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp, "feature_schema.json")
            source.write_text("{}")
            real_close = os.close
            actual_open = os.open
            with patch("groupbench.feature_manifest.os.open", side_effect=lambda *_args, **_kwargs: actual_open(temp, os.O_RDONLY | os.O_DIRECTORY)), \
                    patch("groupbench.feature_manifest.os.close", wraps=real_close) as closing, \
                    patch("groupbench.feature_manifest.os.read") as reading:
                with self.assertRaisesRegex(ValueError, "regular local file"):
                    read_feature_manifest(source)
                reading.assert_not_called()
                closing.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "symlink test")
    def test_symlinks_to_regular_local_files_follow_csv_input_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, link = root / "feature_schema.json", root / "link.json"
            source.write_text(json.dumps(schema_for()))
            link.symlink_to(source)
            self.assertEqual(read_feature_manifest(link), schema_for()["model_feature_allowlist"])
            link.unlink()
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "regular local file"):
                read_feature_manifest(link)


class FeatureManifestCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train, self.new, self.schema = (self.root / name for name in ("train.csv", "new.csv", "feature_schema.json"))
        self.frame = make_materials_fixture(6, 2, 7)
        self.frame["device"] = [f"invented-{i}" for i in range(len(self.frame))]
        self.frame["operator_note"] = "not a predictor"
        self.frame.to_csv(self.train, index=False)
        self.new_frame = make_materials_fixture(2, 2, 19)
        self.selected = schema_for()["model_feature_allowlist"]
        self.new_frame[self.selected].to_csv(self.new, index=False)
        self.schema.write_text(json.dumps(schema_for()), encoding="utf-8")

    def args(self, command, out, flags):
        return [command, str(self.train), *([str(self.new)] if command == "predict" else []),
                "--target", "response", "--group", "batch", "--out", str(out),
                "--inner-splits", "2", "--baselines", "dummy,ridge",
                *(["--outer-splits", "3"] if command == "run" else []), *flags]

    def invoke(self, command, out, flags):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            code = main(self.args(command, out, flags))
        return code, err.getvalue()

    def test_run_and_predict_match_inline_allowlist_without_adding_metadata(self):
        for command in ("run", "predict"):
            file_out, inline_out = self.root / (command + "-file"), self.root / (command + "-inline")
            with self.subTest(command=command):
                code, err = self.invoke(command, file_out, ["--features-file", str(self.schema)])
                self.assertEqual(code, 0, err)
                code, err = self.invoke(command, inline_out, ["--features", ", ".join(self.selected)])
                self.assertEqual(code, 0, err)
                file_manifest = json.loads((file_out / "manifest.json").read_text())
                inline_manifest = json.loads((inline_out / "manifest.json").read_text())
                self.assertEqual(file_manifest["config"], inline_manifest["config"])
                self.assertEqual(file_manifest["config"]["features"], self.selected)
                self.assertEqual([item["name"] for item in file_manifest["feature_schema"]], self.selected)
                pd.testing.assert_frame_equal(pd.read_csv(file_out / "predictions.csv"), pd.read_csv(inline_out / "predictions.csv"))
                if command == "run":
                    self.assertEqual(file_manifest["run_key"], inline_manifest["run_key"])

    def test_run_and_predict_keep_existing_default_when_neither_option_is_given(self):
        expected = [col for col in self.frame if col not in ("response", "batch")]
        self.frame.loc[:3, expected].to_csv(self.new, index=False)
        for command in ("run", "predict"):
            with self.subTest(command=command):
                out = self.root / command
                code, err = self.invoke(command, out, [])
                self.assertEqual(code, 0, err)
                self.assertEqual(json.loads((out / "manifest.json").read_text())["config"]["features"], expected)

    def test_run_and_predict_reject_mutually_exclusive_options_before_io(self):
        for command in ("run", "predict"):
            with self.subTest(command=command), patch("groupbench.cli._read_csv_with_hash") as reading, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(self.args(command, self.root / command, ["--features", "pressure_mpa", "--features-file", str(self.schema)]))
            self.assertEqual(error.exception.code, 2)
            reading.assert_not_called()

    def test_run_and_predict_reject_bad_or_absent_manifest_before_fitting_and_writing(self):
        for raw in (b"{}", b"", b'{"model_feature_allowlist": ["pressure_mpa"]}', None):
            if raw is None:
                self.schema.unlink()
            else:
                self.schema.write_bytes(raw)
            for command in ("run", "predict"):
                with self.subTest(command=command, raw=raw), patch("groupbench.cli.benchmark") as benchmark, \
                        patch("groupbench.cli.fit_predict") as predict:
                    out = self.root / command
                    code, _ = self.invoke(command, out, ["--features-file", str(self.schema)])
                    self.assertEqual(code, 2)
                    benchmark.assert_not_called()
                    predict.assert_not_called()
                    self.assertFalse(out.exists())

    def test_actual_training_column_validation_rejects_missing_target_and_group_features(self):
        for name in ("absent_feature", "response", "batch"):
            schema = schema_for([name])
            # No role declarations may override the actual --target/--group.
            schema["metadata_columns"] = ["device"]
            self.schema.write_text(json.dumps(schema))
            for command in ("run", "predict"):
                with self.subTest(command=command, name=name), patch("groupbench.model.GroupBenchRegressor.fit") as fitting:
                    out = self.root / command
                    code, err = self.invoke(command, out, ["--features-file", str(self.schema)])
                    self.assertEqual(code, 2)
                    self.assertIn("existing column names" if name == "absent_feature" else "cannot be input features", err)
                    fitting.assert_not_called()
                    self.assertFalse(out.exists())

    def test_prediction_rows_still_need_exactly_the_selected_features(self):
        self.new_frame[self.selected + ["batch"]].to_csv(self.new, index=False)
        with patch("groupbench.model.GroupBenchRegressor.fit") as fitting:
            out = self.root / "predict"
            code, err = self.invoke("predict", out, ["--features-file", str(self.schema)])
        self.assertEqual(code, 2)
        self.assertIn("exactly the training feature columns", err)
        fitting.assert_not_called()
        self.assertFalse(out.exists())

    def test_manifest_output_aliases_are_protected_for_both_commands(self):
        kinds = ("direct", "symlink", "hardlink") if os.name == "posix" else ("direct",)
        original = self.schema.read_bytes()
        for command in ("run", "predict"):
            for kind in kinds:
                with self.subTest(command=command, kind=kind):
                    out = self.root / (command + kind)
                    out.mkdir()
                    destination = out / "manifest.json"
                    source = self.schema
                    if kind == "direct":
                        destination.write_bytes(original)
                        source = destination
                    elif kind == "symlink":
                        destination.symlink_to(source)
                    else:
                        os.link(source, destination)
                    code, err = self.invoke(command, out, ["--features-file", str(source), "--overwrite"])
                    self.assertEqual(code, 2)
                    self.assertIn("overwrite an input file", err)
                    self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
