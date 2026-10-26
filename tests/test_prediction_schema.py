"""Original synthetic saved-bundle fixtures; no network, raw datasets or models."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from groupbench import validate_prediction_bundle_schema
from groupbench.prediction_schema import MAX_MANIFEST_BYTES, MAX_PREDICTION_CSV_BYTES
from groupbench.verification import verify_bundle


def manifest(version="1.1"):
    value = {
        "schema_version": version, "mode": "fit_predict", "n_training_rows": 4,
        "n_training_groups": 2, "n_predictions": 2,
        "feature_schema": [{"name": "voltage", "dtype": "float64"}],
        "config": {"features": ["voltage"]},
    }
    if version != "1.0":
        value["input_support"] = {
            "schema_version": "1.0", "reference": "full_training_features_before_imputation",
            "action": "diagnostic_only_no_input_changes", "n_training_rows": 4,
            "n_prediction_rows": 2, "rows_with_any_flag": 1,
            "row_flag_counts": {"missing_value": 0, "numeric_outside_training_range": 1,
                                "unseen_category": 0, "non_missing_value_in_all_missing_training_feature": 0},
            "features": [{"name": "voltage", "kind": "numeric", "training_missing_count": 0,
                          "training_support": "observed_non_missing_values", "prediction_missing_count": 0,
                          "prediction_non_missing_without_training_values_count": 0,
                          "below_training_min_count": 1, "above_training_max_count": 0}],
        }
    if version == "1.2":
        value["protocol"] = "group-fit-predict-distinct-v2"
        value["config"]["categorical_encoding"] = "distinct-v2"
    return value


def write_bundle(directory, value=None, csv=b"row_position,prediction\n0,1.5\n1,-2e-3\n", raw=None):
    raw = raw if raw is not None else json.dumps(manifest() if value is None else value).encode()
    payloads = {"manifest.json": raw, "predictions.csv": csv}
    for name, data in payloads.items():
        (directory / name).write_bytes(data)
    (directory / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": {
        name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}))


class PredictionSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "run"
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, value=None, **kwargs):
        write_bundle(self.root, value, **kwargs)
        return validate_prediction_bundle_schema(self.root)

    def reject(self, value=None, **kwargs):
        write_bundle(self.root, value, **kwargs)
        # Byte integrity can succeed even when this explicit metadata check fails.
        verify_bundle(self.root)
        with self.assertRaises(ValueError):
            validate_prediction_bundle_schema(self.root)

    def test_three_supported_versions_and_read_only(self):
        for version in ("1.0", "1.1", "1.2"):
            with self.subTest(version=version):
                write_bundle(self.root, manifest(version))
                before = {p.name: p.read_bytes() for p in self.root.iterdir()}
                result = validate_prediction_bundle_schema(self.root)
                self.assertEqual(result.schema_version, version)
                self.assertEqual(result.n_predictions, 2)
                self.assertEqual(result.n_features, 1)
                self.assertEqual(result.input_support == "unavailable", version == "1.0")
                self.assertIn("not_validated", result.unchecked_metadata)
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_unknown_metadata_is_explicitly_not_validated_or_followed(self):
        value = manifest()
        value.update(private_source_path="/never/open/this", future_metadata={"opaque": [1, "label"]})
        value["feature_schema"][0]["dtype"] = "arbitrary label not inferred"
        self.assertEqual(self.check(value).n_predictions, 2)

    def test_unsupported_modes_versions_protocols_and_binary(self):
        variants = [dict(mode="benchmark"), dict(schema_version="2.0"), dict(schema_version=True),
                    dict(schema_id="groupbench-binary-prediction"), dict(protocol="unknown")]
        for update in variants:
            with self.subTest(update=update):
                value = manifest(); value.update(update); self.reject(value)
        value = manifest("1.2"); del value["protocol"]; self.reject(value)
        value = manifest("1.2"); value["config"]["categorical_encoding"] = "legacy-v1"; self.reject(value)
        value = manifest(); value["config"]["categorical_encoding"] = "distinct-v2"; self.reject(value)

    def test_old_support_unavailable_not_zero_or_silently_upgraded(self):
        value = manifest("1.0")
        self.assertEqual(self.check(value).input_support, "unavailable")
        value["input_support"] = manifest()["input_support"]; self.reject(value)
        value = manifest(); del value["input_support"]; self.reject(value)

    def test_count_types_and_bounds(self):
        for key in ("n_training_rows", "n_training_groups", "n_predictions"):
            for bad in (True, False, 0, -1, 1.0, "2", None, 20001):
                with self.subTest(key=key, bad=bad):
                    value = manifest(); value[key] = bad; self.reject(value)
        value = manifest(); value["n_training_groups"] = 5; self.reject(value)

    def test_feature_schema_and_order(self):
        for bad in (None, [], {}, [None], [{"name": "", "dtype": "int64"}],
                    [{"name": "voltage", "dtype": ""}], [{"name": "v\0", "dtype": "float"}],
                    [{"name": "voltage", "dtype": "float"}] * 2,
                    [{"name": str(i), "dtype": "float"} for i in range(101)]):
            with self.subTest(bad=str(bad)[:50]):
                value = manifest(); value["feature_schema"] = bad; self.reject(value)
        value = manifest(); value["config"]["features"] = ["other"]; self.reject(value)

    def test_csv_structure_and_positions(self):
        for text in ("", "prediction,row_position\n1,0\n2,1\n", "row_position,prediction,prediction\n0,1,1\n",
                     "row_position,prediction\n0,1\n", "row_position,prediction\n0,1\n1,2\n2,3\n",
                     "row_position,prediction\n0,1\n0,2\n", "row_position,prediction\n1,1\n0,2\n",
                     "row_position,prediction\n0,1\n\n1,2\n", "row_position,prediction\n0,1,extra\n1,2\n"):
            with self.subTest(text=text): self.reject(csv=text.encode())
        for pos in ("00", "+0", "0.0", " 0", "-0", "０"):
            self.reject(csv=f"row_position,prediction\n{pos},1\n1,2\n".encode())

    def test_prediction_numbers_strict_and_finite(self):
        for token in ("", "NaN", "Infinity", "-inf", "1e309", "1_0", "0x10", "true", " 1", "１", "1" * 65):
            with self.subTest(token=token): self.reject(csv=f"row_position,prediction\n0,{token}\n1,2\n".encode())
        self.check(csv=b'row_position,prediction\r\n0,"+1.0e-9"\r\n1,-0\r\n')

    def test_invalid_encoding_and_nul(self):
        for bad in (b"\xff", b"row_position,prediction\n0,1\x00\n1,2\n", b'row_position,prediction\n0,"1\n1,2\n'):
            self.reject(csv=bad)
        self.reject(raw=b"\xff")

    def test_strict_global_json_even_in_unchecked_metadata(self):
        valid = json.dumps(manifest())
        for suffix in ('"extra":NaN', '"extra":Infinity', '"extra":1e999', '"extra":' + '1'*101,
                       '"extra":1,"extra":2'):
            self.reject(raw=(valid[:-1] + ',' + suffix + '}').encode())
        self.reject(raw=b'[]')
        self.reject(raw=(valid[:-1] + ',"config":{}}').encode())
        self.reject(raw=b'['*1100 + b'0' + b']'*1100)

    def test_support_constants_bounds_and_boolean(self):
        for key, bad in (("schema_version", "2.0"), ("reference", "new_rows"), ("action", "filter"),
                         ("n_training_rows", 3), ("n_prediction_rows", 1), ("rows_with_any_flag", True)):
            value = manifest(); value["input_support"][key] = bad; self.reject(value)
        value = manifest(); value["input_support"]["row_flag_counts"]["new_flag"] = 0; self.reject(value)
        value = manifest(); value["input_support"]["row_flag_counts"]["missing_value"] = True; self.reject(value)

    def test_support_union_inequalities_not_sum_assumption(self):
        value = manifest(); support = value["input_support"]
        support["rows_with_any_flag"] = 0; self.reject(value)
        value = manifest(); value["input_support"]["row_flag_counts"]["numeric_outside_training_range"] = 2; self.reject(value)
        # Overlap between flags is permitted; union need not be their sum.
        value = manifest(); support = value["input_support"]
        other = {"name": "category", "dtype": "object"}
        value["feature_schema"].append(other); value["config"]["features"].append("category")
        support["features"].append({"name": "category", "kind": "categorical", "training_missing_count": 0,
             "training_support": "observed_non_missing_values", "prediction_missing_count": 0,
             "prediction_non_missing_without_training_values_count": 0, "unseen_category_count": 1})
        support["row_flag_counts"]["unseen_category"] = 1
        self.check(value)  # both flags on one row remains possible

    def test_support_missingness_constraints(self):
        for update in ({"training_missing_count": 4}, {"prediction_missing_count": 2},
                       {"prediction_non_missing_without_training_values_count": 1},
                       {"below_training_min_count": None}, {"above_training_max_count": 2},
                       {"unseen_category_count": 0}, {"name": "wrong"}, {"kind": "unknown"}):
            value = manifest(); value["input_support"]["features"][0].update(update); self.reject(value)

    def test_all_missing_numeric_and_categorical(self):
        for kind in ("numeric", "categorical"):
            value = manifest(); support = value["input_support"]; item = support["features"][0]
            item.update(kind=kind, training_missing_count=4, training_support="unavailable_all_missing",
                        prediction_missing_count=1, prediction_non_missing_without_training_values_count=1)
            support["rows_with_any_flag"] = 2
            support["row_flag_counts"].update(missing_value=1, numeric_outside_training_range=0,
                        non_missing_value_in_all_missing_training_feature=1)
            if kind == "numeric":
                item.update(below_training_min_count=None, above_training_max_count=None)
            else:
                del item["below_training_min_count"]; del item["above_training_max_count"]
                item["unseen_category_count"] = 1; support["row_flag_counts"]["unseen_category"] = 1
            self.check(value)
            item["prediction_non_missing_without_training_values_count"] = 0; self.reject(value)

    def test_per_feature_disjoint_flags_bound_overall_union(self):
        value = manifest(); support = value["input_support"]; item = support["features"][0]
        item["prediction_missing_count"] = 1
        support["row_flag_counts"]["missing_value"] = 1
        self.reject(value)  # missing1 + outside1 requires two distinct rows
        support["rows_with_any_flag"] = 2; self.check(value)
        for kind in ("numeric", "categorical"):
            value = manifest(); support = value["input_support"]; item = support["features"][0]
            item.update(kind=kind, training_missing_count=4, training_support="unavailable_all_missing",
                        prediction_missing_count=1, prediction_non_missing_without_training_values_count=1)
            support["row_flag_counts"].update(missing_value=1, numeric_outside_training_range=0,
                                             non_missing_value_in_all_missing_training_feature=1)
            if kind == "numeric":
                item.update(below_training_min_count=None, above_training_max_count=None)
            else:
                del item["below_training_min_count"]; del item["above_training_max_count"]
                item["unseen_category_count"] = 1; support["row_flag_counts"]["unseen_category"] = 1
            self.reject(value)  # all rows missing or unsupported, union cannot be one
            support["rows_with_any_flag"] = 2; self.check(value)

    def test_hash_mismatch_evaluation_and_stale_report(self):
        write_bundle(self.root)
        (self.root / "predictions.csv").write_bytes(b"tampered")
        with self.assertRaises(ValueError): validate_prediction_bundle_schema(self.root)
        write_bundle(self.root)
        (self.root / "report.html").write_text("do not render")
        with self.assertRaises(ValueError): validate_prediction_bundle_schema(self.root)
        (self.root / "report.html").unlink()
        records = json.loads((self.root / "checksums.json").read_text())
        records["files"].update({"report.html": "0" * 64, "summary.json": "0" * 64})
        (self.root / "checksums.json").write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, "evaluation"): validate_prediction_bundle_schema(self.root)

    def test_parser_caps_and_unchanged_byte_verify(self):
        value = manifest(); value["opaque"] = "x" * MAX_MANIFEST_BYTES
        self.reject(value)
        self.reject(csv=b" " * (MAX_PREDICTION_CSV_BYTES + 1))

    def test_exact_public_row_feature_and_manifest_limits(self):
        value = manifest("1.0")
        value.update(n_training_rows=20000, n_training_groups=20000, n_predictions=20000)
        value["feature_schema"] = [{"name": f"x{i}", "dtype": "float64"} for i in range(100)]
        value["config"]["features"] = [f"x{i}" for i in range(100)]
        raw_csv = ("row_position,prediction\n" + "".join(f"{i},1.0\n" for i in range(20000))).encode()
        self.assertEqual(self.check(value, csv=raw_csv).n_predictions, 20000)
        value = manifest("1.0"); value["opaque"] = ""
        empty = json.dumps(value).encode()
        value["opaque"] = "x" * (MAX_MANIFEST_BYTES - len(empty))
        raw = json.dumps(value).encode(); self.assertEqual(len(raw), MAX_MANIFEST_BYTES)
        self.check(raw=raw)
        self.reject(raw=raw + b" ")

    @unittest.skipUnless(os.name == "posix", "POSIX safety API")
    def test_symlinks_fifo_and_path_traversal(self):
        write_bundle(self.root)
        target = self.root / "manifest.json"; saved = target.read_bytes(); target.unlink()
        target.symlink_to(self.root.parent / "outside")
        with self.assertRaises(ValueError): validate_prediction_bundle_schema(self.root)
        target.unlink(); os.mkfifo(target)
        with self.assertRaises(ValueError): validate_prediction_bundle_schema(self.root)
        target.unlink(); target.write_bytes(saved)
        with self.assertRaises(ValueError): validate_prediction_bundle_schema(self.root / ".." / "run")
        link = self.root.parent / "link"; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(OSError): validate_prediction_bundle_schema(link)

    def test_final_metadata_race_detected(self):
        from groupbench import prediction_schema as schema
        write_bundle(self.root)
        original = schema._predictions
        def mutate(raw, count):
            original(raw, count)
            (self.root / "predictions.csv").write_bytes(b"changed after read")
        with patch.object(schema, "_predictions", mutate), self.assertRaisesRegex(ValueError, "changed"):
            validate_prediction_bundle_schema(self.root)

    def test_cli_separate_command_and_no_fit(self):
        from groupbench.cli import main
        write_bundle(self.root, manifest("1.0"))
        with patch("groupbench.cli.fit_predict", side_effect=AssertionError("must not fit")), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["validate-prediction-bundle", str(self.root)]), 0)
            self.assertIn("unavailable", output.getvalue())
            self.assertIn("Additional manifest fields", output.getvalue())
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["verify", str(self.root)]), 0)
            self.assertIn("Byte integrity only", output.getvalue())

    def test_actual_writer_both_encodings(self):
        from groupbench import fit_predict, make_materials_fixture
        frame = make_materials_fixture(4, 2, 7)
        # Column names come from the documented fixture, not inferred identifiers.
        self.assertIn("response", frame.columns)
        features = [c for c in frame if c not in ("response", "batch")]
        for encoding in ("legacy-v1", "distinct-v2"):
            result = fit_predict(frame, frame[features].iloc[:2], target="response", group="batch",
                                 inner_splits=2, baselines=("dummy",), categorical_encoding=encoding)
            out = self.root / encoding
            result.write(out)
            checked = validate_prediction_bundle_schema(out)
            self.assertEqual(checked.schema_version, "1.1" if encoding == "legacy-v1" else "1.2")


if __name__ == "__main__":
    unittest.main()
