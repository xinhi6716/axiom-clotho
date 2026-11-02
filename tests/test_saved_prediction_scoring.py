"""Invented complete delayed cohorts, never real measurements or fitted models."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from groupbench import cli
from groupbench import saved_scoring as scoring
from groupbench.prediction_schema import validate_prediction_bundle_schema, _load_prediction_bundle


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_bundle(root, predictions=(1., 4., 2.), version="1.0"):
    root.mkdir(exist_ok=True)
    n = len(predictions)
    manifest = {"schema_version": version, "mode": "fit_predict", "n_training_rows": 6,
                "n_training_groups": 3, "n_predictions": n,
                "feature_schema": [{"name": "x", "dtype": "float64"}], "config": {"features": ["x"]},
                "private_source_path": "/PRIVATE_NEVER_OPEN/source.csv"}
    if version != "1.0":
        manifest["input_support"] = {
            "schema_version": "1.0", "reference": "full_training_features_before_imputation",
            "action": "diagnostic_only_no_input_changes", "n_training_rows": 6,
            "n_prediction_rows": n, "rows_with_any_flag": 0,
            "row_flag_counts": {k: 0 for k in ("missing_value", "numeric_outside_training_range", "unseen_category", "non_missing_value_in_all_missing_training_feature")},
            "features": [{"name": "x", "kind": "numeric", "training_missing_count": 0,
                          "training_support": "observed_non_missing_values", "prediction_missing_count": 0,
                          "prediction_non_missing_without_training_values_count": 0,
                          "below_training_min_count": 0, "above_training_max_count": 0}]}
    if version == "1.2":
        manifest["protocol"] = "group-fit-predict-distinct-v2"
        manifest["config"]["categorical_encoding"] = "distinct-v2"
    if version == "1.3":
        # Invented structurally consistent records only, not evidence of fitting.
        manifest.update(protocol="group-fit-predict-extended-families-v1",
                        capabilities=["group_median-v1"], selected_family="group_median")
        manifest["config"].update(baselines=["dummy", "group_median"],
                                  categorical_encoding="legacy-v1", inner_splits=3, random_state=7)
        def details(rows, groups):
            return {"implementation": "group-balanced-lower-median-v1", "training_rows": rows,
                    "training_groups": groups, "transformed_features": 1, "constant": 2.,
                    "weight_rule": "equal_training_group_then_equal_row", "tie_break": "lower_observed_target"}
        manifest["inner_folds"] = [{"fold": f, "train_rows": 4, "test_rows": 2,
                                      "train_group_count": 2, "test_group_count": 1} for f in range(3)]
        manifest["inner_selection"] = [{"family": "group_median", "candidate": "group_median_lower",
                                        "inner_fit_details": [{"inner_fold": f, "train_rows": 4,
                                                               "test_rows": 2, "details": details(4, 2)} for f in range(3)]}]
        manifest["family_fit_details"] = {"group_median": details(6, 3)}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["row_position", "prediction"])
    writer.writerows(enumerate(predictions))
    payloads = {"manifest.json": json.dumps(manifest).encode(), "predictions.csv": stream.getvalue().encode()}
    for name, raw in payloads.items():
        (root / name).write_bytes(raw)
    (root / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": {n: digest(b) for n, b in payloads.items()}}))


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class SavedPredictionScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bundle = self.root / "prediction"
        self.outcomes = self.root / "later.csv"
        make_bundle(self.bundle)
        self.raw = b"row_position,group,observed\n2,batch_B,1.5\n0,batch_A,2.0\n1,batch_A,3.0\n"
        self.outcomes.write_bytes(self.raw)

    def score(self, raw=None):
        if raw is not None:
            self.outcomes.write_bytes(raw)
        return scoring.score_saved_predictions(self.bundle, self.outcomes)

    def reject_rows(self, text):
        with self.assertRaises(scoring.SavedScoringError):
            self.score(text if isinstance(text, bytes) else text.encode())

    def test_hand_calculated_unequal_groups_and_provenance(self):
        before = snapshot(self.root)
        result = self.score()
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(result["fits_performed"], 0)
        self.assertEqual((result["n_rows"], result["n_groups"]), (3, 2))
        overall = result["overall"]
        for key, expected in {"mae": 5/6, "rmse": math.sqrt(3/4), "bias": 1/6,
                              "r2": -13/14, "group_mae": .75, "group_rmse": .75, "worst_group_mae": 1.}.items():
            self.assertAlmostEqual(overall[key], expected)
        self.assertEqual([r["row_position"] for r in result["scored_rows"]], [0, 1, 2])
        self.assertEqual([r["observed"] for r in result["scored_rows"]], [2., 3., 1.5])
        self.assertEqual([r["group"] for r in result["per_group"]], ["batch_A", "batch_B"])
        self.assertIsNone(result["per_group"][1]["r2"])
        self.assertEqual(result["outcomes_sha256"], digest(self.raw))
        self.assertEqual(result["prediction_bundle"]["files_sha256"],
                         {p.name: digest(p.read_bytes()) for p in self.bundle.iterdir()})
        self.assertNotIn("PRIVATE", json.dumps(result))
        json.dumps(result, allow_nan=False)

    def test_row_reordering_changes_only_source_hash(self):
        first = self.score()
        second = self.score(b"row_position,group,observed\n0,batch_A,2.0\n1,batch_A,3.0\n2,batch_B,1.5\n")
        self.assertNotEqual(first.pop("outcomes_sha256"), second.pop("outcomes_sha256"))
        self.assertEqual(first, second)

    def test_literal_text_groups_remain_distinct_and_whitespace_is_not_normalized(self):
        make_bundle(self.bundle, [1, 2, 3, 4, 5])
        result = self.score("row_position,group,observed\n0,01,1\n1,1,2\n2,NA,3\n3,晶圓,4\n4, A ,5\n".encode())
        self.assertEqual([r["group"] for r in result["per_group"]], ["01", "1", "NA", "晶圓", " A "])
        self.assertEqual(result["overall"]["mae"], 0)

    def test_canonical_positions_complete_no_duplicates_or_aliases(self):
        header = "row_position,group,observed\n"
        cases = ["0,A,2\n1,A,3\n", "0,A,2\n1,A,3\n2,B,1.5\n3,C,5\n",
                 "0,A,2\n0,A,3\n2,B,1.5\n"]
        for bad in ("00", "+0", "0.0", "0e0", " 0", "-1", "3", "100000"):
            cases.append(bad + ",A,2\n1,A,3\n2,B,1.5\n")
        for rows in cases:
            with self.subTest(rows=rows):
                self.reject_rows(header + rows)

    def test_numeric_rejection_before_float_inference(self):
        for token in ("True", "False", "NaN", "Infinity", "1+2j", "0x1", "", "1 V", " 1", "1e400", "1e-400", "9007199254740993", "1"*65):
            with self.subTest(token=token):
                self.reject_rows("row_position,group,observed\n0,A," + token + "\n1,A,3\n2,B,1.5\n")
        # Exactly represented large integer is accepted by the numeric helper;
        # use equal constant values so metric intermediates remain practical.
        make_bundle(self.bundle, [9007199254740992] * 3)
        result = self.score(b"row_position,group,observed\n0,A,9007199254740992\n1,A,9007199254740992\n2,B,9007199254740992\n")
        self.assertEqual(result["overall"]["mae"], 0)
        self.assertIsNone(result["overall"]["r2"])

    def test_saved_prediction_precision_loss_cannot_create_false_perfect_scores(self):
        for token, observed in (("1e-999", "0"), ("9007199254740993", "9007199254740992")):
            with self.subTest(token=token):
                make_bundle(self.bundle, [token])
                # Structural acceptance stays unchanged; scoring is a stricter
                # numeric subset and must not silently turn these into equality.
                self.assertEqual(validate_prediction_bundle_schema(self.bundle).n_predictions, 1)
                self.reject_rows("row_position,group,observed\n0,A," + observed + "\n")

    def test_finite_conversion_but_unsafe_metric_arithmetic_rejects(self):
        self.reject_rows("row_position,group,observed\n0,A,1e200\n1,A,2e200\n2,B,3e200\n")
        make_bundle(self.bundle, [0., 0., 0.])
        self.reject_rows("row_position,group,observed\n0,A,1e-200\n1,A,2e-200\n2,B,3e-200\n")

    def test_csv_exact_header_width_encoding_and_group_bounds(self):
        bad = [b"", b"row_position,observed,group\n0,2,A\n", b"row_position,group,group\n0,A,2\n",
               self.raw.replace(b"batch_B", b""), self.raw.replace(b"batch_B", b"   "),
               self.raw.replace(b"batch_B", b"g"*257), self.raw.replace(b"batch_B", b"bad\0id"),
               self.raw.replace(b"batch_B", b"\xff"), self.raw + b"\n", self.raw.replace(b"2,batch_B,1.5", b"2,batch_B"),
               self.raw.replace(b"2,batch_B,1.5", b"2,batch_B,1.5,extra"),
               self.raw.replace(b"2,batch_B,1.5", b'2,"broken,1.5')]
        for raw in bad:
            with self.subTest(raw=raw[:40]):
                self.reject_rows(raw)
        bom = self.score(b"\xef\xbb\xbf" + self.raw)
        self.assertEqual(bom["outcomes_sha256"], digest(b"\xef\xbb\xbf" + self.raw))

    def test_byte_and_row_bounds_fail_without_partial_result(self):
        with patch.object(scoring, "MAX_OUTCOMES_BYTES", len(self.raw)-1):
            with self.assertRaises(scoring.SavedScoringError): self.score()
        with patch.object(scoring, "MAX_ROWS", 2):
            with self.assertRaises(scoring.SavedScoringError): self.score()

    def test_changed_outcome_file_is_rejected_and_not_reopened_for_values(self):
        real_open = cli._open_csv_source
        @contextmanager
        def changing(path):
            with real_open(path) as source:
                class Reader:
                    changed = False
                    def fileno(self): return source.fileno()
                    def read(self, size):
                        data = source.read(size)
                        if data and not self.changed:
                            self.changed = True
                            with path.open("ab") as f: f.write(b"\n")
                        return data
                yield Reader()
        with patch.object(cli, "_open_csv_source", side_effect=changing):
            with self.assertRaisesRegex(scoring.SavedScoringError, "changed during reading"):
                self.score()

    def test_consumes_exact_accepted_snapshot_not_reopened_prediction_path(self):
        original = scoring._load_prediction_bundle
        expected_hash = digest((self.bundle / "predictions.csv").read_bytes())
        def validated_then_replaced(path):
            result = original(path)
            (self.bundle / "predictions.csv").write_bytes(b"PRIVATE_CHANGED_AFTER_ACCEPTANCE")
            return result
        with patch.object(scoring, "_load_prediction_bundle", side_effect=validated_then_replaced):
            result = self.score()
        self.assertEqual([r["prediction"] for r in result["scored_rows"]], [1.,4.,2.])
        self.assertEqual(result["prediction_bundle"]["files_sha256"]["predictions.csv"], expected_hash)

    def test_no_model_fit_or_source_loading_and_inputs_unchanged(self):
        before = snapshot(self.root)
        with patch("groupbench.model.GroupBenchRegressor.fit", side_effect=AssertionError("fit")), \
             patch("groupbench.prediction.fit_predict", side_effect=AssertionError("fit_predict")), \
             patch("groupbench.evaluation.benchmark", side_effect=AssertionError("benchmark")), \
             patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("pipeline fit")):
            self.score()
        self.assertEqual(before, snapshot(self.root))

    def test_legacy_versions_preserve_validator_result_and_hashes(self):
        for version in ("1.0", "1.1", "1.2"):
            with self.subTest(version=version):
                make_bundle(self.bundle, version=version)
                schema = validate_prediction_bundle_schema(self.bundle)
                loaded, manifest, raw, hashes = _load_prediction_bundle(self.bundle)
                self.assertEqual(schema, loaded)
                self.assertEqual(raw, (self.bundle / "predictions.csv").read_bytes())
                self.assertEqual(hashes["manifest.json"], digest((self.bundle / "manifest.json").read_bytes()))
                self.assertEqual(self.score()["prediction_bundle"]["schema_version"], version)

    def test_extended_schema_uses_same_loader_and_zero_fit_scoring_path(self):
        make_bundle(self.bundle, predictions=(2., 2., 2.), version="1.3")
        before = snapshot(self.root)
        result = self.score()
        self.assertEqual(result["prediction_bundle"]["schema_version"], "1.3")
        self.assertAlmostEqual(result["overall"]["mae"], .5)
        self.assertEqual(result["fits_performed"], 0)
        self.assertEqual(snapshot(self.root), before)

    def test_bundle_corruption_stale_evaluation_and_unsupported_mode_fail(self):
        (self.bundle / "predictions.csv").write_bytes(b"tampered")
        with self.assertRaises(scoring.SavedScoringError): self.score()
        make_bundle(self.bundle)
        (self.bundle / "summary.json").write_text("{}")
        with self.assertRaises(scoring.SavedScoringError): self.score()
        (self.bundle / "summary.json").unlink()
        manifest = json.loads((self.bundle / "manifest.json").read_bytes()); manifest["mode"] = "binary"
        raw = json.dumps(manifest).encode(); (self.bundle / "manifest.json").write_bytes(raw)
        checksums = json.loads((self.bundle / "checksums.json").read_bytes()); checksums["files"]["manifest.json"] = digest(raw)
        (self.bundle / "checksums.json").write_text(json.dumps(checksums))
        with self.assertRaises(scoring.SavedScoringError): self.score()

    @unittest.skipUnless(os.name == "posix", "safe bundle reader is POSIX-only")
    def test_special_files_and_prediction_symlinks_rejected_outcome_local_alias_allowed(self):
        original = self.bundle / "predictions.csv"; target = self.root / "saved.csv"; original.rename(target); original.symlink_to(target)
        with self.assertRaises(scoring.SavedScoringError): self.score()
        original.unlink(); shutil.copyfile(target, original)
        alias = self.root / "alias.csv"; alias.symlink_to(self.outcomes)
        self.assertEqual(scoring.score_saved_predictions(self.bundle, alias), self.score())
        fifo = self.root / "fifo"; os.mkfifo(fifo)
        with self.assertRaises(scoring.SavedScoringError): scoring.score_saved_predictions(self.bundle, fifo)
        original.unlink(); os.mkfifo(original)
        with self.assertRaises(scoring.SavedScoringError): self.score()

    def test_cli_json_no_writes_and_redacted_rejection(self):
        args = ["score-predictions", "--run-dir", str(self.bundle), "--outcomes", str(self.outcomes)]
        before = snapshot(self.root)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err): self.assertEqual(cli.main(args), 0)
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(json.loads(out.getvalue()), self.score())
        self.assertEqual(before, snapshot(self.root))
        for bad_args in (args[:-1]+[str(self.root/"PRIVATE_MISSING")], ["score-predictions", "--PRIVATE_OPTION"]):
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                try: code = cli.main(bad_args)
                except SystemExit as error: code = error.code
            self.assertEqual((code, out.getvalue()), (2, ""))
            self.assertNotIn("PRIVATE", err.getvalue()); self.assertNotIn(str(self.root), err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_real_module_cli_and_help(self):
        env = dict(os.environ, PYTHONPATH=str(Path(scoring.__file__).resolve().parents[1]), PYTHONDONTWRITEBYTECODE="1")
        for module in ("groupbench", "mlworkflow"):
            prefix = [sys.executable, "-m", module] + (["tabular"] if module=="mlworkflow" else [])
            result = subprocess.run(prefix+["score-predictions", "--run-dir", str(self.bundle), "--outcomes", str(self.outcomes)], env=env, cwd=self.root, capture_output=True, text=True, timeout=20)
            self.assertEqual((result.returncode, result.stderr), (0, ""))
            self.assertEqual(json.loads(result.stdout), self.score())
            result = subprocess.run(prefix+["score-predictions", "--help"], env=env, cwd=self.root, capture_output=True, text=True, timeout=20)
            self.assertEqual((result.returncode, result.stderr), (0, "")); self.assertIn("--outcomes", result.stdout)


if __name__ == "__main__":
    unittest.main()
