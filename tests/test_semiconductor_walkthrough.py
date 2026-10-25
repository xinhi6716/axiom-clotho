"""Synthetic-only wrapper and actual local cross-repository regression tests."""
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "examples/semiconductor-heldout-walkthrough"
spec = importlib.util.spec_from_file_location("semiconductor_walkthrough_example", HERE / "run.py")
walk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walk)
SEMICONDUCTOR = os.environ.get("SEMICONDUCTOR_TOOLBOX_SOURCE")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def body(path):
    return json.loads(path.read_bytes())["body"]


def copy_endpoints(root):
    root.mkdir()
    for name in ("training_endpoints.csv", "heldout_outcomes.csv"):
        shutil.copyfile(HERE / "fixtures" / name, root / name)
    return root


class WalkthroughContractTests(unittest.TestCase):
    def test_exact_csv_shapes_byte_bound_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            for blob in (b"a,b\n1,2,3\n", b"a,b\n1\n", b"a,a\n1,2\n", b"a,b\n", b"\xff", b"x" * (1024 * 1024 + 1)):
                with self.subTest(blob_size=len(blob)):
                    path.write_bytes(blob)
                    with self.assertRaises(walk.WalkthroughError):
                        walk.read_csv(path, ("a", "b"))
            path.write_bytes(b"a,b\n1,2\n")
            self.assertEqual(walk.read_csv(path, ("a", "b")), [{"a": "1", "b": "2"}])

    def test_literal_invented_acceptance_before_sources(self):
        with patch.object(walk, "component_hashes") as inspect:
            for value in (False, None, 1, "true"):
                with self.assertRaises(walk.WalkthroughError):
                    walk.run("missing", "missing", "unused", invented_example_declared=value)
            inspect.assert_not_called()

    def test_existing_directory_file_and_dangling_symlink_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dir").mkdir()
            (root / "file").write_bytes(b"preserve")
            (root / "link").symlink_to(root / "absent", target_is_directory=True)
            with patch.object(walk, "component_hashes") as inspect:
                for name in ("dir", "file", "link"):
                    with self.assertRaises(walk.WalkthroughError):
                        walk.run("missing", "missing", root / name, invented_example_declared=True)
                inspect.assert_not_called()
            self.assertEqual((root / "file").read_bytes(), b"preserve")
            self.assertTrue((root / "link").is_symlink())

    def test_exclusive_write_and_restricted_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result"
            walk.write_new(path, b"first")
            with self.assertRaises(FileExistsError):
                walk.write_new(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_stage_uses_same_python_train_source_and_redacts_failure(self):
        log = []
        failed = subprocess.CompletedProcess([], 9, b"private stdout", b"/private/path")
        with patch.object(walk.subprocess, "run", return_value=failed) as execute:
            with self.assertRaises(walk.WalkthroughError) as caught:
                walk.run_stage("example", Path("script.py"), ["input.csv"], log, Path("calculator"))
        self.assertNotIn("private", str(caught.exception))
        args, kwargs = execute.call_args
        self.assertEqual(args[0], [sys.executable, "script.py", "input.csv"])
        self.assertEqual(kwargs["env"]["PYTHONPATH"], "calculator/src")
        self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(kwargs["timeout"], 120)
        self.assertEqual(log, [{"stage": "example", "status": "command_rejected", "exit_code": 9}])

    def test_stage_timeout_is_categorical(self):
        log = []
        with patch.object(walk.subprocess, "run", side_effect=subprocess.TimeoutExpired("private", 120)):
            with self.assertRaises(walk.WalkthroughError):
                walk.run_stage("example", Path("script.py"), [], log, ROOT)
        self.assertEqual(log[0]["status"], "execution_failed")

    def test_cli_missing_sources_never_creates_output_or_prints_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            result = subprocess.run([sys.executable, str(HERE / "run.py"), "--semiconductor-source", "/SECRET", "--calculator-source", "/SECRET", "--output-dir", str(output), "--run-invented-example"], capture_output=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            self.assertEqual(result.returncode, 2)
            self.assertNotIn(b"SECRET", result.stderr)
            self.assertFalse(output.exists())


@unittest.skipUnless(SEMICONDUCTOR, "set SEMICONDUCTOR_TOOLBOX_SOURCE to a trusted matching local checkout; no download is attempted")
class WalkthroughIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.output = cls.root / "baseline"
        cls.source = Path(SEMICONDUCTOR)
        cls.receipt = walk.run(cls.source, ROOT, cls.output, invented_example_declared=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_clis_counts_units_target_free_join_and_metrics(self):
        receipt = self.receipt
        self.assertEqual([r["stage"] for r in receipt["stages"]], ["original_synthetic_generator", "measurement_to_features", "actual_lot_holdout_audit", "target_free_prediction", "separate_exact_cohort_scoring"])
        self.assertTrue(all(r["exit_code"] == 0 for r in receipt["stages"]))
        cohort = rows(self.output / "cohort.csv")
        self.assertEqual(len(cohort), 18)
        self.assertEqual(len({r["lot_id"] for r in cohort}), 3)
        self.assertEqual(len({r["wafer_id"] for r in cohort}), 6)
        self.assertTrue(all(r["physical_device_id"].startswith("invented_fab/") for r in cohort))
        self.assertEqual(len(rows(self.output / "raw/iv_sweeps.csv")), 324)
        features = rows(self.output / "features/features.csv")
        self.assertEqual(sum(int(r["cycle_count"]) for r in features), 54)
        training, future = rows(self.output / "training.csv"), rows(self.output / "future.csv")
        self.assertEqual((len(training), len(future)), (12, 6))
        self.assertNotIn("target", future[0])
        self.assertEqual(set(future[0]), {"device_id", "dependency_block", "environment", *walk.FEATURES})
        self.assertFalse({r["dependency_block"] for r in training} & {r["dependency_block"] for r in future})
        audit = json.loads((self.output / "holdout.json").read_bytes())
        self.assertEqual(audit["status"], "NO_SELECTED_HOLDOUT_OVERLAP")
        prediction, score = body(self.output / "predictions.json"), body(self.output / "scores.json")
        self.assertEqual(prediction["fit_contract"]["attempted_fits"], 5)
        self.assertEqual(score["coverage"]["scored_devices"], 6)
        self.assertEqual(score["target_unit"], "arbitrary_test_units")
        for arm, metrics in score["metrics"].items():
            errors = [r["absolute_errors"][arm] for r in score["rows"]]
            self.assertAlmostEqual(metrics["mae"], sum(errors) / 6)
            self.assertAlmostEqual(metrics["rmse"], (sum(e * e for e in errors) / 6) ** .5)
        self.assertLess(score["metrics"]["dummy"]["mae"], score["metrics"]["a"]["mae"])
        self.assertLess(score["metrics"]["dummy"]["mae"], score["metrics"]["b"]["mae"])
        text = (self.output / "INTERPRETATION.txt").read_text()
        for phrase in ("repeat the same six", "not causal-effect", "not a formal permutation", "higher than the mean dummy", "Native400 is a separate workflow"):
            self.assertIn(phrase, text)

    def test_receipt_integrity_private_root_and_repeated_templates(self):
        for name, digest in self.receipt["output_files_sha256"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(self.receipt["component_source_sha256"], walk.component_hashes(self.source, ROOT))
        self.assertIn("walkthrough/run.py", self.receipt["component_source_sha256"])
        self.assertFalse(self.receipt["scientific_validation"])
        self.assertFalse(self.receipt["causal_effect_estimation"])
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o700)
        data = rows(self.output / "features/features.csv")
        for offset in (6, 12):
            for first, copied in zip(data[:6], data[offset:offset + 6]):
                self.assertEqual([first[f] for f in walk.FEATURES], [copied[f] for f in walk.FEATURES])

    def test_altered_future_labels_do_not_change_predictions_or_fit_states(self):
        fixtures = copy_endpoints(self.root / "altered-fixtures")
        changed = rows(fixtures / "heldout_outcomes.csv")
        for row in changed:
            row["target"] = str(float(row["target"]) + 100)
        (fixtures / "heldout_outcomes.csv").write_bytes(walk.csv_bytes(("device_id", "target"), changed))
        output = self.root / "altered-label-run"
        walk.run(self.source, ROOT, output, invented_example_declared=True, fixture_dir=fixtures)
        self.assertEqual((output / "predictions.json").read_bytes(), (self.output / "predictions.json").read_bytes())
        self.assertNotEqual(body(output / "scores.json")["metrics"], body(self.output / "scores.json")["metrics"])

    def test_missing_heldout_file_first_fails_after_successful_prediction(self):
        fixtures = self.root / "no-outcomes"
        fixtures.mkdir()
        shutil.copyfile(HERE / "fixtures/training_endpoints.csv", fixtures / "training_endpoints.csv")
        output = self.root / "no-outcomes-run"
        with self.assertRaises(walk.WalkthroughError):
            walk.run(self.source, ROOT, output, invented_example_declared=True, fixture_dir=fixtures)
        self.assertEqual(body(output / "predictions.json")["status"], "COMPLETE_PREDICTIONS")
        self.assertFalse((output / "heldout_outcomes.csv").exists())
        self.assertFalse((output / "COMPLETE.json").exists())
        failed = json.loads((output / "FAILED.json").read_bytes())
        self.assertEqual(failed["stages"][-1]["stage"], "separate_exact_cohort_scoring")
        self.assertEqual(failed["stages"][-1]["status"], "command_rejected")

    def test_exact_training_endpoint_join_and_feature_identity(self):
        cohort = rows(self.output / "cohort.csv")
        original = rows(HERE / "fixtures/training_endpoints.csv")
        for index, changed in enumerate((original[:-1], original + [original[0]], original + [{"device_id": "unknown", "target": "1"}])):
            fixture = self.root / ("bad-train-" + str(index)); fixture.mkdir()
            (fixture / "training_endpoints.csv").write_bytes(walk.csv_bytes(("device_id", "target"), changed))
            output = self.root / ("bad-preparation-" + str(index)); output.mkdir()
            with self.assertRaises(walk.WalkthroughError):
                walk.prepare_prediction_tables(self.output / "features", cohort, fixture, output)
            self.assertEqual(list(output.iterdir()), [])
        altered_cohort = [dict(r) for r in cohort]
        altered_cohort[0]["wafer_id"] = "different"
        with self.assertRaises(walk.WalkthroughError):
            walk.prepare_prediction_tables(self.output / "features", altered_cohort, HERE / "fixtures", self.root)

    def test_endpoint_changed_after_score_cannot_be_complete(self):
        fixtures = copy_endpoints(self.root / "racing-fixtures")
        output = self.root / "racing-run"
        original_stage = walk.run_stage
        def replace_after_score(name, *args):
            original_stage(name, *args)
            if name == "separate_exact_cohort_scoring":
                path = fixtures / "heldout_outcomes.csv"
                path.write_bytes(path.read_bytes() + b"\n")
        with patch.object(walk, "run_stage", side_effect=replace_after_score):
            with self.assertRaises(walk.WalkthroughError):
                walk.run(self.source, ROOT, output, invented_example_declared=True, fixture_dir=fixtures)
        self.assertTrue((output / "scores.json").is_file())
        self.assertFalse((output / "COMPLETE.json").exists())
        self.assertFalse((output / "heldout_outcomes.csv").exists())


if __name__ == "__main__":
    unittest.main()
