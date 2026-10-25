"""Source-companion software tests; invented fixtures are not causal evidence."""
import copy
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "examples/environment-heldout-comparison"

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

probe = load("environment_probe_companion", "probe.py")
generator = load("environment_probe_fixtures", "generate_synthetic.py")


def rows(blob):
    return list(csv.DictReader(io.StringIO(blob.decode())))


def data(records):
    return generator.csv_bytes(list(records[0]), records)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = generator.fixtures()
        self.spec = json.loads(self.fixture["spec.json"])

    def predict(self, train=None, future=None, spec=None):
        return probe.predict(self.fixture["training.csv"] if train is None else train,
                             self.fixture["future.csv"] if future is None else future,
                             self.fixture["spec.json"] if spec is None else json.dumps(spec).encode(), contract_declared=True)

    def score(self, prediction=None, outcomes=None):
        return probe.score(probe.report_bytes(self.predict() if prediction is None else prediction),
                           self.fixture["outcomes.csv"] if outcomes is None else outcomes)

    def rejected_before_fit(self, train=None, future=None, spec=None):
        with patch.object(probe, "make_pipeline") as model, self.assertRaises(probe.InputError):
            self.predict(train, future, spec)
        model.assert_not_called()

    def test_fixture_bytes_and_exact_five_fits(self):
        for name, blob in self.fixture.items():
            self.assertEqual(blob, (HERE / "fixtures" / name).read_bytes())
        with patch.object(probe, "make_pipeline", wraps=probe.make_pipeline) as build:
            result = self.predict()["body"]
        self.assertEqual(build.call_count, 5)
        self.assertEqual(result["fit_contract"]["completed_arms"], 5)
        self.assertEqual(result["status"], "COMPLETE_PREDICTIONS")
        self.assertEqual(self.predict(), self.predict())

    def test_analytic_single_feature_ridge_and_dummy(self):
        body = self.predict()["body"]
        for record, source in zip(body["predictions"], rows(self.fixture["future.csv"])):
            x = float(source["preprocess_setting"])
            self.assertAlmostEqual(record["predictions"]["dummy"], 3.9)
            self.assertAlmostEqual(record["predictions"]["b"], 3.9 + .2 * (16 / 17) * (x - 4.5), places=11)

    def test_two_feature_normal_equation_oracle(self):
        train, future = rows(self.fixture["training.csv"]), rows(self.fixture["future.csv"])
        features = self.spec["features_a"]
        x = np.array([[float(r[f]) for f in features] for r in train])
        y = np.array([float(r["target"]) for r in train])
        z = (x - x.mean(axis=0)) / x.std(axis=0)
        beta = np.linalg.solve(z.T @ z + np.eye(2), z.T @ (y - y.mean()))
        expected = ((np.array([[float(r[f]) for f in features] for r in future]) - x.mean(axis=0)) / x.std(axis=0)) @ beta + y.mean()
        actual = [r["predictions"]["a"] for r in self.predict()["body"]["predictions"]]
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    def test_unknown_future_labels_never_accepted_for_fit(self):
        future = rows(self.fixture["future.csv"])
        for row in future:
            row["target"] = "123"
        self.rejected_before_fit(future=data(future))

    def test_future_values_change_no_fitted_state_or_other_rows(self):
        original = self.predict()["body"]
        future = rows(self.fixture["future.csv"])
        future[0]["preprocess_setting"] = "1000000"
        future[0]["proxy_readout"] = ""
        changed = self.predict(future=data(future))["body"]
        self.assertEqual(original["arms"], changed["arms"])
        self.assertEqual(original["shuffle"], changed["shuffle"])
        self.assertEqual(original["predictions"][1:], changed["predictions"][1:])
        new = dict(future[0], device_id="new_device", dependency_block="new_block")
        extended = self.predict(future=data(future + [new]))["body"]
        self.assertEqual(changed["arms"], extended["arms"])
        self.assertEqual(changed["predictions"], extended["predictions"][:-1])

    def test_scoring_has_no_fit_and_does_not_mutate_prediction(self):
        prediction = self.predict()
        before = probe.report_bytes(prediction)
        outcomes = rows(self.fixture["outcomes.csv"])
        for row in outcomes:
            row["target"] = str(float(row["target"]) + 100)
        with patch.object(probe, "make_pipeline") as fit:
            altered = self.score(prediction, data(outcomes))
        fit.assert_not_called()
        self.assertEqual(probe.report_bytes(prediction), before)
        self.assertNotEqual(altered["body"]["metrics"], self.score(prediction)["body"]["metrics"])

    def test_outcome_exact_key_join_not_order_or_subset(self):
        prediction = self.predict()
        out = rows(self.fixture["outcomes.csv"])
        self.assertEqual(self.score(prediction)["body"]["rows"], self.score(prediction, data(list(reversed(out))))["body"]["rows"])
        for bad in (out[:-1], out + [out[0]], out + [dict(out[0], device_id="extra")]):
            with self.assertRaises(probe.InputError):
                self.score(prediction, data(bad))

    def test_shared_shuffle_mapping_and_multisets(self):
        train, _, _ = probe.prepare(self.fixture["training.csv"], self.fixture["future.csv"], self.spec)
        shuffled, evidence = probe.shuffle_labels(train, 7)
        by_id = {r["device_id"]: r for r in train}
        for env in {r["environment"] for r in train}:
            indices = [i for i, r in enumerate(train) if r["environment"] == env]
            self.assertEqual(sorted(shuffled[i] for i in indices), sorted(train[i]["target"] for i in indices))
        for pair in evidence["mapping"]:
            self.assertEqual(by_id[pair["recipient_device_id"]]["environment"], by_id[pair["donor_device_id"]]["environment"])
        labels = []
        real = probe.make_pipeline
        def builder(*args, **kwargs):
            model = real(*args, **kwargs)
            original = model.fit
            def spy(X, y):
                labels.append(np.array(y).copy())
                return original(X, y)
            model.fit = spy
            return model
        with patch.object(probe, "make_pipeline", side_effect=builder):
            self.predict()
        np.testing.assert_array_equal(labels[3], labels[4])
        np.testing.assert_array_equal(labels[0], labels[1])
        np.testing.assert_array_equal(labels[1], labels[2])

    def test_identity_shuffle_reported_never_redrawn(self):
        train = rows(self.fixture["training.csv"])
        small = train[:2] + train[8:10]
        found = None
        for seed in range(30):
            spec = dict(self.spec, shuffle_seed=seed)
            result = self.predict(train=data(small), spec=spec)["body"]
            if all(r["identity_mapping"] for r in result["shuffle"]["environment_results"]):
                found = result
                break
        self.assertIsNotNone(found)
        self.assertFalse(found["shuffle"]["redrawn"])
        self.assertTrue(all(r["all_target_values_unchanged"] for r in found["shuffle"]["environment_results"]))
        for row in found["predictions"]:
            self.assertEqual(row["predictions"]["a"], row["predictions"]["shuffle_a"])
            self.assertEqual(row["predictions"]["b"], row["predictions"]["shuffle_b"])

    def test_arm_order_reversal_preserves_corresponding_models(self):
        original = self.predict()["body"]
        flipped = dict(self.spec, features_a=self.spec["features_b"], features_b=self.spec["features_a"])
        changed = self.predict(spec=flipped)["body"]
        self.assertEqual(original["shuffle"], changed["shuffle"])
        for a, b in zip(original["predictions"], changed["predictions"]):
            self.assertEqual(a["predictions"]["a"], b["predictions"]["b"])
            self.assertEqual(a["predictions"]["shuffle_a"], b["predictions"]["shuffle_b"])

    def test_duplicate_and_cross_boundary_device_and_block(self):
        train, future = rows(self.fixture["training.csv"]), rows(self.fixture["future.csv"])
        self.rejected_before_fit(train=data(train + [train[0]]))
        for field in ("device_id", "dependency_block"):
            changed = copy.deepcopy(future)
            changed[0][field] = train[0][field]
            self.rejected_before_fit(future=data(changed))

    def test_environment_boundary_and_tiny_or_constant_training(self):
        train, future = rows(self.fixture["training.csv"]), rows(self.fixture["future.csv"])
        bad = copy.deepcopy(train)
        bad[0]["environment"] = self.spec["heldout_environment"]
        self.rejected_before_fit(train=data(bad))
        bad = copy.deepcopy(future)
        bad[0]["environment"] = "unrequested"
        self.rejected_before_fit(future=data(bad))
        self.rejected_before_fit(train=data(train[:8]))
        self.rejected_before_fit(train=data(train[:1] + train[8:]))
        bad = copy.deepcopy(train)
        for row in bad[:8]:
            row["target"] = "1"
        self.rejected_before_fit(train=data(bad))

    def test_missing_feature_train_medians_only_and_all_missing_rejected(self):
        train = rows(self.fixture["training.csv"])
        train[0]["preprocess_setting"] = ""
        report = self.predict(train=data(train))["body"]
        self.assertEqual(report["status"], "COMPLETE_PREDICTIONS")
        for row in train:
            row["preprocess_setting"] = ""
        self.rejected_before_fit(train=data(train))

    def test_bounded_numeric_schema_and_forbidden_roles(self):
        for value in ("NaN", "inf", "1e101", "1e-999", " 1", "1 "):
            future = rows(self.fixture["future.csv"])
            future[0]["preprocess_setting"] = value
            self.rejected_before_fit(future=data(future))
        for feature in (*probe.META, "target"):
            self.rejected_before_fit(spec=dict(self.spec, features_a=[feature]))
        for alpha in (0, -1, True, 10 ** 1000, float("inf")):
            self.rejected_before_fit(spec=dict(self.spec, ridge_alpha=alpha))
        for seed in (True, -1, 2 ** 32, 1.0):
            self.rejected_before_fit(spec=dict(self.spec, shuffle_seed=seed))
        self.rejected_before_fit(spec=dict(self.spec, features_b=self.spec["features_a"]))

    def test_tiny_target_and_feature_scale_rejected_before_fit(self):
        train = rows(self.fixture["training.csv"])
        for row in train:
            row["target"] = format(float(row["target"]) * 1e-200, ".17g")
        self.rejected_before_fit(train=data(train))
        for name in ("preprocess_setting", "proxy_readout"):
            future = rows(self.fixture["future.csv"])
            future[0][name] = "1e-101"
            self.rejected_before_fit(future=data(future))
        self.assertEqual(probe.finite_token("1e-100"), 1e-100)
        self.assertEqual(probe.finite_token("-1e100"), -1e100)
        self.assertEqual(probe.finite_token("0"), 0)

    def test_scaled_rmse_never_false_zero_for_tiny_errors(self):
        self.assertEqual(probe.stable_rmse([1e-200, 1e-200]), 1e-200)
        self.assertEqual(probe.stable_rmse([0., 0.]), 0.)
        prediction = self.predict()
        for row in prediction["body"]["predictions"]:
            row["predictions"] = {arm: 1e-200 for arm in probe.ARMS}
        prediction = probe.seal(prediction["body"])
        outcomes = rows(self.fixture["outcomes.csv"])
        for row in outcomes:
            row["target"] = "0"
        scored = self.score(prediction, data(outcomes))["body"]
        self.assertTrue(all(metric["rmse"] == 1e-200 for metric in scored["metrics"].values()))

    def test_unrepresentable_nonzero_aggregate_not_reported_as_zero(self):
        prediction = self.predict()
        for index, row in enumerate(prediction["body"]["predictions"]):
            row["predictions"] = {arm: 5e-324 if index == 0 else 0.0 for arm in probe.ARMS}
        prediction = probe.seal(prediction["body"])
        outcomes = rows(self.fixture["outcomes.csv"])
        for row in outcomes:
            row["target"] = "0"
        with self.assertRaises(probe.InputError):
            self.score(prediction, data(outcomes))

    def test_exact_text_device_identity(self):
        future = rows(self.fixture["future.csv"])
        future[0]["device_id"], future[1]["device_id"] = "001", "1"
        body = self.predict(future=data(future))["body"]
        self.assertEqual([r["device_id"] for r in body["predictions"]][:2], ["001", "1"])

    def test_failure_record_retains_whole_cohort_and_refuses_scoring(self):
        count = 0
        original = probe.make_pipeline
        def fail_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise ValueError("private detail never echo")
            return original(*args, **kwargs)
        with patch.object(probe, "make_pipeline", side_effect=fail_second):
            result = self.predict()
        self.assertEqual(count, 5)
        self.assertEqual(result["body"]["exit_code"], 1)
        self.assertEqual(result["body"]["coverage"]["future_devices"], 4)
        self.assertEqual(result["body"]["coverage"]["failed_arms"], ["a"])
        self.assertTrue(all(r["predictions"]["a"] is None for r in result["body"]["predictions"]))
        self.assertNotIn("private detail", probe.report_bytes(result).decode())
        with self.assertRaises(probe.InputError):
            self.score(result)

    def test_artifact_integrity_and_rehashed_invalid_cohorts(self):
        original = self.predict()
        changed = copy.deepcopy(original)
        changed["body"]["predictions"][0]["predictions"]["a"] = 0
        with self.assertRaises(probe.InputError):
            self.score(changed)
        mutations = [lambda b: b["predictions"].pop(),
                     lambda b: b["predictions"][0]["predictions"].pop("shuffle_a"),
                     lambda b: b["predictions"][0]["predictions"].update(a=True),
                     lambda b: b["predictions"][0].update(environment="wrong"),
                     lambda b: b["training_membership"][0].update(dependency_block=b["predictions"][0]["dependency_block"]),
                     lambda b: b["fit_contract"].update(completed_arms=4),
                     lambda b: b["coverage"].update(future_devices=True),
                     lambda b: b["shuffle"]["mapping"][0].update(donor_device_id="missing")]
        for mutate in mutations:
            body = copy.deepcopy(original["body"])
            mutate(body)
            with self.assertRaises(probe.InputError):
                self.score(probe.seal(body))

    def test_seeded_mapping_cannot_be_replaced_by_another_bijection(self):
        prediction = self.predict()
        body = copy.deepcopy(prediction["body"])
        for item in body["shuffle"]["mapping"]:
            item["donor_device_id"] = item["recipient_device_id"]
        for item in body["shuffle"]["environment_results"]:
            item.update(changed_device_assignments=0, changed_target_values=0,
                        identity_mapping=True, all_target_values_unchanged=True)
        with self.assertRaises(probe.InputError):
            self.score(probe.seal(body))

    def test_target_scale_linearity_at_declared_range_boundaries(self):
        ordinary = self.predict()["body"]["predictions"]
        for factor in (1e-100, 1e99):
            train = rows(self.fixture["training.csv"])
            for row in train:
                row["target"] = format(float(row["target"]) * factor, ".17g")
            scaled = self.predict(train=data(train))["body"]["predictions"]
            for expected, actual in zip(ordinary, scaled):
                for arm in probe.ARMS:
                    self.assertAlmostEqual(actual["predictions"][arm] / factor,
                                           expected["predictions"][arm], places=10)

    def test_scoring_hand_metrics_and_units_preserved(self):
        score = self.score()["body"]
        self.assertEqual(score["target_unit"], "V")
        self.assertEqual(score["coverage"]["scored_devices"], 4)
        self.assertAlmostEqual(score["metrics"]["dummy"]["mae"], .4)
        self.assertAlmostEqual(score["metrics"]["dummy"]["rmse"], math.sqrt(.2))
        self.assertAlmostEqual(score["paired_mean_absolute_error_b_minus_a"], score["metrics"]["b"]["mae"] - score["metrics"]["a"]["mae"])
        self.assertIn("not a randomized intervention", " ".join(score["limitations"]))

    def test_literal_declaration_and_caps(self):
        for value in (1, "yes", None, False):
            with self.assertRaises(probe.InputError):
                probe.predict(self.fixture["training.csv"], self.fixture["future.csv"], self.fixture["spec.json"], contract_declared=value)
        with patch.object(probe, "MAX_BYTES", 1), self.assertRaises(probe.InputError):
            self.predict()
        with patch.object(probe, "MAX_ROWS", 1), self.assertRaises(probe.InputError):
            self.predict()
        with patch.object(probe, "MAX_REPORT_BYTES", 1), self.assertRaises(probe.InputError):
            self.predict()
        self.assertEqual(probe.MAX_BYTES, probe.MAX_REPORT_BYTES)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.f = generator.fixtures()
        for n, b in self.f.items():
            (self.root / n).write_bytes(b)

    def command(self, output):
        return [sys.executable, str(HERE / "probe.py"), "predict", str(self.root / "training.csv"), str(self.root / "future.csv"),
                "--spec", str(self.root / "spec.json"), "--accept-heldout-contract", "--output", str(output)]

    def run_cli(self, args):
        return subprocess.run(args, capture_output=True, timeout=20)

    def test_predict_then_score_files_private(self):
        dest = self.root / "prediction.json"
        result = self.run_cli(self.command(dest))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(stat.S_IMODE(dest.stat().st_mode), 0o600)
        scoring = self.root / "score.json"
        result = self.run_cli([sys.executable, str(HERE / "probe.py"), "score", str(dest), str(self.root / "outcomes.csv"), "--output", str(scoring)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(scoring.read_bytes())["body"]["coverage"]["scored_devices"], 4)

    def test_no_overwrite_input_symlinks_directories(self):
        target = self.root / "existing"
        target.write_bytes(b"keep")
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "absent")
        live = self.root / "live"
        live.symlink_to(target)
        for path in (target, dangling, live, self.root, self.root / "training.csv"):
            result = self.run_cli(self.command(path))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, b"")
        self.assertEqual(target.read_bytes(), b"keep")
        self.assertEqual((self.root / "training.csv").read_bytes(), self.f["training.csv"])

    def test_invalid_redacted_no_output(self):
        (self.root / "future.csv").write_text("PRIVATE_ID,PRIVATE_VALUE\n")
        dest = self.root / "out.json"
        result = self.run_cli(self.command(dest))
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(b"PRIVATE", result.stderr)
        self.assertNotIn(str(self.root).encode(), result.stderr)
        self.assertFalse(dest.exists())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX")
    def test_fifo_does_not_block(self):
        (self.root / "future.csv").unlink()
        os.mkfifo(self.root / "future.csv")
        result = self.run_cli(self.command(self.root / "out.json"))
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
