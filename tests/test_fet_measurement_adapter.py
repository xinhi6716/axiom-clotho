"""Synthetic-only FET integration gates; no scientific accuracy claims."""
import copy
import csv
from decimal import Decimal
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlworkflow import contracts as c, regression as flow
from mlworkflow.adapters import FET_FEATURES, FET_FILES, FET_INVENTORIES, load_bundle
from mlworkflow.adapters import fet_transfer as fet
from mlworkflow.components import RESOURCE_ROOT, verify_components

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/fet-device-regression/fixtures"
RAW_COLUMNS = (*fet.ACQUISITION, "temperature_K", "drain_voltage", "body_voltage", "gate_voltage", "drain_current")


def rehash(path):
    receipt = c.parse_json((path / "COMPLETE.json").read_bytes())
    receipt["files_sha256"] = {n: c.sha((path / n).read_bytes()) for n in receipt["files_sha256"]}
    (path / "COMPLETE.json").write_bytes(c.json_bytes(receipt))


class FetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        shutil.copytree(FIXTURE, self.inputs)
        self.output = self.root / "prepared"

    def config(self, **changes):
        value = c.parse_json((self.inputs / "config.json").read_bytes())
        value.update(changes)
        (self.inputs / "config.json").write_bytes(c.json_bytes(value))
        return value

    def rows(self, name, columns):
        return c.parse_csv((self.inputs / name).read_bytes(), columns)

    def prepare(self, output=None):
        return flow.prepare(self.inputs / "raw", self.inputs / "partitions.csv", self.inputs / "training_targets.csv", self.inputs / "config.json", output or self.output)

    def test_complete_inventory_and_original_byte_snapshot(self):
        raw = (self.inputs / "raw/transfer.csv").read_bytes()
        result = self.prepare()
        self.assertEqual((result["training_rows"], result["prediction_rows"], result["ml_fits_performed"]), (12, 3, 0))
        payloads, _ = load_bundle(self.output, "prepared")
        self.assertEqual(set(payloads), FET_INVENTORIES["prepared"])
        self.assertEqual(payloads["features/transfer_input.csv"], raw)
        config, train, future, _, _ = fet.prepared(self.output)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(future), 3)
        self.assertNotIn("target", future[0])
        self.assertFalse(result["caller_prediction_outcomes_opened"])

    def test_exact_upstream_artifact_parity_and_only_two_feature_allowlist(self):
        config = self.config()
        self.prepare()
        extractor = flow._load_component("_test_fet_upstream", "semiconductor/examples/fet-measurement-to-features/build_features.py")
        direct = extractor.build(self.inputs / "raw/transfer.csv", **config["measurement"], assumptions_accepted=True)
        self.assertEqual(set(direct), set(FET_FILES))
        for name in direct:
            self.assertEqual((self.output / "features" / name).read_bytes(), direct[name], name)
        schema = c.parse_json(direct["feature_schema.json"])
        self.assertEqual(schema["model_feature_allowlist"], list(FET_FEATURES))
        self.assertEqual(schema["row_key"], list(fet.ACQUISITION))

    def test_negative_signed_current_and_decreasing_sweep(self):
        rows = self.rows("raw/transfer.csv", RAW_COLUMNS)
        groups = {}
        for row in rows:
            row["drain_current"] = str(-Decimal(row["drain_current"]))
            groups.setdefault(tuple(row[n] for n in fet.ACQUISITION), []).append(row)
        changed = [r for group in groups.values() for r in reversed(group)]
        (self.inputs / "raw/transfer.csv").write_bytes(c.csv_bytes(RAW_COLUMNS, changed))
        config = self.config()
        config["measurement"].update(current_polarity="negative", criterion_current="-50")
        self.config(measurement=config["measurement"])
        self.prepare()
        self.assertEqual(len(fet.prepared(self.output)[1]), 12)

    def test_explicit_voltage_current_unit_equivalence(self):
        self.prepare()
        first = (self.output / "training.csv").read_bytes()
        rows = self.rows("raw/transfer.csv", RAW_COLUMNS)
        for row in rows:
            for n in ("gate_voltage", "drain_voltage", "body_voltage"):
                row[n] = str(Decimal(row[n]) * 1000)
            row["drain_current"] = str(Decimal(row["drain_current"]) / 10 ** 9)
        (self.inputs / "raw/transfer.csv").write_bytes(c.csv_bytes(RAW_COLUMNS, rows))
        config = self.config()
        config["measurement"].update(voltage_unit="mV", current_unit="A", max_bracket_span="100", window_min="100", window_max="400")
        self.config(measurement=config["measurement"])
        self.prepare(self.root / "second")
        self.assertEqual((self.root / "second/training.csv").read_bytes(), first)

    def test_valid_zero_window_slope_is_retained_not_missing(self):
        rows = self.rows("raw/transfer.csv", RAW_COLUMNS)
        currents = {"0": "1", "0.1": "10", "0.2": "10", "0.3": "10", "0.4": "10", "0.5": "100", "0.6": "200"}
        for row in rows:
            row["drain_current"] = currents[str(Decimal(row["gate_voltage"]).normalize())]
        (self.inputs / "raw/transfer.csv").write_bytes(c.csv_bytes(RAW_COLUMNS, rows))
        self.prepare()
        train = fet.prepared(self.output)[1]
        self.assertTrue(all(c.number(r[FET_FEATURES[1]]) == 0 for r in train))

    def test_second_valid_branch_for_same_physical_device_rejected(self):
        rows = self.rows("raw/transfer.csv", RAW_COLUMNS)
        branch = [{**r, "branch": "second_valid_branch"} for r in rows[:7]]
        (self.inputs / "raw/transfer.csv").write_bytes(c.csv_bytes(RAW_COLUMNS, rows + branch))
        with self.assertRaisesRegex(c.WorkflowError, "duplicate composite"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_one_rejected_branch_is_audit_only_not_partial_ml(self):
        rows = self.rows("raw/transfer.csv", RAW_COLUMNS)
        for row in rows[:7]:
            row["drain_voltage"] = "0"
        (self.inputs / "raw/transfer.csv").write_bytes(c.csv_bytes(RAW_COLUMNS, rows))
        with patch("mlworkflow.regression._plans", side_effect=AssertionError("no ML planning")):
            with self.assertRaisesRegex(c.WorkflowError, "audit-only"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_exact_config_keys_types_and_no_metadata_predictors(self):
        base = self.config()
        changes = [{"schema_version": "measurement-regression-v1"}, {"adapter": "unknown"}, {"features": ["device"]},
                   {"features": []}, {"features": [FET_FEATURES[0]] * 2}, {"group_by": "batch"}, {"objective": "mae"},
                   {"outer_splits": True}, {"inner_splits": 1}, {"max_fits": 0}, {"baselines": ["ridge"]},
                   {"extra": "unknown"}, {"measurement": {}}, {"declarations": {**base["declarations"], "pre_outcome_features": 1}}]
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(c.WorkflowError):
                fet.validate_config({**base, **changed})
        for name, value in [("criterion_current", 50), ("criterion_current", "-50"), ("criterion_current", "NaN"),
                            ("max_bracket_span", "0"), ("window_min", "1"), ("current_unit", "density"), ("current_polarity", "auto")]:
            with self.subTest(name=name, value=value), self.assertRaises(c.WorkflowError):
                fet.validate_config({**base, "measurement": {**base["measurement"], name: value}})

    def test_every_grouping_level_uses_composite_prefixes(self):
        for level in ("lot", "wafer", "device"):
            self.config(group_by=level)
            self.prepare(self.root / level)
            rows = c.parse_csv((self.root / level / "identity_map.csv").read_bytes(), fet.MAP_COLUMNS)
            train = [r for r in rows if r["partition"] == "train"]
            self.assertEqual(len({r["group"] for r in train}), 12 if level == "device" else 4)
            self.assertEqual(len({r["device"] for r in train}), 3)

    def test_original_group_leakage_rejected_before_token_namespaces(self):
        rows = self.rows("partitions.csv", (*fet.PHYSICAL, "partition"))
        rows[0]["partition"] = "predict"
        (self.inputs / "partitions.csv").write_bytes(c.csv_bytes((*fet.PHYSICAL, "partition"), rows))
        with self.assertRaisesRegex(c.WorkflowError, "original composite physical groups"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_exact_composite_partition_and_target_join_no_future_labels(self):
        for name, columns in (("partitions.csv", (*fet.PHYSICAL, "partition")), ("training_targets.csv", (*fet.PHYSICAL, "target"))):
            original = (self.inputs / name).read_bytes()
            rows = c.parse_csv(original, columns)
            for changed in (rows[:-1], rows + [rows[0]], rows + [{**rows[0], "lot": "extra"}]):
                (self.inputs / name).write_bytes(c.csv_bytes(columns, changed))
                with self.subTest(name=name), self.assertRaises(c.WorkflowError):
                    self.prepare()
                self.assertFalse(self.output.exists())
            (self.inputs / name).write_bytes(original)
        rows = self.rows("training_targets.csv", (*fet.PHYSICAL, "target")) + self.rows("heldout_outcomes.csv", (*fet.PHYSICAL, "target"))
        (self.inputs / "training_targets.csv").write_bytes(c.csv_bytes((*fet.PHYSICAL, "target"), rows))
        with self.assertRaisesRegex(c.WorkflowError, "prediction labels are forbidden"):
            self.prepare()

    def test_future_identity_order_does_not_change_training_rows_or_splits(self):
        self.prepare()
        first_plan = flow.plan(self.output)
        first_train = (self.output / "training.csv").read_bytes()
        partitions = self.rows("partitions.csv", (*fet.PHYSICAL, "partition"))
        future_lots = {r["lot"] for r in partitions if r["partition"] == "predict"}
        rows = self.rows("raw/transfer.csv", RAW_COLUMNS)
        for row in rows:
            if row["lot"] in future_lots:
                row["lot"] = "000_future_before_train"
        for row in partitions:
            if row["partition"] == "predict":
                row["lot"] = "000_future_before_train"
        chunks = [rows[i:i + 7] for i in range(0, len(rows), 7)]
        (self.inputs / "raw/transfer.csv").write_bytes(c.csv_bytes(RAW_COLUMNS, [r for g in reversed(chunks) for r in g]))
        (self.inputs / "partitions.csv").write_bytes(c.csv_bytes((*fet.PHYSICAL, "partition"), list(reversed(partitions))))
        second = self.root / "second"
        self.prepare(second)
        self.assertEqual((second / "training.csv").read_bytes(), first_train)
        self.assertEqual(flow.plan(second)["comparison"], first_plan["comparison"])

    def test_semantic_map_remap_rejected_after_new_checksums(self):
        self.prepare()
        rows = c.parse_csv((self.output / "identity_map.csv").read_bytes(), fet.MAP_COLUMNS)
        rows[0]["row_id"], rows[1]["row_id"] = rows[1]["row_id"], rows[0]["row_id"]
        (self.output / "identity_map.csv").write_bytes(c.csv_bytes(fet.MAP_COLUMNS, rows))
        rehash(self.output)
        with self.assertRaisesRegex(c.WorkflowError, "identity map"):
            flow.plan(self.output)

    def test_corrupt_feature_identity_rejected_after_new_checksums(self):
        self.prepare()
        rows = c.parse_csv((self.output / "cohort.csv").read_bytes(), fet.COHORT_COLUMNS)
        rows[0]["device"] = "unrecorded_device"
        (self.output / "cohort.csv").write_bytes(c.csv_bytes(fet.COHORT_COLUMNS, rows))
        rehash(self.output)
        with self.assertRaisesRegex(c.WorkflowError, "cohort"):
            flow.plan(self.output)

    def test_plan_checks_typed_training_source_provenance(self):
        self.prepare()
        path = self.output / "provenance.json"
        original = c.parse_json(path.read_bytes())
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-m", "mlworkflow", "plan", "--prepared", str(self.output)]

        def plan():
            return subprocess.run(command, cwd=self.root, env=env, text=True,
                                  capture_output=True, timeout=15)

        result = plan()
        self.assertEqual(result.returncode, 0, result.stderr)
        for label, edit in (
            ("missing_source", lambda p: p["input_sha256"].pop("training_targets.csv")),
            ("malformed_digest", lambda p: p["input_sha256"].update({"config.json": "not-a-digest"})),
            ("unknown_role", lambda p: p["input_sha256"].update({"other.csv": "f" * 64})),
            ("wrong_count_type", lambda p: p.update(training_rows=float(p["training_rows"]))),
            ("false_fit_claim", lambda p: p.update(ml_fits_performed=7)),
            ("boolean_fit_count", lambda p: p.update(ml_fits_performed=False)),
        ):
            with self.subTest(label=label):
                changed = copy.deepcopy(original)
                edit(changed)
                path.write_bytes(c.json_bytes(changed))
                rehash(self.output)
                result = plan()
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("FET join provenance", result.stderr)
                self.assertFalse(result.stdout)

    def test_mixed_schema_or_adapter_receipt_cannot_switch_validation(self):
        self.prepare()
        path = self.output / "COMPLETE.json"
        raw = path.read_bytes()
        for field, value in (("schema_version", "measurement-workflow-bundle-v1"), ("adapter", "rram-v1")):
            receipt = c.parse_json(raw)
            receipt[field] = value
            path.write_bytes(c.json_bytes(receipt))
            with self.assertRaises(c.WorkflowError):
                flow.plan(self.output)
        path.write_bytes(raw)

    def test_plan_rejects_inconsistent_support_metadata_after_rehash(self):
        self.prepare()
        feature_path = self.output / "features/features.csv"
        provenance_path = self.output / "features/provenance.json"
        original = feature_path.read_bytes()
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-m", "mlworkflow", "plan", "--prepared", str(self.output)]

        def plan():
            return subprocess.run(command, cwd=self.root, env=env, text=True,
                                  capture_output=True, timeout=15)

        baseline = plan()
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        for name, wrong in (
            ("branch_sample_count", "1"),
            ("log_selected_sample_count", "1"),
            ("criterion_method", "invented_method"),
            ("criterion_bracket_span_V", "77"),
        ):
            with self.subTest(name=name):
                rows = c.parse_csv(original, fet.FEATURE_COLUMNS)
                self.assertNotEqual(rows[0][name], wrong)
                rows[0][name] = wrong
                featured = c.csv_bytes(fet.FEATURE_COLUMNS, rows)
                feature_path.write_bytes(featured)
                provenance = c.parse_json(provenance_path.read_bytes())
                provenance["output_sha256"]["features.csv"] = c.sha(featured)
                provenance_path.write_bytes(c.json_bytes(provenance))
                rehash(self.output)
                failed = plan()
                self.assertEqual(failed.returncode, 2, failed.stderr)
                self.assertIn("FET support metadata", failed.stderr)
                self.assertFalse(failed.stdout)

    def test_model_feature_range_not_silently_widened(self):
        self.prepare()
        payloads, _ = load_bundle(self.output, "prepared")
        features = {n: payloads["features/" + n] for n in FET_FILES}
        rows = c.parse_csv(features["features.csv"], fet.FEATURE_COLUMNS)
        rows[0][FET_FEATURES[0]] = "1e101"
        features["features.csv"] = c.csv_bytes(fet.FEATURE_COLUMNS, rows)
        with self.assertRaisesRegex(c.WorkflowError, "numeric magnitude"):
            fet._feature_rows(features, self.config())

    def test_fresh_output_protection_and_incomplete_marker(self):
        with self.assertRaisesRegex(c.WorkflowError, "outside"):
            self.prepare(self.inputs / "raw/prepared")
        original = c.write_new
        def fail(path, data):
            if Path(path).name == "future.csv":
                raise OSError("injected")
            return original(path, data)
        with patch.object(c, "write_new", side_effect=fail), self.assertRaises(OSError):
            self.prepare()
        self.assertTrue((self.output / "INCOMPLETE.json").exists())
        self.assertFalse((self.output / "COMPLETE.json").exists())
        with self.assertRaises(c.WorkflowError):
            self.prepare()

    def test_extra_file_and_symlink_in_saved_bundle_reject(self):
        self.prepare()
        extra = self.output / "unexpected.csv"
        extra.write_text("extra")
        with self.assertRaises(c.WorkflowError):
            flow.plan(self.output)
        extra.unlink()
        if hasattr(os, "symlink"):
            link = self.output / "alias"
            link.symlink_to(self.output / "config.json")
            with self.assertRaisesRegex(c.WorkflowError, "symbolic"):
                flow.plan(self.output)

    def test_plan_zero_ml_fits_and_budget_guard(self):
        self.config(max_fits=1)
        self.prepare()
        with patch("groupbench.benchmark", side_effect=AssertionError("ML fit")), patch("groupbench.fit_predict", side_effect=AssertionError("ML fit")):
            with self.assertRaisesRegex(c.WorkflowError, "max_fits"):
                flow.run(self.output, self.root / "run")
        self.assertFalse((self.root / "run").exists())

    def test_component_files_pinned_and_no_site_import_boundary(self):
        manifest = verify_components()
        semiconductor = next(c for c in manifest["components"] if c["component"] == "semiconductor")
        self.assertEqual(semiconductor["commit"], "4c0ec2ae3eaa4e0b7afc9aaf8f9bada544e7c0de")
        self.assertEqual(len(semiconductor["files"]), 16)
        code = "import sys; from mlworkflow import regression; from mlworkflow.adapters import fet_transfer; assert not any(n in sys.modules for n in ['numpy','pandas','sklearn','groupbench'])"
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=self.root, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_score_binds_engine_prediction_receipt_and_query_to_retained_run(self):
        self.prepare()
        run = self.root / "run"
        flow.run(self.output, run)
        engine = run / "prediction"
        manifest = engine / "manifest.json"
        checksums = engine / "checksums.json"
        manifest_raw, checksums_raw = manifest.read_bytes(), checksums.read_bytes()
        original = c.parse_json(manifest_raw)
        metadata_path = run / "run.json"
        metadata = c.parse_json(metadata_path.read_bytes())
        self.assertEqual(original["prediction_source_file_sha256"],
                         metadata["prepared_files_sha256"]["future.csv"])
        outcomes = self.inputs / "heldout_outcomes.csv"
        for name, edit_manifest, include_engine_checksum in (
            ("stale_engine_receipt", lambda m: m.update(comment="different bytes"), False),
            ("wrong_query", lambda m: m.update(prediction_source_file_sha256="f" * 64), True),
            ("missing_query", lambda m: m.pop("prediction_source_file_sha256"), True),
            ("wrong_count", lambda m: m.update(n_predictions=2), True),
            ("float_count", lambda m: m.update(n_predictions=3.0), True),
            ("wrong_mode", lambda m: m.update(mode="benchmark"), True),
            ("wrong_features", lambda m: m["config"].update(features=["other"]), True),
        ):
            with self.subTest(name=name):
                changed = copy.deepcopy(original)
                edit_manifest(changed)
                manifest.write_bytes(c.json_bytes(changed))
                receipt = c.parse_json(checksums_raw)
                if include_engine_checksum:
                    receipt["files"]["manifest.json"] = c.sha(manifest.read_bytes())
                checksums.write_bytes(c.json_bytes(receipt))
                rehash(run)
                before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
                output = self.root / name
                with patch.object(c, "read_regular", wraps=c.read_regular) as read, self.assertRaisesRegex(
                        c.WorkflowError, "saved FET engine prediction"):
                    flow.score(run, outcomes, output)
                self.assertNotIn(outcomes, [Path(call.args[0]) for call in read.call_args_list])
                self.assertFalse(output.exists())
                self.assertEqual(before, {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()})
        manifest.write_bytes(manifest_raw)
        checksums.write_bytes(checksums_raw)
        rehash(run)
        score = flow.score(run, outcomes, self.root / "valid-score")
        self.assertEqual((score["cohort_rows"], score["fits_performed"]), (3, 0))

    def test_no_site_prepare_and_saved_score_no_future_outcome_read(self):
        heldout = self.inputs / "heldout_outcomes.csv"
        outcomes = heldout.read_bytes()
        heldout.unlink()
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        command = [sys.executable, "-S", "-m", "mlworkflow", "prepare", "--input-dir", str(self.inputs / "raw"), "--partitions", str(self.inputs / "partitions.csv"), "--training-targets", str(self.inputs / "training_targets.csv"), "--config", str(self.inputs / "config.json"), "--output-dir", str(self.output)]
        result = subprocess.run(command, cwd=self.root, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        run = self.root / "run"
        planned = flow.plan(self.output)
        self.assertEqual(planned["fits_performed"], 0)
        flow.run(self.output, run)
        heldout.write_bytes(outcomes)
        # Score requires only the saved run, not original features/raw/prepared input.
        shutil.rmtree(self.output)
        code = "import sys; from mlworkflow import regression; result=regression.score(*sys.argv[1:]); assert result['fits_performed']==0; assert not any(n in sys.modules for n in ['numpy','pandas','sklearn','groupbench'])"
        result = subprocess.run([sys.executable, "-S", "-c", code, str(run), str(heldout), str(self.root / "score")], cwd=self.root, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        scores = c.parse_json((self.root / "score/scores.json").read_bytes())
        self.assertEqual(scores["schema_version"], "measurement-saved-score-v2")
        self.assertEqual(scores["cohort_rows"], 3)
        before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        changed = self.rows("heldout_outcomes.csv", (*fet.PHYSICAL, "target"))
        for r in changed:
            r["target"] = str(c.number(r["target"]) + 100)
        heldout.write_bytes(c.csv_bytes((*fet.PHYSICAL, "target"), changed))
        second = flow.score(run, heldout, self.root / "score2")
        self.assertNotEqual(scores["scorecard"], second["scorecard"])
        self.assertEqual(before, {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
