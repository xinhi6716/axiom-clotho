"""Real CSV contracts: complete joins, no leakage, bounded fitting and saved scoring."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlworkflow import regression as flow
from mlworkflow import contracts as c
from mlworkflow.components import RESOURCE_ROOT

FIXTURE = Path(__file__).resolve().parents[1] / "examples/measurement-regression/fixtures"


def tree(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


class MeasurementRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        shutil.copytree(FIXTURE, self.inputs)
        self.prepared, self.run_dir, self.score_dir = (self.root / n for n in ("prepared", "run", "score"))

    def config(self, **updates):
        path = self.inputs / "config.json"
        config = json.loads(path.read_bytes())
        config.update(updates)
        path.write_bytes(c.json_bytes(config))
        return config

    def rows(self, name, columns):
        return c.parse_csv((self.inputs / name).read_bytes(), columns)

    def prepare(self):
        return flow.prepare(self.inputs / "raw", self.inputs / "partitions.csv", self.inputs / "training_targets.csv", self.inputs / "config.json", self.prepared)

    def command(self, *args, no_site=False):
        env = dict(os.environ, PYTHONPATH=str(RESOURCE_ROOT.parents[1]), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        return subprocess.run([sys.executable, *(["-S"] if no_site else []), "-m", "mlworkflow", *map(str, args)],
                              env=env, cwd=self.root, text=True, capture_output=True, timeout=90)

    def test_complete_cli_reproduction_without_second_checkout_or_early_outcomes(self):
        outcomes = self.inputs / "later_outcomes.csv"
        saved = outcomes.read_bytes()
        outcomes.unlink()
        result = self.command("prepare", "--input-dir", self.inputs / "raw", "--partitions", self.inputs / "partitions.csv", "--training-targets", self.inputs / "training_targets.csv", "--config", self.inputs / "config.json", "--output-dir", self.prepared, no_site=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = tree(self.root)
        planned = self.command("plan", "--prepared", self.prepared)
        self.assertEqual(planned.returncode, 0, planned.stderr)
        self.assertEqual(tree(self.root), before)
        self.assertEqual(json.loads(planned.stdout)["fits_performed"], 0)
        self.assertEqual(json.loads(planned.stdout)["total_planned_fits"], 40)
        result = self.command("run", "--prepared", self.prepared, "--output-dir", self.run_dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("target", (self.prepared / "future.csv").read_text().splitlines()[0])
        before = tree(self.run_dir)
        outcomes.write_bytes(saved)
        result = self.command("score", "--run-dir", self.run_dir, "--outcomes", outcomes, "--output-dir", self.score_dir, no_site=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(tree(self.run_dir), before)
        for path, kind in ((self.prepared, "prepared"), (self.run_dir, "run"), (self.score_dir, "score")):
            payloads, digest = c.load_bundle(path, kind)
            self.assertEqual(set(payloads), c.INVENTORIES[kind])
            self.assertEqual(len(digest), 64)
        report = json.loads((self.score_dir / "scores.json").read_bytes())
        self.assertEqual((report["cohort_rows"], report["fits_performed"]), (6, 0))
        self.assertIn("Axiom Clotho", (self.run_dir / "report.html").read_text())

    def test_stdlib_import_prepare_and_score_do_not_import_ml(self):
        code = "import sys; from mlworkflow import regression,contracts; assert not any(n in sys.modules for n in ['groupbench','numpy','pandas','sklearn'])"
        result = subprocess.run([sys.executable, "-S", "-c", code], env=dict(os.environ, PYTHONPATH=str(RESOURCE_ROOT.parents[1])), cwd=self.root, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_version_identity_and_core_delegation_preserve_compatibility(self):
        from groupbench import __version__ as core_version
        from mlworkflow import __version__ as workflow_version
        self.assertEqual(core_version, workflow_version)
        self.assertEqual(core_version, "0.3.0b7")
        result = self.command("--version", no_site=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Axiom Clotho 0.3.0b7", result.stdout)
        result = self.command("tabular", "--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0.3.0b7", result.stdout)

    def test_exact_target_and_partition_joins_reject_missing_extra_duplicate(self):
        for name, columns in (("partitions.csv", ("device", "partition")), ("training_targets.csv", ("device", "target"))):
            original = (self.inputs / name).read_bytes()
            rows = c.parse_csv(original, columns)
            for modified in (rows[:-1], rows + [rows[0]], rows + [{**rows[0], "device": "UNKNOWN_DEVICE"}]):
                with self.subTest(name=name, size=len(modified)):
                    (self.inputs / name).write_bytes(c.csv_bytes(columns, modified))
                    with self.assertRaises(c.WorkflowError):
                        self.prepare()
                    self.assertFalse(self.prepared.exists())
            (self.inputs / name).write_bytes(original)

    def test_future_targets_cannot_enter_training_file(self):
        train = self.rows("training_targets.csv", ("device", "target"))
        future = self.rows("later_outcomes.csv", ("device", "target"))
        (self.inputs / "training_targets.csv").write_bytes(c.csv_bytes(("device", "target"), train + future))
        with self.assertRaisesRegex(c.WorkflowError, "prediction-device labels are forbidden"):
            self.prepare()
        self.assertFalse(self.prepared.exists())

    def test_group_leakage_and_unknown_partition_fail_before_writes(self):
        rows = self.rows("partitions.csv", ("device", "partition"))
        for partition in ("predict", "test", "TRAIN"):
            changed = copy.deepcopy(rows)
            changed[0]["partition"] = partition
            (self.inputs / "partitions.csv").write_bytes(c.csv_bytes(("device", "partition"), changed))
            with self.assertRaises(c.WorkflowError):
                self.prepare()
            self.assertFalse(self.prepared.exists())

    def test_explicit_grouping_choices_and_leading_zero_ids_are_preserved(self):
        for group_by in ("batch", "wafer", "device"):
            with self.subTest(group_by=group_by):
                self.config(group_by=group_by)
                self.prepare()
                rows = c.parse_csv((self.prepared / "cohort.csv").read_bytes(), flow.COHORT_COLUMNS)
                self.assertTrue(all(r["group"] == r[group_by] for r in rows))
                shutil.rmtree(self.prepared)
        self.assertEqual(c.identifier("001"), "001")
        self.assertNotEqual(c.identifier("001"), c.identifier("1"))

    def test_custom_endpoint_values_and_selected_features_are_not_fixed_fixture_contract(self):
        self.config(endpoint_name="Caller-selected endpoint", target_unit="caller_units", features=[c.FEATURES[-1]])
        targets = self.rows("training_targets.csv", ("device", "target"))
        for i, row in enumerate(targets):
            row["target"] = str(100 + i)
        (self.inputs / "training_targets.csv").write_bytes(c.csv_bytes(("device", "target"), targets))
        self.prepare()
        training = c.parse_csv((self.prepared / "training.csv").read_bytes(), ("device", "group", "target", c.FEATURES[-1]))
        self.assertEqual({r["device"]: r["target"] for r in training}, {r["device"]: r["target"] for r in targets})

    def test_config_exact_keys_types_declarations_and_metadata_feature_rejected(self):
        original = json.loads((self.inputs / "config.json").read_bytes())
        mutations = [dict(features=["device"]), dict(features=[]), dict(features=[c.FEATURES[0]] * 2),
                     dict(baselines=["ridge"]), dict(baselines=["dummy", "neural"]), dict(group_by="automatic"),
                     dict(outer_splits=True), dict(inner_splits=1), dict(max_fits=0), dict(seed=-1),
                     dict(max_bracket_span_V=0.11), dict(declarations={**original["declarations"], "pre_outcome_features": 1}),
                     dict(extra="ignored"), dict(target_unit=" PRIVATE\nVALUE"), dict(endpoint_name="\ud800")]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(c.WorkflowError):
                c.validate_config({**original, **change})
        with self.assertRaises(c.WorkflowError):
            c.parse_json(b'{"x":1,"x":2}')

    def test_bad_numbers_headers_encoding_and_special_files_fail_safely(self):
        for value in ("NaN", "inf", "1_000", "1e101", "1e-101", "1 ohm", "", "1" * 65):
            with self.subTest(value=value), self.assertRaises(c.WorkflowError):
                c.number(value)
        for raw in (b"device,device\na,b\n", b"device,target\na,1,extra\n", b"device,target\n", b"\xff", b'device,target\n"unterminated'):
            with self.assertRaises(c.WorkflowError):
                c.parse_csv(raw, ("device", "target"))
        if hasattr(os, "mkfifo"):
            fifo = self.root / "input.fifo"
            os.mkfifo(fifo)
            with self.assertRaises(c.WorkflowError):
                c.read_regular(fifo)

    def test_no_fit_plan_budget_guard_and_insufficient_groups(self):
        self.config(max_fits=1)
        self.prepare()
        with patch("groupbench.benchmark", side_effect=AssertionError("unexpected fit")), patch("groupbench.fit_predict", side_effect=AssertionError("unexpected fit")):
            with self.assertRaisesRegex(c.WorkflowError, "max_fits"):
                flow.run(self.prepared, self.run_dir)
        self.assertFalse(self.run_dir.exists())
        shutil.rmtree(self.prepared)
        self.config(max_fits=200, outer_splits=5)
        with self.assertRaisesRegex(c.WorkflowError, "too few training groups"):
            self.prepare()

    def test_input_snapshots_and_supplied_data_unchanged(self):
        before = tree(self.inputs)
        self.prepare()
        self.assertEqual(tree(self.inputs), before)
        provenance = json.loads((self.prepared / "provenance.json").read_bytes())
        for name in flow.RAW_FILES:
            self.assertEqual(provenance["input_sha256"][name], hashlib.sha256(before["raw/" + name]).hexdigest())

    def test_existing_output_symlink_missing_parent_and_cli_privacy(self):
        self.prepared.mkdir()
        (self.prepared / "keep").write_text("preserve")
        with self.assertRaises(c.WorkflowError):
            self.prepare()
        self.assertEqual((self.prepared / "keep").read_text(), "preserve")
        for output in (self.root / "absent/child", self.prepared):
            with self.assertRaises(c.WorkflowError):
                c.check_fresh_output(output)
        if hasattr(os, "symlink"):
            link = self.root / "link"
            link.symlink_to(self.root / "nonexistent")
            with self.assertRaises(c.WorkflowError):
                c.check_fresh_output(link)
        result = self.command("prepare", "--private-option", "SECRET_VALUE", no_site=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("SECRET_VALUE", result.stderr)
        self.assertNotIn(str(self.root), result.stderr)

    def test_corrupted_incomplete_bundle_rejected_before_ml(self):
        self.prepare()
        path = self.prepared / "future.csv"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(c.WorkflowError, "bytes changed"):
            flow.plan(self.prepared)
        path.write_bytes(original)
        (self.prepared / "INCOMPLETE.json").write_text("{}")
        with self.assertRaisesRegex(c.WorkflowError, "incomplete"):
            flow.plan(self.prepared)

    def test_nested_output_and_resolved_alias_refused_before_fitting(self):
        self.prepare()
        before = tree(self.prepared)
        candidates = [self.prepared / "run"]
        if hasattr(os, "symlink"):
            link = self.root / "alias"
            link.symlink_to(self.prepared, target_is_directory=True)
            candidates.append(link / "run")
        with patch.object(flow, "_plans", side_effect=AssertionError("should reject before plan/fit")):
            for output in candidates:
                with self.assertRaisesRegex(c.WorkflowError, "outside"):
                    flow.run(self.prepared, output)
        self.assertEqual(tree(self.prepared), before)
        with self.assertRaisesRegex(c.WorkflowError, "outside"):
            flow.score(self.prepared, self.inputs / "later_outcomes.csv", self.prepared / "score")
        with self.assertRaisesRegex(c.WorkflowError, "outside"):
            flow.prepare(self.inputs / "raw", self.inputs / "partitions.csv", self.inputs / "training_targets.csv", self.inputs / "config.json", self.inputs / "raw/prepared")

    def test_engine_prediction_count_and_numeric_range_fail_before_output(self):
        import pandas as pd
        self.prepare()
        for values in ([1.0] * 5, [1.0] * 7, [1e101] * 6):
            result = SimpleNamespace(predictions=pd.DataFrame({"prediction": values}))
            with patch("groupbench.benchmark", return_value=None), patch("groupbench.fit_predict", return_value=result):
                with self.assertRaises(c.WorkflowError):
                    flow.run(self.prepared, self.run_dir)
            self.assertFalse(self.run_dir.exists())

    def test_write_failure_keeps_incomplete_marker_and_never_complete(self):
        real = c.write_new
        def fail(path, data):
            if Path(path).name == "future.csv":
                raise OSError("injected failure")
            return real(path, data)
        with patch.object(c, "write_new", side_effect=fail), self.assertRaises(OSError):
            self.prepare()
        self.assertTrue((self.prepared / "INCOMPLETE.json").is_file())
        self.assertFalse((self.prepared / "COMPLETE.json").exists())
        with self.assertRaises(c.WorkflowError):
            c.load_bundle(self.prepared, "prepared")

    def test_altered_later_outcomes_change_only_saved_scores_and_use_original_metric_engine(self):
        self.prepare()
        flow.run(self.prepared, self.run_dir)
        before = tree(self.run_dir)
        first = flow.score(self.run_dir, self.inputs / "later_outcomes.csv", self.score_dir)
        rows = self.rows("later_outcomes.csv", ("device", "target"))
        for row in rows:
            row["target"] = str(c.number(row["target"]) + 100)
        (self.inputs / "later_outcomes.csv").write_bytes(c.csv_bytes(("device", "target"), rows))
        second = flow.score(self.run_dir, self.inputs / "later_outcomes.csv", self.root / "score2")
        self.assertNotEqual(first["scorecard"], second["scorecard"])
        self.assertEqual(tree(self.run_dir), before)
        scored = c.parse_csv((self.score_dir / "scored_rows.csv").read_bytes(), ("id", "group", "y_true", "y_pred"))
        spec = importlib.util.spec_from_file_location("direct_scorecard", RESOURCE_ROOT / "scorecard/tools/prediction-scorecard/prediction_scorecard.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        direct = module.score([{**r, "y_true": float(r["y_true"]), "y_pred": float(r["y_pred"])} for r in scored], grouped=True)
        self.assertEqual(first["scorecard"], direct)

    def test_saved_score_exact_cohort_and_inconsistent_mapping_fail_without_fitting(self):
        self.prepare()
        flow.run(self.prepared, self.run_dir)
        original = self.rows("later_outcomes.csv", ("device", "target"))
        for rows in (original[:-1], original + [original[0]], original + [{"device": "EXTRA", "target": "1"}]):
            (self.inputs / "later_outcomes.csv").write_bytes(c.csv_bytes(("device", "target"), rows))
            with self.assertRaises(c.WorkflowError):
                flow.score(self.run_dir, self.inputs / "later_outcomes.csv", self.score_dir)
            self.assertFalse(self.score_dir.exists())
        # Even updating the outer byte receipt cannot make inconsistent row mapping valid.
        mapping = c.parse_csv((self.run_dir / "prediction_rows.csv").read_bytes(), ("device", "group", "prediction"))
        mapping[0]["prediction"] = "999"
        (self.run_dir / "prediction_rows.csv").write_bytes(c.csv_bytes(("device", "group", "prediction"), mapping))
        receipt = json.loads((self.run_dir / "COMPLETE.json").read_bytes())
        receipt["files_sha256"]["prediction_rows.csv"] = c.sha((self.run_dir / "prediction_rows.csv").read_bytes())
        (self.run_dir / "COMPLETE.json").write_bytes(c.json_bytes(receipt))
        with self.assertRaisesRegex(c.WorkflowError, "differ from"):
            flow.score(self.run_dir, self.inputs / "later_outcomes.csv", self.score_dir)

    def test_report_escapes_endpoint_markup_and_has_no_external_assets(self):
        config = self.config(endpoint_name="<script>alert(1)</script>")
        report = flow._report("<unsafe>", config, ["<script>bad</script>"], [("scores.json", "<metrics>")]).decode()
        self.assertNotIn("<script>", report)
        self.assertIn("&lt;script&gt;", report)
        self.assertNotIn("https://", report)
        self.assertIn("default-src 'none'", report)


if __name__ == "__main__":
    unittest.main()
