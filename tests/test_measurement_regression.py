"""Real CSV contracts: complete joins, no leakage, bounded fitting and saved scoring."""
import copy
import csv
from decimal import Inexact, ROUND_DOWN, ROUND_UP, localcontext
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

    def test_prepare_retains_the_config_bytes_used_for_v1_dispatch(self):
        path = self.inputs / "config.json"
        initial = path.read_bytes()
        changed = c.json_bytes({**json.loads(initial), "endpoint_name": "changed-endpoint"})
        reads = []

        def read_once_or_changed(candidate):
            if Path(candidate) == path:
                reads.append(candidate)
                return initial if len(reads) == 1 else changed
            return c.read_regular(candidate)

        with patch.object(flow, "read_regular", side_effect=read_once_or_changed):
            prepared = self.prepare()
        self.assertEqual(len(reads), 1)
        self.assertEqual(prepared["input_sha256"]["config.json"], hashlib.sha256(initial).hexdigest())
        self.assertEqual(json.loads((self.prepared / "config.json").read_bytes()), json.loads(initial))

    def test_run_rejects_reordered_calculator_rows_before_bundle_publication(self):
        from groupbench import fit_predict

        self.prepare()

        def reverse_result(*args, **kwargs):
            result = fit_predict(*args, **kwargs)
            result.predictions.sort_values("row_position", ascending=False, inplace=True)
            return result

        with patch("groupbench.fit_predict", side_effect=reverse_result):
            with self.assertRaisesRegex(c.WorkflowError, "prediction row positions"):
                flow.run(self.prepared, self.run_dir)
        self.assertFalse(self.run_dir.exists())

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

    def test_prepare_rejects_exact_target_beyond_magnitude_even_if_binary64_rounds_to_bound(self):
        path = self.inputs / "training_targets.csv"
        original = path.read_bytes()
        columns = ("device", "target")
        for token in ("1.000000000000000000000000000001e100", "-9.999999999999999999999999999999e-101"):
            if self.prepared.exists():
                shutil.rmtree(self.prepared)
            rows = c.parse_csv(original, columns)
            rows[0]["target"] = token
            path.write_bytes(c.csv_bytes(columns, rows))
            with self.subTest(token=token):
                with self.assertRaisesRegex(c.WorkflowError, "numeric magnitude"):
                    self.prepare()
                self.assertFalse(self.prepared.exists())

    def test_measurement_number_admission_is_independent_of_decimal_context(self):
        rejected = (
            "1.000000000000000000000000000001e100", "-1.000000000000000000000000000001e100",
            "9.999999999999999999999999999999e-101", "-9.999999999999999999999999999999e-101",
        )
        accepted = ("-0.34", "1e100", "-1e100", "1e-100", "-1e-100", "-0")
        for rounding in (ROUND_DOWN, ROUND_UP):
            with self.subTest(rounding=rounding), localcontext() as context:
                context.prec = 1
                context.Emin, context.Emax = -1, 1
                context.rounding = rounding
                context.traps[Inexact] = True
                for token in accepted:
                    with self.subTest(token=token):
                        self.assertEqual(c.number(token).hex(), float(token).hex())
                for token in rejected:
                    with self.subTest(token=token), self.assertRaisesRegex(c.WorkflowError, "numeric magnitude"):
                        c.number(token)

    def test_replay_requires_canonical_device_order_not_only_exact_membership(self):
        configuration = self.config()
        self.prepare()
        reference = tree(self.prepared)
        self.assertEqual(flow.plan(self.prepared)["fits_performed"], 0)
        columns = {
            "cohort.csv": flow.COHORT_COLUMNS,
            "training.csv": ("device", "group", "target", *configuration["features"]),
            "future.csv": ("device", "group", *configuration["features"]),
        }
        for names in (("training.csv",), ("future.csv",), tuple(columns)):
            with self.subTest(names=names):
                for name, data in reference.items():
                    (self.prepared / name).write_bytes(data)
                for name in names:
                    rows = c.parse_csv(reference[name], columns[name])
                    (self.prepared / name).write_bytes(c.csv_bytes(columns[name], rows[::-1]))
                path = self.prepared / "COMPLETE.json"
                receipt = c.parse_json(reference["COMPLETE.json"])
                receipt["files_sha256"] = {n: c.sha((self.prepared / n).read_bytes())
                                           for n in receipt["files_sha256"]}
                path.write_bytes(c.json_bytes(receipt))
                c.load_bundle(self.prepared, "prepared")
                saved = tree(self.prepared)
                with patch.object(flow, "begin_bundle", side_effect=AssertionError("unexpected publication")):
                    with self.assertRaisesRegex(c.WorkflowError, "canonical.*order"):
                        flow.plan(self.prepared)
                    with self.assertRaisesRegex(c.WorkflowError, "canonical.*order"):
                        flow.run(self.prepared, self.run_dir)
                self.assertEqual(tree(self.prepared), saved)
                self.assertFalse(self.run_dir.exists())

    def test_cli_plan_rejects_conflicting_feature_source_hashes(self):
        self.prepare()
        baseline = self.command("plan", "--prepared", self.prepared)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        path = self.prepared / "features/provenance.json"
        provenance = c.parse_json(path.read_bytes())
        provenance["inputs_sha256"]["manifest.csv"] = "0" * 64
        path.write_bytes(c.json_bytes(provenance))
        receipt_path = self.prepared / "COMPLETE.json"
        receipt = c.parse_json(receipt_path.read_bytes())
        receipt["files_sha256"]["features/provenance.json"] = c.sha(path.read_bytes())
        receipt_path.write_bytes(c.json_bytes(receipt))
        result = self.command("plan", "--prepared", self.prepared)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("feature source declarations", result.stderr)
        self.assertFalse(result.stdout)

    def test_cli_revalidates_embedded_feature_contract_after_receipt_check(self):
        self.prepare()
        baseline = self.command("plan", "--prepared", self.prepared)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        relative = "features/feature_schema.json"
        path = self.prepared / relative
        receipt_path = self.prepared / "COMPLETE.json"
        good = c.parse_json(path.read_bytes())
        receipt = c.parse_json(receipt_path.read_bytes())
        cases = (
            ("feature_units", {**good["feature_units"], c.FEATURES[0]: "A"}),
            ("schema_version", True),
            ("row_key", ["wafer"]),
            ("target_columns", ["target"]),
            ("model_feature_allowlist", list(reversed(c.FEATURES))),
        )
        for field, value in cases:
            with self.subTest(field=field):
                raw = c.json_bytes({**good, field: value})
                path.write_bytes(raw)
                receipt["files_sha256"][relative] = c.sha(raw)
                receipt_path.write_bytes(c.json_bytes(receipt))
                c.load_bundle(self.prepared, "prepared")
                saved = tree(self.prepared)
                result = self.command("plan", "--prepared", self.prepared)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("measurement feature schema", result.stderr)
                self.assertFalse(result.stdout)
                self.assertEqual(tree(self.prepared), saved)

    def test_cli_rejects_saved_readpoint_policy_inconsistent_with_preparation(self):
        # A bound reported as 1/10 V but prepared under 0.11 V is not
        # evidence of having applied the tighter acquisition filter.
        self.prepare()
        good = tree(self.prepared)
        baseline = self.command("plan", "--prepared", self.prepared)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        provenance = "features/provenance.json"
        report = "features/iv_readpoints.json"
        cases = (
            ("config.json", None, "max_bracket_span_V", "0.10"),
            (provenance, "parameters", "max_bracket_span_V_input", "0.110000000000000001"),
            (provenance, "parameters", "input_voltage_unit", "mV"),
            (provenance, "parameters", "input_current_unit", "mA"),
            (provenance, "parameters", "measurement_assumptions_declared", 1),
            (report, None, "max_bracket_span_V", 0.10),
            (report, None, "schema_version", True),
        )
        for relative, outer, field, value in cases:
            with self.subTest(relative=relative, field=field):
                for name, data in good.items():
                    (self.prepared / name).write_bytes(data)
                payload = c.parse_json(good[relative])
                (payload[outer] if outer else payload)[field] = value
                raw = c.json_bytes(payload)
                (self.prepared / relative).write_bytes(raw)
                receipt = c.parse_json(good["COMPLETE.json"])
                receipt["files_sha256"][relative] = c.sha(raw)
                (self.prepared / "COMPLETE.json").write_bytes(c.json_bytes(receipt))
                c.load_bundle(self.prepared, "prepared")
                saved = tree(self.prepared)
                for command in (("plan",), ("run", "--output-dir", self.run_dir)):
                    result = self.command(command[0], "--prepared", self.prepared, *command[1:])
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("saved RRAM readpoint policy", result.stderr)
                    self.assertFalse(result.stdout)
                self.assertFalse(self.run_dir.exists())
                self.assertEqual(tree(self.prepared), saved)

    def test_replay_rejects_contradictory_preparation_provenance_before_work(self):
        self.prepare()
        receipt_path = self.prepared / "COMPLETE.json"
        provenance_path = self.prepared / "provenance.json"
        original = provenance_path.read_bytes()
        receipt = c.parse_json(receipt_path.read_bytes())
        self.assertEqual(flow.plan(self.prepared)["fits_performed"], 0)
        for field, value in (
            ("caller_prediction_outcomes_opened", True),
            ("caller_prediction_outcomes_opened", 0),
            ("training_rows", 1),
            ("training_rows", float(c.parse_json(original)["training_rows"])),
            ("prediction_rows", 1),
            ("feature_row_unit", "measurement-cycle"),
            ("endpoint_join", "unbound source rows"),
            ("input_sha256", {"config.json": "1" * 64}),
            ("schema_version", "measurement-join-v2"),
        ):
            with self.subTest(field=field, value=value):
                altered = c.parse_json(original)
                altered[field] = value
                raw = c.json_bytes(altered)
                provenance_path.write_bytes(raw)
                receipt["files_sha256"]["provenance.json"] = c.sha(raw)
                receipt_path.write_bytes(c.json_bytes(receipt))
                c.load_bundle(self.prepared, "prepared")
                before = tree(self.prepared)
                with patch("groupbench.plan_benchmark", side_effect=AssertionError("planning must not start")), patch(
                    "groupbench.benchmark", side_effect=AssertionError("fitting must not start")
                ):
                    with self.assertRaisesRegex(c.WorkflowError, "preparation provenance"):
                        flow.plan(self.prepared)
                    with self.assertRaisesRegex(c.WorkflowError, "preparation provenance"):
                        flow.run(self.prepared, self.run_dir)
                self.assertFalse(self.run_dir.exists())
                self.assertEqual(tree(self.prepared), before)

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

    def test_saved_score_consumes_verified_keyed_prediction_values(self):
        self.prepare()
        flow.run(self.prepared, self.run_dir)
        with patch.object(flow, "number", wraps=c.number) as parse_number:
            result = flow.score(self.run_dir, self.inputs / "later_outcomes.csv", self.score_dir)
        # Three numeric fields per scored device (engine prediction, keyed
        # prediction, outcome). Config validation uses contracts.number separately.
        self.assertEqual(parse_number.call_count, 3 * result["cohort_rows"])

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
