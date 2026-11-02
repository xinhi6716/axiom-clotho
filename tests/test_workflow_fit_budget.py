"""Opt-in count inspection never fits, writes, or relaxes normal budget guards."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
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

from mlworkflow import cli, contracts as c, regression as flow

ROOT = Path(__file__).resolve().parents[1]
SOURCE = str(Path(cli.__file__).resolve().parents[1])
FIXTURES = {"rram": ROOT / "examples/measurement-regression/fixtures",
            "fet": ROOT / "examples/fet-device-regression/fixtures"}


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@contextmanager
def no_execution():
    with ExitStack() as stack:
        for target in ("groupbench.model.make_pipeline", "groupbench.model.Pipeline.fit",
                       "groupbench.model.GroupBenchRegressor._fit_prepared",
                       "groupbench.benchmark", "groupbench.fit_predict",
                       "groupbench.report.write_payloads", "mlworkflow.regression.begin_bundle",
                       "mlworkflow.regression.write_new"):
            stack.enter_context(patch(target, side_effect=AssertionError("unexpected execution: " + target)))
        yield


class WorkflowFitBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.counter = 0

    def prepare(self, workflow, **updates):
        self.counter += 1
        case = self.root / (workflow + str(self.counter))
        inputs, prepared = case / "inputs", case / "prepared"
        shutil.copytree(FIXTURES[workflow], inputs)
        config = c.parse_json((inputs / "config.json").read_bytes())
        config.update(updates)
        (inputs / "config.json").write_bytes(c.json_bytes(config))
        flow.prepare(inputs / "raw", inputs / "partitions.csv", inputs / "training_targets.csv",
                     inputs / "config.json", prepared)
        return prepared

    def invoke(self, prepared, *extra):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(["plan", "--prepared", str(prepared), *extra])
        return code, out.getvalue(), err.getvalue()

    def assert_summary(self, summary, plan, within):
        self.assertEqual(set(summary), {"schema_version", "unit", "fits_performed", "comparison_fits",
                         "prediction_fits", "total_planned_fits", "max_fits", "within_budget", "limitations"})
        self.assertEqual(summary["schema_version"], "measurement-fit-budget-summary-v1")
        self.assertEqual(summary["unit"], "pipeline.fit calls")
        self.assertEqual(summary["fits_performed"], 0)
        self.assertEqual(summary["comparison_fits"], plan["comparison"]["fit_budget"]["total_fits"])
        self.assertEqual(summary["prediction_fits"], plan["prediction"]["fit_budget"]["total_fits"])
        self.assertEqual(summary["total_planned_fits"], summary["comparison_fits"] + summary["prediction_fits"])
        self.assertEqual(summary["total_planned_fits"], plan["total_planned_fits"])
        self.assertEqual(summary["max_fits"], plan["max_fits"])
        self.assertIs(summary["within_budget"], within)
        for name in ("fits_performed", "comparison_fits", "prediction_fits", "total_planned_fits", "max_fits"):
            self.assertIs(type(summary[name]), int)
        limitations = " ".join(summary["limitations"])
        for text in ("trees", "CPU time", "memory", "max_fits", "scientific"):
            self.assertIn(text, limitations)
        json.dumps(summary, allow_nan=False)

    def test_both_workflows_match_exact_plans_without_writes_or_fits(self):
        for workflow in FIXTURES:
            with self.subTest(workflow=workflow):
                prepared = self.prepare(workflow)
                before = snapshot(self.root)
                with no_execution():
                    plan = flow.plan(prepared)
                    summary = flow.budget_summary(prepared)
                    code, out, err = self.invoke(prepared, "--budget-summary")
                self.assert_summary(summary, plan, True)
                self.assertEqual((code, err), (0, ""))
                self.assertEqual(json.loads(out), summary)
                self.assertEqual(snapshot(self.root), before)
                # Only aggregate counts and static copy, not IDs, endpoints or paths.
                for value in (str(prepared), "device", "target_unit", "features", "prepared_receipt_sha256"):
                    self.assertNotIn(value, out)

    def test_over_budget_inspection_preserves_cap_and_strict_default_plan_and_run(self):
        for workflow in FIXTURES:
            with self.subTest(workflow=workflow):
                prepared = self.prepare(workflow, max_fits=1)
                before = snapshot(self.root)
                with no_execution():
                    summary = flow.budget_summary(prepared)
                    code, out, err = self.invoke(prepared, "--budget-summary")
                    self.assertEqual((code, err), (2, ""))
                    self.assertEqual(json.loads(out), summary)
                    self.assertIs(summary["within_budget"], False)
                    self.assertEqual(summary["max_fits"], 1)
                    self.assertGreater(summary["total_planned_fits"], 1)
                    self.assertEqual(summary["fits_performed"], 0)
                    with self.assertRaisesRegex(c.WorkflowError, "exceed.*max_fits"):
                        flow.plan(prepared)
                    with self.assertRaisesRegex(c.WorkflowError, "exceed.*max_fits"):
                        flow.run(prepared, self.root / "forbidden-run")
                    code, out, err = self.invoke(prepared)
                    self.assertEqual((code, out), (2, ""))
                    self.assertEqual(err, "error: planned pipeline fits exceed the explicitly declared max_fits; no fitting was performed\n")
                self.assertEqual(snapshot(self.root), before)
                self.assertFalse((self.root / "forbidden-run").exists())

    def test_exact_cap_is_within_budget_for_both_workflows(self):
        for workflow in FIXTURES:
            with self.subTest(workflow=workflow):
                reference = self.prepare(workflow)
                with no_execution():
                    total = flow.plan(reference)["total_planned_fits"]
                prepared = self.prepare(workflow, max_fits=total)
                with no_execution():
                    plan = flow.plan(prepared)
                    code, out, err = self.invoke(prepared, "--budget-summary")
                self.assertEqual((code, err), (0, ""))
                self.assert_summary(json.loads(out), plan, True)

    def test_candidate_grid_changes_are_read_from_real_schedules_not_hardcoded(self):
        from groupbench.model import Candidate, candidates_for
        prepared = self.prepare("rram")
        with no_execution():
            baseline = flow.budget_summary(prepared)

            def expanded(families):
                return candidates_for(families) + [Candidate("invented_extra_ridge", "ridge", {"alpha": 2.5})]

            with patch("groupbench.model.candidates_for", side_effect=expanded):
                plan = flow.plan(prepared)
                changed = flow.budget_summary(prepared)
        self.assert_summary(changed, plan, True)
        self.assertGreater(changed["comparison_fits"], baseline["comparison_fits"])
        self.assertGreater(changed["prediction_fits"], baseline["prediction_fits"])

    def test_alternative_families_use_the_existing_schedule_accounting(self):
        for families in (["dummy", "extra_trees"], ["ridge", "dummy", "extra_trees"]):
            prepared = self.prepare("rram", baselines=families)
            with no_execution():
                plan = flow.plan(prepared)
                summary = flow.budget_summary(prepared)
            self.assert_summary(summary, plan, True)

    def test_default_plan_stdout_is_unchanged_after_opt_in_inspection(self):
        prepared = self.prepare("rram")
        with no_execution():
            expected = json.dumps(flow.plan(prepared), indent=2, sort_keys=True, allow_nan=False) + "\n"
            first = self.invoke(prepared)
            self.invoke(prepared, "--budget-summary")
            second = self.invoke(prepared)
        self.assertEqual(first, (0, expected, ""))
        self.assertEqual(second, first)
        self.assertNotIn("within_budget", first[1])

    def test_infeasible_grouped_schedule_does_not_become_a_budget_summary(self):
        prepared = self.prepare("rram", inner_splits=3, max_fits=1)
        before = snapshot(self.root)
        with no_execution():
            code, out, err = self.invoke(prepared, "--budget-summary")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("grouped plan is infeasible", err)
        self.assertEqual(snapshot(self.root), before)

    def test_changed_bundle_and_invalid_budget_still_fail_without_json(self):
        for workflow in FIXTURES:
            prepared = self.prepare(workflow)
            path = prepared / "config.json"
            config = c.parse_json(path.read_bytes())
            config["max_fits"] = True
            path.write_bytes(c.json_bytes(config))
            # First fail changed bytes, then valid checksums but invalid configuration.
            for rehash in (False, True):
                if rehash:
                    receipt_path = prepared / "COMPLETE.json"
                    receipt = c.parse_json(receipt_path.read_bytes())
                    receipt["files_sha256"]["config.json"] = c.sha(path.read_bytes())
                    receipt_path.write_bytes(c.json_bytes(receipt))
                before = snapshot(self.root)
                with no_execution():
                    code, out, err = self.invoke(prepared, "--budget-summary")
                self.assertEqual((code, out), (2, ""))
                self.assertIn("error:", err)
                self.assertEqual(snapshot(self.root), before)

    def test_summary_flag_is_not_an_execution_override(self):
        with patch.object(flow, "run", side_effect=AssertionError("run called")), \
                redirect_stderr(io.StringIO()) as err, redirect_stdout(io.StringIO()) as out:
            code = cli.main(["run", "--prepared", "PRIVATE_INPUT", "--output-dir", "PRIVATE_OUTPUT", "--budget-summary"])
        self.assertEqual((code, out.getvalue()), (2, ""))
        self.assertIn("clotho run --help", err.getvalue())
        self.assertNotIn("PRIVATE", err.getvalue())

    def test_module_cli_returns_complete_over_budget_json_for_both_workflows(self):
        for workflow in FIXTURES:
            prepared = self.prepare(workflow, max_fits=1)
            before = snapshot(self.root)
            result = subprocess.run(
                [sys.executable, "-m", "mlworkflow", "plan", "--prepared", str(prepared), "--budget-summary"],
                cwd=self.root, env=dict(os.environ, PYTHONPATH=SOURCE, PYTHONDONTWRITEBYTECODE="1",
                                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual((result.returncode, result.stderr), (2, ""))
            summary = json.loads(result.stdout)
            self.assertIs(summary["within_budget"], False)
            self.assertEqual((summary["max_fits"], summary["fits_performed"]), (1, 0))
            self.assertEqual(summary["total_planned_fits"], summary["comparison_fits"] + summary["prediction_fits"])
            self.assertEqual(snapshot(self.root), before)

    def test_help_is_stdlib_only_and_explains_opt_in_exit_status(self):
        script = """import sys
from mlworkflow.cli import main
try:
    main(['plan', '--help'])
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('groupbench', 'numpy', 'pandas', 'sklearn'))
"""
        result = subprocess.run([sys.executable, "-S", "-c", script], cwd=self.root,
            env=dict(os.environ, PYTHONPATH=SOURCE, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, timeout=10)
        self.assertEqual((result.returncode, result.stderr), (0, ""))
        self.assertIn("--budget-summary", result.stdout)
        self.assertIn("exit 2", " ".join(result.stdout.split()))
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
