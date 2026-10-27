"""Main-CLI discovery and delegation for the existing saved-interval evaluator."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from groupbench import cli, interval_evaluation


def fixture():
    return {
        "schema_version": 1,
        "metadata": {
            "target_definition": "invented response",
            "value_unit": "arbitrary test units",
            "group_definition": "invented unit",
            "cohort_definition": "all three invented rows",
        },
        "expected_row_ids": ["a", "b", "c"],
        "alpha": 0.2,
        "rows": [
            {"row_id": "a", "group": "unit-A", "observed": 0,
             "lower": 0, "upper": 2, "status": "ok"},
            {"row_id": "b", "group": "unit-A", "observed": 3,
             "lower": 0, "upper": 2, "status": "ok"},
            {"row_id": "c", "group": "unit-B", "observed": 10,
             "lower": 0, "upper": 0, "status": "ok"},
        ],
    }


def invoke(function, args):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = function(args)
        except SystemExit as exc:
            code = exc.code
    return code, stdout.getvalue(), stderr.getvalue()


class SavedIntervalMainCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / "saved intervals.json"
        self.input.write_text(json.dumps(fixture()), encoding="utf-8")

    def both(self, args):
        direct = invoke(interval_evaluation.main, args)
        alias = invoke(cli.main, ["evaluate-intervals", *args])
        self.assertEqual(alias, direct)
        return alias

    def test_main_help_discovers_operation(self):
        code, stdout, stderr = invoke(cli.main, ["--help"])
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("evaluate-intervals", stdout)
        self.assertIn("no fitting or calibration", stdout)

    def test_help_and_argument_rejections_are_identical(self):
        for args, expected in [(["--help"], 0), ([], 2),
                               ([str(self.input), "extra"], 2),
                               ([str(self.input), "--unknown"], 2),
                               (["--out", str(self.root / "report.json")], 2)]:
            with self.subTest(args=args):
                code, stdout, stderr = self.both(args)
                self.assertEqual(code, expected)
                self.assertIn("mlcalc evaluate-intervals", stdout if code == 0 else stderr)
                if code:
                    self.assertEqual(stdout, "")
        self.assertFalse((self.root / "report.json").exists())

    def test_valid_result_and_null_alpha_use_existing_evaluator(self):
        for alpha in (0.2, None):
            with self.subTest(alpha=alpha):
                document = fixture()
                document["alpha"] = alpha
                self.input.write_text(json.dumps(document), encoding="utf-8")
                code, stdout, stderr = self.both([str(self.input)])
                self.assertEqual((code, stderr), (0, ""))
                parsed = interval_evaluation.load_document(self.input.read_text(encoding="utf-8"))
                self.assertEqual(json.loads(stdout), interval_evaluation.evaluate_saved_intervals(parsed))

    def test_data_and_file_errors_are_not_swallowed(self):
        reversed_bounds = fixture()
        reversed_bounds["rows"][0]["lower"] = 3
        missing_row = fixture()
        missing_row["rows"].pop()
        unknown_key = fixture()
        unknown_key["unexpected"] = True
        inputs = ["not json", '{"schema_version":1,"schema_version":1}',
                  json.dumps(reversed_bounds), json.dumps(missing_row), json.dumps(unknown_key)]
        for document in inputs:
            with self.subTest(document=document):
                self.input.write_text(document, encoding="utf-8")
                code, stdout, stderr = self.both([str(self.input)])
                self.assertEqual((code, stdout), (2, ""))
                self.assertIn("interval evaluation rejected:", stderr)
        for path in (self.root / "missing.json", self.root):
            code, stdout, stderr = self.both([str(path)])
            self.assertEqual((code, stdout), (2, ""))
            self.assertIn("interval evaluation rejected:", stderr)

    def test_delegates_raw_remaining_arguments_once(self):
        args = ["--", str(self.input)]
        with patch.object(interval_evaluation, "main", return_value=37) as delegated:
            self.assertEqual(cli.main(["evaluate-intervals", *args]), 37)
        delegated.assert_called_once_with(args)

    def test_no_fit_plan_or_existing_file_write(self):
        sentinel = self.root / "report.html"
        sentinel.write_bytes(b"existing report must remain untouched")
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        with patch.object(cli, "benchmark", side_effect=AssertionError("no fitting")), \
             patch.object(cli, "fit_predict", side_effect=AssertionError("no fitting")), \
             patch.object(cli, "plan_benchmark", side_effect=AssertionError("no training plan")), \
             patch.object(cli, "plan_prediction", side_effect=AssertionError("no training plan")):
            self.assertEqual(self.both([str(self.input)])[0], 0)
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, before)

    def test_sys_argv_route_does_not_mutate_process_arguments(self):
        args = ["mlcalc", "evaluate-intervals", str(self.input)]
        with patch.object(sys, "argv", args):
            actual = invoke(cli.main, None)
            self.assertEqual(sys.argv, args)
        self.assertEqual(actual, invoke(interval_evaluation.main, [str(self.input)]))

    def test_actual_module_entrypoints_match_outside_source_cwd(self):
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                   PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        for args in ([str(self.input)], ["--help"], [], [str(self.input), "--out", "no.json"]):
            with self.subTest(args=args):
                direct = subprocess.run([sys.executable, "-m", "groupbench.interval_evaluation", *args],
                                        cwd=self.root, env=env, capture_output=True, timeout=20)
                alias = subprocess.run([sys.executable, "-m", "groupbench", "evaluate-intervals", *args],
                                       cwd=self.root, env=env, capture_output=True, timeout=20)
                self.assertEqual((alias.returncode, alias.stdout, alias.stderr),
                                 (direct.returncode, direct.stdout, direct.stderr))
        self.assertFalse((self.root / "no.json").exists())


if __name__ == "__main__":
    unittest.main()
