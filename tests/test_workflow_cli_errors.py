"""Workflow syntax errors stay actionable without echoing private arguments."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlworkflow import cli


class WorkflowArgumentGuidanceTests(unittest.TestCase):
    def invoke(self, arguments):
        out, err = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            operations = [stack.enter_context(patch(target)) for target in (
                "mlworkflow.cli.verify_components", "mlworkflow.cli.subprocess.run",
                "mlworkflow.cli._snapshot_calculator", "mlworkflow.scaffold.initialize",
                "mlworkflow.regression.prepare", "mlworkflow.regression.plan",
                "mlworkflow.regression.run", "mlworkflow.regression.score",
            )]
            stack.enter_context(redirect_stdout(out))
            stack.enter_context(redirect_stderr(err))
            code = cli.main(arguments)
            for operation in operations:
                operation.assert_not_called()
        return code, out.getvalue(), err.getvalue()

    def test_missing_arguments_show_correct_command_and_required_options(self):
        cases = {
            "init": ("--workflow {rram,fet}", "--out OUT"),
            "prepare": ("--input-dir INPUT_DIR", "--partitions PARTITIONS",
                        "--training-targets TRAINING_TARGETS", "--config CONFIG",
                        "--output-dir OUTPUT_DIR"),
            "plan": ("--prepared PREPARED",),
            "run": ("--prepared PREPARED", "--output-dir OUTPUT_DIR"),
            "score": ("--run-dir RUN_DIR", "--outcomes OUTCOMES", "--output-dir OUTPUT_DIR"),
            "demo": ("--output-dir OUTPUT_DIR", "--run-invented-example"),
        }
        for command, required in cases.items():
            with self.subTest(command=command):
                code, out, err = self.invoke([command])
                self.assertEqual((code, out), (2, ""))
                self.assertIn("usage: clotho " + command, err)
                self.assertIn("clotho " + command + " --help", err)
                for option in required:
                    self.assertIn(option, " ".join(err.split()))
                self.assertNotIn("complete distribution", err)
                self.assertNotIn("Traceback", err)

    def test_unknown_options_and_invalid_choices_never_echo_supplied_values(self):
        secret = "PRIVATE_TOKEN_123"
        cases = [
            [secret],
            ["init", "--workflow", secret, "--out", "/PRIVATE_PATH/output"],
            ["init", "--workflow", "rram", "--out", "/PRIVATE_PATH/output", "--" + secret],
            ["plan", "--prepared", "/PRIVATE_PATH/bundle", "--" + secret, secret],
            ["run", "--prepared", "/PRIVATE_PATH/bundle", "--output-dir"],
            ["demo", "--output-dir", "/PRIVATE_PATH/output"],
            ["score", "--run-dir", "PRIVATE\nINJECTED\x1b[31m", "--outcomes"],
        ]
        for arguments in cases:
            with self.subTest(command=arguments[0]):
                code, out, err = self.invoke(arguments)
                self.assertEqual((code, out), (2, ""))
                self.assertIn("usage: clotho", err)
                self.assertIn("--help", err)
                for token in (secret, "PRIVATE_PATH", "PRIVATE", "INJECTED", "\x1b", "Traceback"):
                    self.assertNotIn(token, err)

    def test_executable_path_is_not_exposed_in_root_error_or_help(self):
        with patch.object(sys, "argv", ["/PRIVATE_EXECUTABLE/PRIVATE_LAUNCHER"]):
            code, out, err = self.invoke(["--PRIVATE_OPTION"])
            self.assertEqual((code, out), (2, ""))
            self.assertIn("usage: clotho", err)
            self.assertNotIn("PRIVATE", err)
            with redirect_stdout(io.StringIO()) as help_output:
                with self.assertRaises(SystemExit) as raised:
                    cli.main(["--help"])
            self.assertEqual(raised.exception.code, 0)
            self.assertIn("usage: clotho", help_output.getvalue())
            self.assertNotIn("PRIVATE", help_output.getvalue())

    def test_resource_failures_remain_distinct_and_redacted(self):
        for failure in (cli.ComponentError("PRIVATE_RESOURCE"), OSError("PRIVATE_PATH")):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(cli, "verify_components", side_effect=failure), \
                        redirect_stderr(io.StringIO()) as err, redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(cli.main(["components"]), 2)
                self.assertEqual(out.getvalue(), "")
                self.assertIn("complete distribution", err.getvalue())
                self.assertNotIn("usage:", err.getvalue())
                self.assertNotIn("PRIVATE", err.getvalue())

    def test_errors_are_stdlib_only_and_leave_working_directory_untouched(self):
        source = str(Path(cli.__file__).resolve().parents[1])
        script = """import sys
from mlworkflow.cli import main
assert main(sys.argv[1:]) == 2
assert not any(name in sys.modules for name in ('groupbench', 'numpy', 'pandas', 'sklearn'))
"""
        with tempfile.TemporaryDirectory() as directory:
            for command in ("init", "prepare", "plan", "run", "score", "demo", "PRIVATE_COMMAND"):
                with self.subTest(command=command):
                    result = subprocess.run(
                        [sys.executable, "-S", "-c", script, command], cwd=directory,
                        env=dict(os.environ, PYTHONPATH=source, PYTHONDONTWRITEBYTECODE="1"),
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("usage: clotho", result.stderr)
                    self.assertNotIn("PRIVATE_COMMAND", result.stderr)
                    self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
