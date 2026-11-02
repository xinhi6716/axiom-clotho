"""Demo output guidance is redacted, read-only, and not a filesystem reservation."""
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


class DemoOutputPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def arguments(self, output):
        return ["demo", "--output-dir", str(output), "--run-invented-example"]

    def reject_before_work(self, output, expected):
        out, err = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            operations = [stack.enter_context(patch(target)) for target in (
                "mlworkflow.cli.verify_components", "mlworkflow.cli.tempfile.TemporaryDirectory",
                "mlworkflow.cli._snapshot_calculator", "mlworkflow.cli.subprocess.run",
            )]
            stack.enter_context(redirect_stdout(out))
            stack.enter_context(redirect_stderr(err))
            self.assertEqual(cli.main(self.arguments(output)), 2)
            for operation in operations:
                operation.assert_not_called()
        self.assertEqual(out.getvalue(), "")
        self.assertIn(expected, err.getvalue())
        self.assertIn("--output-dir", err.getvalue())
        for forbidden in (str(self.root), "PRIVATE", "\x1b", "Traceback", "complete distribution"):
            self.assertNotIn(forbidden, err.getvalue())
        return err.getvalue()

    def test_existing_file_and_directory_get_fresh_destination_guidance(self):
        file = self.root / "PRIVATE_file"
        file.write_bytes(b"PRIVATE_KEEP")
        directory = self.root / "PRIVATE_directory"
        directory.mkdir()
        (directory / "KEEP").write_bytes(b"PRIVATE_KEEP")
        for output in (file, directory):
            with self.subTest(kind=output.name):
                error = self.reject_before_work(output, "demo output already exists")
                self.assertIn("choose a fresh", error)
                self.assertIn("not overwritten", error)
        self.assertEqual(file.read_bytes(), b"PRIVATE_KEEP")
        self.assertEqual(list(directory.iterdir()), [directory / "KEEP"])
        self.assertEqual((directory / "KEEP").read_bytes(), b"PRIVATE_KEEP")

    def test_missing_and_non_directory_parents_get_parent_guidance(self):
        file = self.root / "PRIVATE_parent_file"
        file.write_bytes(b"KEEP")
        paths = (self.root / "PRIVATE_missing" / "child", file / "child", file / "child" / "nested")
        for output in paths:
            with self.subTest(output=output.name):
                self.reject_before_work(output, "existing parent directory")
        self.assertEqual(list(self.root.iterdir()), [file])
        self.assertEqual(file.read_bytes(), b"KEEP")

    def test_existing_symlink_destinations_remain_untouched(self):
        file = self.root / "PRIVATE_target"
        file.write_bytes(b"KEEP")
        for name, target in (("PRIVATE_file_link", file), ("PRIVATE_dir_link", self.root),
                             ("PRIVATE_dangling", self.root / "PRIVATE_absent")):
            link = self.root / name
            try:
                link.symlink_to(target, target_is_directory=target == self.root)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.subTest(name=name):
                self.reject_before_work(link, "demo output already exists")
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.readlink(), target)
        self.assertEqual(file.read_bytes(), b"KEEP")
        self.assertFalse((self.root / "PRIVATE_absent").exists())

    def test_dangling_parent_link_is_not_created_or_repaired(self):
        link = self.root / "PRIVATE_parent_link"
        target = self.root / "PRIVATE_absent"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation unavailable")
        self.reject_before_work(link / "child", "existing parent directory")
        self.assertTrue(link.is_symlink())
        self.assertFalse(target.exists())

    def test_paths_with_control_characters_do_not_enter_diagnostics(self):
        self.reject_before_work(self.root / "PRIVATE\nINJECTED\x1b[31m" / "child",
                                "existing parent directory")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_directory_symlink_parent_keeps_existing_preflight_behavior(self):
        parent = self.root / "parent"
        parent.mkdir()
        link = self.root / "PRIVATE_parent_link"
        try:
            link.symlink_to(parent, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation unavailable")
        with patch.object(cli, "verify_components", side_effect=cli.ComponentError("PRIVATE_STOP")) as verify, \
                redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.main(self.arguments(link / "fresh")), 2)
        verify.assert_called_once_with()
        self.assertIn("complete distribution", err.getvalue())
        self.assertNotIn("PRIVATE", err.getvalue())
        self.assertEqual(list(parent.iterdir()), [])

    def test_preflight_oserror_is_redacted_and_never_stages(self):
        with patch.object(cli.os.path, "lexists", side_effect=PermissionError("PRIVATE_PATH")), \
                patch.object(cli, "verify_components") as verify, \
                patch.object(cli, "_snapshot_calculator") as stage, \
                redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.main(self.arguments(self.root / "fresh")), 2)
        verify.assert_not_called()
        stage.assert_not_called()
        self.assertIn("complete distribution", err.getvalue())
        self.assertNotIn("PRIVATE", err.getvalue())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_destination_created_after_preflight_is_rejected_by_existing_runner(self):
        output = self.root / "PRIVATE_racing_output"
        original_snapshot = cli._snapshot_calculator
        original_run = subprocess.run
        results = []

        def competing_creation(destination):
            original_snapshot(destination)
            output.mkdir()
            (output / "KEEP").write_bytes(b"PRIVATE_KEEP")

        def capture_runner(*args, **kwargs):
            result = original_run(*args, **kwargs, capture_output=True, text=True, timeout=10)
            results.append(result)
            return result

        with patch.object(cli, "_snapshot_calculator", side_effect=competing_creation), \
                patch.object(cli.subprocess, "run", side_effect=capture_runner), \
                redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.main(self.arguments(output)), 2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].stdout, "")
        self.assertIn("fresh output destination", results[0].stderr)
        self.assertNotIn("PRIVATE", results[0].stderr)
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))
        self.assertEqual(list(output.iterdir()), [output / "KEEP"])
        self.assertEqual((output / "KEEP").read_bytes(), b"PRIVATE_KEEP")

    def test_fresh_destination_preserves_delegation_and_exit_code(self):
        output = self.root / "fresh"
        stage = self.root / "stage"
        with patch.object(cli, "verify_components") as verify, \
                patch.object(cli.tempfile, "TemporaryDirectory") as temporary, \
                patch.object(cli, "_snapshot_calculator") as snapshot, \
                patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as execute, \
                redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            temporary.return_value.__enter__.return_value = str(stage)
            self.assertEqual(cli.main(self.arguments(output)), 7)
        verify.assert_called_once_with()
        snapshot.assert_called_once_with(stage / "calculator")
        args, kwargs = execute.call_args
        self.assertEqual(args[0], [sys.executable,
            str(stage / "calculator/examples/semiconductor-heldout-walkthrough/run.py"),
            "--semiconductor-source", str(cli.RESOURCE_ROOT / "semiconductor"),
            "--calculator-source", str(stage / "calculator"), "--output-dir", str(output.absolute()),
            "--run-invented-example"])
        self.assertEqual(kwargs, {"env": dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), "check": False})
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))
        self.assertFalse(output.exists())

    def test_component_and_io_failures_remain_redacted_not_destination_errors(self):
        for failure in (cli.ComponentError("PRIVATE_COMPONENT"), OSError("PRIVATE_PATH")):
            with self.subTest(kind=type(failure).__name__), \
                    patch.object(cli, "verify_components", side_effect=failure), \
                    patch.object(cli, "_snapshot_calculator") as stage, \
                    redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                self.assertEqual(cli.main(self.arguments(self.root / "fresh")), 2)
                stage.assert_not_called()
                self.assertEqual(out.getvalue(), "")
                self.assertIn("complete distribution", err.getvalue())
                self.assertNotIn("demo output already exists", err.getvalue())
                self.assertNotIn("PRIVATE", err.getvalue())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cli_rejection_needs_no_ml_dependencies_or_output_writes(self):
        output = self.root / "PRIVATE_missing" / "child"
        script = """import sys
from mlworkflow.cli import main
assert main(sys.argv[1:]) == 2
assert not any(name in sys.modules for name in ('groupbench', 'numpy', 'pandas', 'sklearn'))
"""
        result = subprocess.run([sys.executable, "-S", "-c", script, *self.arguments(output)],
            cwd=self.root, env=dict(os.environ, PYTHONPATH=str(Path(cli.__file__).resolve().parents[1]),
                                   PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("existing parent directory", result.stderr)
        self.assertNotIn("PRIVATE", result.stderr)
        self.assertNotIn(str(self.root), result.stderr)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
