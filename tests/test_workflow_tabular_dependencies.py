"""Unavailable delegated ML imports get safe guidance, never implicit repairs."""
import builtins
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from mlworkflow import cli

SOURCE = str(Path(cli.__file__).resolve().parents[1])


@contextmanager
def package_only_path():
    # SOURCE is the whole site-packages directory for an installed wheel.
    # Exposing it under -S would also re-enable the supposedly absent ML deps.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ("mlworkflow", "groupbench"):
            shutil.copytree(Path(SOURCE) / name, root / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        yield str(root)


class TabularDependencyGuidanceTests(unittest.TestCase):
    def invoke(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(args)
        return code, out.getvalue(), err.getvalue()

    def assert_safe_failure(self, result):
        code, out, err = result
        self.assertEqual((code, out), (2, ""))
        self.assertIn("existing ML dependencies are unavailable", err)
        self.assertIn("current Python environment", err)
        self.assertIn("clotho --help", err)
        for private in ("Traceback", "PRIVATE", "INJECTED", "\x1b", "numpy", "site-packages"):
            self.assertNotIn(private, err)

    def test_import_failures_do_not_echo_error_text_or_trigger_work(self):
        original = builtins.__import__
        for failure in (ModuleNotFoundError("PRIVATE/site-packages/numpy"),
                        ImportError("PRIVATE\nINJECTED\x1b[31m")):
            def import_hook(name, *args, **kwargs):
                if name == "groupbench.cli":
                    raise failure
                return original(name, *args, **kwargs)
            with self.subTest(kind=type(failure).__name__), ExitStack() as stack:
                stack.enter_context(patch("builtins.__import__", side_effect=import_hook))
                operations = [stack.enter_context(patch(target)) for target in (
                    "mlworkflow.cli.verify_components", "mlworkflow.cli.subprocess.run",
                    "mlworkflow.cli._snapshot_calculator", "mlworkflow.cli.tempfile.TemporaryDirectory")]
                self.assert_safe_failure(self.invoke(["tabular", "run", "PRIVATE.csv", "--out", "PRIVATE_OUT"]))
                for operation in operations:
                    operation.assert_not_called()

    def fake_calculator(self, main):
        module = types.ModuleType("groupbench.cli")
        module.main = main
        return patch.dict(sys.modules, {"groupbench.cli": module})

    def test_late_dependency_import_failure_is_also_redacted(self):
        call = Mock(side_effect=ImportError("PRIVATE_RUNTIME_PATH"))
        with self.fake_calculator(call):
            self.assert_safe_failure(self.invoke(["tabular", "binary", "--help"]))
        call.assert_called_once_with(["binary", "--help"])

    def test_successful_delegation_preserves_arguments_stdout_stderr_and_status(self):
        arguments = ["tabular", "plan", "run", "filename with spaces.csv", "--target", "exact-name"]
        original = arguments.copy()
        for status in (0, 2, 7):
            def delegated(values):
                self.assertEqual(values, original[1:])
                print("CALCULATOR_OUTPUT")
                print("CALCULATOR_DIAGNOSTIC", file=sys.stderr)
                return status
            call = Mock(side_effect=delegated)
            with self.subTest(status=status), self.fake_calculator(call):
                self.assertEqual(self.invoke(arguments), (status, "CALCULATOR_OUTPUT\n", "CALCULATOR_DIAGNOSTIC\n"))
                self.assertEqual(arguments, original)
            call.assert_called_once_with(original[1:])

    def test_help_systemexit_and_unrelated_programming_errors_are_not_hidden(self):
        for failure in (SystemExit(0), RuntimeError("programming defect")):
            with self.subTest(kind=type(failure).__name__), self.fake_calculator(Mock(side_effect=failure)):
                with self.assertRaises(type(failure)) as raised:
                    self.invoke(["tabular", "--help"])
                self.assertIs(raised.exception, failure)

    def test_real_stdlib_only_entrypoint_fails_cleanly_without_output_files(self):
        with package_only_path() as package_root, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "KEEP"
            sentinel.write_bytes(b"unchanged")
            env = dict(os.environ, PYTHONPATH=package_root, PYTHONDONTWRITEBYTECODE="1")
            script = """import importlib.util, sys
from pathlib import Path
absent = tuple(importlib.util.find_spec(n) is None for n in ('numpy', 'pandas', 'sklearn'))
owned = tuple(Path(importlib.util.find_spec(n).origin).resolve().is_relative_to(Path(sys.argv[1]).resolve())
              for n in ('mlworkflow', 'groupbench'))
print(absent + owned)
"""
            probe = subprocess.run(
                [sys.executable, "-S", "-c", script, package_root],
                cwd=directory, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual((probe.returncode, probe.stdout, probe.stderr),
                             (0, "(True, True, True, True, True)\n", ""))
            for args in (("--help",), ("binary", "--help"), ("run", "PRIVATE.csv", "--out", "PRIVATE_OUT")):
                with self.subTest(command=args[0]):
                    result = subprocess.run(
                        [sys.executable, "-S", "-m", "mlworkflow", "tabular", *args],
                        cwd=directory, env=env,
                        capture_output=True, text=True, timeout=15)
                    self.assert_safe_failure((result.returncode, result.stdout, result.stderr))
                    self.assertEqual(list(root.iterdir()), [sentinel])
                    self.assertEqual(sentinel.read_bytes(), b"unchanged")
                    self.assertNotIn(directory, result.stderr)
                    self.assertNotIn(SOURCE, result.stderr)
                    self.assertNotIn(package_root, result.stderr)

    def test_workflow_help_remains_stdlib_only_and_explains_tabular_boundary(self):
        script = """import sys
from mlworkflow.cli import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('groupbench', 'numpy', 'pandas', 'sklearn'))
"""
        with package_only_path() as package_root, tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-S", "-c", script], cwd=directory,
                env=dict(os.environ, PYTHONPATH=package_root, PYTHONDONTWRITEBYTECODE="1"),
                capture_output=True, text=True, timeout=15)
            self.assertEqual((result.returncode, result.stderr), (0, ""))
            self.assertIn("Most delegated tabular commands, including general tabular help", result.stdout)
            self.assertIn("Metadata-only plan-group-forward-windows and its help", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
