"""Blank onboarding inputs stay separate from invented examples and fitted bundles."""
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from mlworkflow import cli, contracts as c, scaffold
from mlworkflow.adapters.fet_transfer import validate_config as validate_fet


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {
    "rram": ROOT / "examples/measurement-regression/fixtures",
    "fet": ROOT / "examples/fet-device-regression/fixtures",
}


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class WorkflowInitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # resolve handles platform temp roots that are themselves symlink aliases.
        self.root = Path(self.temp.name).resolve()

    def command(self, *arguments):
        env = dict(os.environ, PYTHONPATH=str(Path(cli.__file__).resolve().parents[1]),
                   PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-S", "-m", "mlworkflow", *map(str, arguments)],
                              cwd=self.root, env=env, capture_output=True, text=True, timeout=30)

    def invoke(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(list(map(str, arguments)))
        return code, out.getvalue(), err.getvalue()

    def prepare_args(self, inputs, output):
        return ("prepare", "--input-dir", inputs / "raw", "--partitions", inputs / "partitions.csv",
                "--training-targets", inputs / "training_targets.csv", "--config", inputs / "config.json",
                "--output-dir", output)

    def test_both_exact_header_only_inventories_match_existing_public_examples(self):
        for workflow, fixture in FIXTURES.items():
            with self.subTest(workflow=workflow):
                destination = self.root / workflow
                result = self.command("init", "--workflow", workflow, "--out", destination)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("not ready to prepare", result.stdout)
                self.assertIn("No models were fitted", result.stdout)
                expected = {"README.md", "config.json", "partitions.csv", "training_targets.csv", "later_outcomes.csv", scaffold.STATUS_FILE}
                expected |= {"raw/" + p.name for p in (fixture / "raw").glob("*.csv")}
                self.assertEqual(set(snapshot(destination)), expected)
                for name in expected - {"README.md", "config.json", scaffold.STATUS_FILE}:
                    source = fixture / name
                    if workflow == "fet" and name == "later_outcomes.csv":
                        source = fixture / "heldout_outcomes.csv"
                    header = next(csv.reader(io.StringIO(source.read_text())))
                    self.assertEqual(list(csv.reader(io.StringIO((destination / name).read_text()))), [header])
                text = (destination / "README.md").read_text()
                for required in ("No measurement rows", "0.3.0b4", "0.3.0b3", "繁體中文", "only `score`", "NOT scientifically verified"):
                    self.assertIn(required, text)
                self.assertEqual(json.loads((destination / scaffold.STATUS_FILE).read_bytes()), {
                    "schema_version": "clotho-input-scaffold-v1", "workflow": workflow,
                    "status": "READY_TO_EDIT_NOT_VALIDATED"})
                if os.name == "posix":
                    self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
                    self.assertEqual((destination / "raw").stat().st_mode & 0o777, 0o700)
                    for name in expected:
                        self.assertEqual((destination / name).stat().st_mode & 0o777, 0o600)

    def test_configuration_reuses_schema_but_never_invents_scientific_choices(self):
        for workflow, fixture in FIXTURES.items():
            destination = scaffold.initialize(workflow, self.root / workflow)
            actual = json.loads((destination / "config.json").read_bytes())
            expected = json.loads((fixture / "config.json").read_bytes())
            expected.update(endpoint_name=None, target_unit=None, group_by=None, features=[],
                            declarations={key: False for key in c.DECLARATIONS})
            if workflow == "rram":
                expected["max_bracket_span_V"] = None
            else:
                expected["measurement"] = {key: None for key in expected["measurement"]}
            self.assertEqual(actual, expected)
            validator = c.validate_config if workflow == "rram" else validate_fet
            with self.assertRaises(c.WorkflowError):
                validator(actual)
            # Every scientific placeholder remains individually invalid, even if
            # someone has already completed all the other fields correctly.
            complete = json.loads((fixture / "config.json").read_bytes())
            for key in ("endpoint_name", "target_unit", "group_by", "features", "declarations"):
                with self.subTest(workflow=workflow, key=key), self.assertRaises(c.WorkflowError):
                    validator({**complete, key: actual[key]})
            measurement_keys = ("max_bracket_span_V",) if workflow == "rram" else tuple(actual["measurement"])
            for key in measurement_keys:
                changed = copy.deepcopy(complete)
                (changed if workflow == "rram" else changed["measurement"])[key] = None
                with self.subTest(workflow=workflow, key=key), self.assertRaises(c.WorkflowError):
                    validator(changed)

    def test_untouched_scaffold_prepare_fails_before_output_or_extraction(self):
        for workflow in FIXTURES:
            destination = scaffold.initialize(workflow, self.root / workflow)
            output = self.root / (workflow + "-prepared")
            before = snapshot(destination)
            result = self.command(*self.prepare_args(destination, output))
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(snapshot(destination), before)
            with patch("mlworkflow.regression._load_component", side_effect=AssertionError("extraction")), \
                    patch("mlworkflow.regression.write_bundle", side_effect=AssertionError("write")), \
                    patch("mlworkflow.adapters.fet_transfer.write_bundle", side_effect=AssertionError("write")):
                self.assertEqual(self.invoke(*self.prepare_args(destination, output))[0], 2)

    def test_public_invented_fixtures_fill_scaffolds_and_prepare_without_later_outcomes(self):
        for workflow, fixture in FIXTURES.items():
            destination = scaffold.initialize(workflow, self.root / workflow)
            # Test data only: users must provide their own real measurements.
            for relative in ("config.json", "partitions.csv", "training_targets.csv"):
                shutil.copyfile(fixture / relative, destination / relative)
            for source in (fixture / "raw").glob("*.csv"):
                shutil.copyfile(source, destination / "raw" / source.name)
            (destination / "later_outcomes.csv").unlink()
            output = self.root / (workflow + "-prepared")
            result = self.command(*self.prepare_args(destination, output))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "COMPLETE.json").is_file())
            self.assertFalse((output / "INCOMPLETE.json").exists())

    def test_invalid_arguments_are_read_only_and_do_not_echo_private_values(self):
        output = self.root / "PRIVATE_PATH"
        for args in (
            ("init",), ("init", "--workflow", "rram"), ("init", "--out", output),
            ("init", "--workflow", "PRIVATE_WORKFLOW", "--out", output),
            ("init", "--workflow", "rram", "--out", output, "--force"),
            ("init", "--workflow", "rram", "--o", output),
        ):
            code, out, err = self.invoke(*args)
            self.assertEqual((code, out), (2, ""))
            self.assertFalse(output.exists())
            for token in ("PRIVATE_PATH", "PRIVATE_WORKFLOW", "Traceback", str(self.root)):
                self.assertNotIn(token, err)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_paths_missing_parent_and_path_alias_never_overwrite(self):
        existing_file = self.root / "file"
        existing_file.write_bytes(b"preserve")
        existing_dir = self.root / "dir"
        existing_dir.mkdir()
        empty = self.root / "empty"
        empty.mkdir()
        (existing_dir / "marker").write_bytes(b"preserve")
        before = snapshot(self.root)
        for output in (existing_file, existing_dir, empty, self.root / "missing/child", existing_dir / "../dir"):
            self.assertEqual(self.invoke("init", "--workflow", "rram", "--out", output)[0], 2)
            self.assertEqual(snapshot(self.root), before)
        self.assertFalse((self.root / "missing").exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_final_dangling_and_ancestor_symlinks_are_rejected(self):
        target = self.root / "target"
        target.mkdir()
        (target / "marker").write_bytes(b"preserve")
        link = self.root / "alias"
        dangling = self.root / "dangling"
        link.symlink_to(target, target_is_directory=True)
        dangling.symlink_to(self.root / "absent", target_is_directory=True)
        for output in (link, dangling, link / "new", link / "../new"):
            self.assertEqual(self.invoke("init", "--workflow", "fet", "--out", output)[0], 2)
        self.assertEqual(snapshot(target), {"marker": b"preserve"})
        self.assertFalse((self.root / "new").exists())
        self.assertFalse((self.root / "absent").exists())

    def test_creation_race_preserves_the_competing_directory(self):
        output = self.root / "raced"
        original = Path.mkdir
        def race(path, *args, **kwargs):
            if path == output:
                original(path)
                (path / "other-owner").write_bytes(b"preserve")
            return original(path, *args, **kwargs)
        with patch.object(Path, "mkdir", autospec=True, side_effect=race):
            code, out, err = self.invoke("init", "--workflow", "rram", "--out", output)
        self.assertEqual((code, out), (2, ""))
        self.assertEqual(snapshot(output), {"other-owner": b"preserve"})

    def test_exclusive_file_write_race_never_clobbers_or_cleans_up(self):
        output = self.root / "file-race"
        original = c.write_new
        def race(path, data):
            if path.name == "config.json":
                path.write_bytes(b"other-owner")
            return original(path, data)
        with patch.object(c, "write_new", side_effect=race):
            code, out, err = self.invoke("init", "--workflow", "rram", "--out", output)
        self.assertEqual((code, out), (2, ""))
        self.assertEqual((output / "config.json").read_bytes(), b"other-owner")
        self.assertEqual(json.loads((output / scaffold.STATUS_FILE).read_bytes())["status"], "INCOMPLETE_DO_NOT_USE")
        self.assertFalse((output / "COMPLETE.json").exists())

    def test_injected_write_failure_leaves_incomplete_output_and_retry_refuses(self):
        output = self.root / "interrupted"
        original = c.write_new
        def interrupted(path, data):
            if path.name == "config.json":
                raise OSError("PRIVATE_PATH")
            return original(path, data)
        with patch.object(c, "write_new", side_effect=interrupted):
            code, out, err = self.invoke("init", "--workflow", "fet", "--out", output)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("partial directory may remain", err)
        self.assertNotIn("PRIVATE_PATH", err)
        self.assertEqual(json.loads((output / scaffold.STATUS_FILE).read_bytes())["status"], "INCOMPLETE_DO_NOT_USE")
        self.assertFalse((output / "COMPLETE.json").exists())
        before = snapshot(output)
        self.assertEqual(self.invoke("init", "--workflow", "fet", "--out", output)[0], 2)
        self.assertEqual(snapshot(output), before)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_observed_root_replacement_fails_and_never_deletes_victim_marker(self):
        for as_link in (True, False):
            with self.subTest(as_link=as_link):
                parent = self.root / str(as_link)
                parent.mkdir()
                output, moved, victim = (parent / name for name in ("output", "original", "victim"))
                victim.mkdir()
                (victim / "raw").mkdir()
                (victim / "INCOMPLETE.json").write_bytes(b"foreign marker: preserve exactly")
                original = c.write_new
                def replace_root(path, data):
                    if path.name == "README.md":
                        output.rename(moved)
                        if as_link:
                            output.symlink_to(victim, target_is_directory=True)
                        else:
                            victim.rename(output)
                    return original(path, data)
                with patch.object(c, "write_new", side_effect=replace_root):
                    code, out, err = self.invoke("init", "--workflow", "rram", "--out", output)
                self.assertEqual((code, out), (2, ""))
                self.assertIn("identity changed", err)
                self.assertEqual((output / "INCOMPLETE.json").read_bytes(), b"foreign marker: preserve exactly")
                self.assertEqual(json.loads((moved / scaffold.STATUS_FILE).read_bytes())["status"], "INCOMPLETE_DO_NOT_USE")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_status_replacement_or_same_inode_symlink_fails_without_touching_foreign_bytes(self):
        for as_link in (True, False):
            output = self.root / str(as_link)
            moved = self.root / (str(as_link) + "-original-status")
            original = c.write_new
            def replace_status(path, data):
                if path.name == "README.md":
                    marker = output / scaffold.STATUS_FILE
                    marker.rename(moved)
                    if as_link:
                        marker.symlink_to(moved)
                    else:
                        marker.write_bytes(b"foreign status: preserve exactly")
                return original(path, data)
            with patch.object(c, "write_new", side_effect=replace_status):
                code, out, err = self.invoke("init", "--workflow", "fet", "--out", output)
            self.assertEqual((code, out), (2, ""))
            self.assertIn("identity changed", err)
            self.assertEqual(json.loads(moved.read_bytes())["status"], "INCOMPLETE_DO_NOT_USE")
            if as_link:
                self.assertTrue((output / scaffold.STATUS_FILE).is_symlink())
            else:
                self.assertEqual((output / scaffold.STATUS_FILE).read_bytes(), b"foreign status: preserve exactly")

    def test_late_status_replacement_only_updates_original_handle_and_returns_failure(self):
        output = self.root / "late"
        moved = self.root / "original-status"
        original = scaffold._write_status
        def replace_after_ready(handle, workflow, status):
            original(handle, workflow, status)
            if status == "READY_TO_EDIT_NOT_VALIDATED":
                (output / scaffold.STATUS_FILE).rename(moved)
                (output / scaffold.STATUS_FILE).write_bytes(b"foreign: preserve")
        with patch.object(scaffold, "_write_status", side_effect=replace_after_ready):
            code, out, err = self.invoke("init", "--workflow", "rram", "--out", output)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("identity changed", err)
        self.assertEqual((output / scaffold.STATUS_FILE).read_bytes(), b"foreign: preserve")
        self.assertEqual(json.loads(moved.read_bytes())["status"], "INCOMPLETE_DO_NOT_USE")

    def test_final_inventory_and_payload_changes_fail_without_cleanup(self):
        for mutation in ("inventory", "bytes"):
            output = self.root / mutation
            original = c.write_new
            def alter_after_last_write(path, data):
                original(path, data)
                if path.name == "training_targets.csv":
                    if mutation == "inventory":
                        (output / "foreign").write_bytes(b"preserve")
                    else:
                        (output / "config.json").write_bytes(b"foreign replacement bytes")
            with patch.object(c, "write_new", side_effect=alter_after_last_write):
                code, out, err = self.invoke("init", "--workflow", "rram", "--out", output)
            self.assertEqual((code, out), (2, ""))
            self.assertIn(mutation + " changed", err)
            preserved = output / ("foreign" if mutation == "inventory" else "config.json")
            self.assertEqual(preserved.read_bytes(), b"preserve" if mutation == "inventory" else b"foreign replacement bytes")
            self.assertEqual(json.loads((output / scaffold.STATUS_FILE).read_bytes())["status"], "INCOMPLETE_DO_NOT_USE")

    def test_help_discovery_is_read_only_under_python_without_site_packages(self):
        for args in ((), ("--help",), ("init", "--help")):
            result = self.command(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--workflow", result.stdout)
            self.assertIn("0.3.0b4", result.stdout)
            self.assertIn("0.3.0b3" if args and args[0] == "init" else "frozen b3", result.stdout)
            if args != ("init", "--help"):
                for command in ("init", "prepare", "plan", "run", "score", "tabular", "features", "demo", "components"):
                    self.assertIn(command, result.stdout)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_init_has_no_ml_input_read_network_or_subprocess_work(self):
        # -S makes the no-runtime-dependency promise independently executable.
        code = """
import sys
from pathlib import Path
from unittest.mock import patch
from mlworkflow import cli, contracts
read_regular = contracts.read_regular
def read_generated(path):
    assert Path(path).is_relative_to(Path('no-dependencies')), 'read outside generated output'
    return read_regular(path)
with patch('mlworkflow.cli.subprocess.run', side_effect=AssertionError('subprocess')), \\
     patch('mlworkflow.contracts.read_regular', side_effect=read_generated), \\
     patch('socket.socket.connect', side_effect=AssertionError('network')):
    assert cli.main(['init', '--workflow', 'fet', '--out', 'no-dependencies']) == 0
assert not any(name in sys.modules for name in ('groupbench', 'numpy', 'pandas', 'sklearn'))
"""
        env = dict(os.environ, PYTHONPATH=str(Path(cli.__file__).resolve().parents[1]), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=self.root, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
