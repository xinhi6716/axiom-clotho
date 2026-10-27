"""Single-distribution orchestration, dependency boundaries and pinned parity."""
from contextlib import redirect_stderr
import copy
import hashlib
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

from mlworkflow import cli
from mlworkflow.components import ComponentError, RESOURCE_ROOT, verify_components


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, *arguments, no_site=False):
        env = dict(os.environ, PYTHONPATH=str(RESOURCE_ROOT.parents[1]), PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.pop("SEMICONDUCTOR_TOOLBOX_SOURCE", None)
        return subprocess.run([sys.executable, *(["-S"] if no_site else []), "-m", "mlworkflow", *map(str, arguments)],
                              cwd=self.root, env=env, capture_output=True, text=True, timeout=90)

    def fixture(self):
        raw = self.root / "raw"
        result = subprocess.run([sys.executable, "-S", str(RESOURCE_ROOT / "semiconductor/examples/measurement-to-features/generate_synthetic.py"),
                                 "--output-dir", str(raw)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return raw

    def feature_args(self, raw, output):
        return ["features", "--input-dir", raw, "--output-dir", output,
                "--max-bracket-span-V", "0.11", "--accept-measurement-assumptions"]

    def test_import_and_components_are_stdlib_without_groupbench(self):
        code = "import sys; import mlworkflow.cli; from mlworkflow.components import verify_components; verify_components(); assert not any(n in sys.modules for n in ['groupbench','numpy','pandas','sklearn'])"
        env = dict(os.environ, PYTHONPATH=str(RESOURCE_ROOT.parents[1]))
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=self.root, env=env, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        for args in [("--help",), ("components",), ("features", "--help"), ("demo", "--help")]:
            result = self.invoke(*args, no_site=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_manifest_identity_and_git_blob_parity(self):
        manifest = verify_components()
        self.assertEqual([c["commit"] for c in manifest["components"]],
                         ["4c0ec2ae3eaa4e0b7afc9aaf8f9bada544e7c0de", "1f26d265411c86b0c6b30a01bc04d3605f312997", "f5704c42e2cfcf8006fb8da1f88b5308df7e92d6"])
        for component in manifest["components"]:
            self.assertEqual(component["license"], "MIT")
            self.assertIn("Permission is hereby granted", (RESOURCE_ROOT / component["license_file"]).read_text())
            for entry in component["files"]:
                data = (RESOURCE_ROOT / component["component"] / entry["path"]).read_bytes()
                self.assertEqual(hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(), entry["git_blob_sha1"])

    def test_missing_changed_and_invalid_resources_fail_closed(self):
        copy = self.root / "resources"
        shutil.copytree(RESOURCE_ROOT, copy)
        path = copy / "semiconductor/tools/iv-readpoint/iv_readpoint.py"
        original = path.read_bytes()
        for value in (b"changed", None):
            if value is None:
                path.unlink()
            else:
                path.write_bytes(value)
            with self.assertRaises(ComponentError):
                verify_components(copy)
        path.write_bytes(original)
        (copy / "manifest.json").write_text("{}")
        with self.assertRaises(ComponentError):
            verify_components(copy)

    def test_incomplete_duplicate_and_malformed_manifest_fails_closed(self):
        resources = self.root / "resources"
        shutil.copytree(RESOURCE_ROOT, resources)
        original = verify_components()
        mutations = [
            lambda m: m.update(components=[]),
            lambda m: m["components"].pop(),
            lambda m: m["components"].__setitem__(1, m["components"][0]),
            lambda m: m["components"][0].update(files=[]),
            lambda m: m["components"][0]["files"].pop(),
            lambda m: m["components"][0]["files"].__setitem__(1, m["components"][0]["files"][0]),
            lambda m: m["components"][0].update(commit="main"),
            lambda m: m["components"][0].update(repository="https://example.invalid"),
            lambda m: m["components"][0].update(license_file="calculator/LICENSE"),
            lambda m: m["components"][0]["files"][0].update(path="../escape"),
            lambda m: m["components"][0]["files"][0].update(bytes=True),
            lambda m: m["components"][0]["files"][0].update(sha256="broken"),
            lambda m: m["components"][0]["files"][0].update(git_blob_sha1="0" * 40),
            lambda m: m["components"][0].update(component=[]),
        ]
        for index, mutate in enumerate(mutations):
            manifest = copy.deepcopy(original)
            mutate(manifest)
            (resources / "manifest.json").write_text(json.dumps(manifest))
            with self.subTest(index=index), self.assertRaises(ComponentError):
                verify_components(resources)

    def test_generic_measurements_and_direct_upstream_bytes_agree_without_site(self):
        raw = self.fixture()
        output, direct = self.root / "features", self.root / "direct"
        result = self.invoke(*self.feature_args(raw, output), no_site=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        script = RESOURCE_ROOT / "semiconductor/examples/measurement-to-features/build_features.py"
        direct_result = subprocess.run([sys.executable, "-S", str(script), *map(str, self.feature_args(raw, direct)[1:])],
                                       capture_output=True, text=True, timeout=15)
        self.assertEqual(direct_result.returncode, 0, direct_result.stderr)
        self.assertEqual(files(output), files(direct))
        self.assertEqual(len(files(output)), 6)
        schema = json.loads((output / "feature_schema.json").read_bytes())
        self.assertEqual(schema["target_columns"], [])
        before = files(output)
        repeated = self.invoke(*self.feature_args(raw, output), no_site=True)
        self.assertEqual(repeated.returncode, 2)
        self.assertEqual(files(output), before)

    def test_invalid_inputs_and_unknown_options_never_create_output(self):
        raw = self.fixture()
        (raw / "iv_sweeps.csv").write_text("PRIVATE_COLUMN\nPRIVATE_VALUE\n")
        output = self.root / "features"
        for arguments in (self.feature_args(raw, output), ["features", "--PRIVATE_OPTION", "PRIVATE_VALUE"],
                          ["demo", "--output-dir", output], ["PRIVATE_OPTION"]):
            result = self.invoke(*arguments, no_site=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())
            for token in ("PRIVATE_VALUE", "PRIVATE_COLUMN", str(self.root), "Traceback"):
                self.assertNotIn(token, result.stderr)

    def test_missing_calculator_and_resources_never_execute(self):
        with patch.object(cli.importlib.util, "find_spec", return_value=None):
            with self.assertRaises(ComponentError):
                cli._snapshot_calculator(self.root / "missing")
        self.assertFalse((self.root / "missing").exists())
        with patch.object(cli, "verify_components", side_effect=ComponentError("bad")), patch.object(cli.subprocess, "run") as execute:
            with redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["features", "--help"]), 2)
            execute.assert_not_called()

    def test_demo_refuses_existing_and_dangling_paths_before_staging(self):
        paths = [self.root / "file", self.root / "dir", self.root / "missing/child"]
        paths[0].write_bytes(b"preserve")
        paths[1].mkdir()
        if hasattr(os, "symlink"):
            link = self.root / "link"
            link.symlink_to(self.root / "absent")
            paths.append(link)
        with patch.object(cli, "_snapshot_calculator") as stage:
            for output in paths:
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.main(["demo", "--output-dir", str(output), "--run-invented-example"]), 2)
            stage.assert_not_called()
        self.assertEqual(paths[0].read_bytes(), b"preserve")
        self.assertFalse((self.root / "missing").exists())

    def test_complete_demo_and_legacy_runner_have_exact_deterministic_parity(self):
        output, reference = self.root / "demo", self.root / "legacy"
        result = self.invoke("demo", "--output-dir", output, "--run-invented-example")
        self.assertEqual(result.returncode, 0, result.stderr)
        staged = self.root / "staged"
        cli._snapshot_calculator(staged)
        result = subprocess.run([sys.executable, str(staged / "examples/semiconductor-heldout-walkthrough/run.py"),
                                 "--semiconductor-source", str(RESOURCE_ROOT / "semiconductor"),
                                 "--calculator-source", str(staged), "--output-dir", str(reference), "--run-invented-example"],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(files(output), files(reference))
        receipt = json.loads((output / "COMPLETE.json").read_bytes())
        self.assertEqual(len(receipt["stages"]), 5)
        self.assertTrue(all(stage["exit_code"] == 0 for stage in receipt["stages"]))
        self.assertFalse(receipt["scientific_validation"])
        self.assertEqual(set(files(output)), set(receipt["output_files_sha256"]) | {"COMPLETE.json"})
        for name, expected in receipt["output_files_sha256"].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
        self.assertNotIn("target", (output / "future.csv").read_text().splitlines()[0])
        scores = json.loads((output / "scores.json").read_bytes())["body"]
        self.assertEqual(scores["coverage"]["scored_devices"], 6)
        before = files(output)
        result = self.invoke("demo", "--output-dir", output, "--run-invented-example")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(files(output), before)


if __name__ == "__main__":
    unittest.main()
