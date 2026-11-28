"""Actual command conformance for the no-fit saved temporal validator."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from groupbench.verification import _SAFE_OPEN_SUPPORTED
from test_group_time_saved import SavedFixtureTestCase, canonical, validate, write


SOURCE = str(Path(__file__).resolve().parents[1] / "src")


@unittest.skipUnless(_SAFE_OPEN_SUPPORTED, "Safe temporal validation needs POSIX no-follow")
class SavedTemporalCliTests(SavedFixtureTestCase):
    def invoke(self, module, prefix, *arguments, stdlib_only=False):
        env = dict(os.environ, PYTHONPATH=SOURCE, PYTHONDONTWRITEBYTECODE="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
        return subprocess.run(
            [sys.executable, *(["-S"] if stdlib_only else []), "-m", module, *prefix, *map(str, arguments)],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=20)

    def test_help_and_success_on_calculator_and_workflow_cli(self):
        directory = write(self.result, self.fresh("valid")).parent
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        expected = json.loads(json.dumps(asdict(validate(directory))))
        for module, prefix in (("groupbench", ()), ("mlworkflow", ("tabular",))):
            with self.subTest(module=module):
                root_help = self.invoke(module, prefix, "--help", stdlib_only=True)
                self.assertEqual((root_help.returncode, root_help.stderr), (0, ""))
                self.assertIn("validate-group-forward-bundle", root_help.stdout)
                help_result = self.invoke(module, prefix, "validate-group-forward-bundle", "--help",
                                          stdlib_only=True)
                self.assertEqual((help_result.returncode, help_result.stderr), (0, ""))
                self.assertIn("zero fits", help_result.stdout)
                out = self.invoke(module, prefix, "validate-group-forward-bundle", directory)
                self.assertEqual((out.returncode, out.stderr), (0, ""))
                record = json.loads(out.stdout)
                self.assertEqual(record.pop("schema_id"), "mlcalc.saved_group_forward_validation")
                self.assertEqual(record.pop("schema_version"), 1)
                self.assertEqual(record, expected)
                self.assertEqual(record["validation_fits_performed"], 0)
                self.assertEqual(record["n_tested_rows"] + record["n_untested_rows"], record["n_rows"])
                for private in ("invented_feature", "invented-clock", "prediction_ridge", str(directory)):
                    self.assertNotIn(private, out.stdout)
        self.assertEqual({p.name: p.read_bytes() for p in directory.iterdir()}, before)

    def test_byte_verified_but_semantically_inconsistent_bundle_emits_no_json(self):
        directory = write(self.result, self.fresh("PRIVATE-directory")).parent
        payload = directory / "temporal_result.json"
        document = json.loads(payload.read_bytes())
        document["result"]["coverage"]["untested_predictions_present"] = True
        content = canonical(document)
        payload.write_bytes(content)
        (directory / "checksums.json").write_bytes(canonical({
            "algorithm": "sha256", "files": {"temporal_result.json": hashlib.sha256(content).hexdigest()}}))

        verified = self.invoke("groupbench", (), "verify", directory, stdlib_only=True)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        out = self.invoke("groupbench", (), "validate-group-forward-bundle", directory)
        self.assertEqual((out.returncode, out.stdout), (2, ""))
        self.assertEqual(out.stderr, "Saved temporal bundle is invalid, changed, or unsupported for safe reading.\n")
        self.assertNotIn("PRIVATE", out.stderr)
