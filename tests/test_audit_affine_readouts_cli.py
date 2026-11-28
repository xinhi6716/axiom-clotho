"""Actual no-site command tests for exact readout qualification diagnostics."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class AffineReadoutCliTests(unittest.TestCase):
    def test_exact_path_and_interior_flags_on_actual_aliases(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        env = dict(os.environ, PYTHONPATH=source, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private readouts.json"
            valid = ('{"schema_version":1,"standards":[0,1,3],"reference":[0,1,3],'
                     '"comparison":[0.01,1.01,3.01],"max_interior_error":0,'
                     '"max_path_difference":0.01}')
            routes = [("groupbench",), ("mlworkflow", "tabular")]
            def invoke(route, *args):
                return subprocess.run([sys.executable, "-S", "-m", *route, *args],
                                      cwd=directory, env=env, capture_output=True, timeout=15)
            for route in routes:
                help_result = invoke(route, "audit-affine-readouts", "--help")
                self.assertEqual((help_result.returncode, help_result.stderr), (0, b""))
                self.assertIn(b"interior", help_result.stdout)
                for text, expected in [(valid, True),
                                       (valid.replace("3.01", "3.01000000000000000001"), False)]:
                    path.write_text(text, encoding="utf-8")
                    out = invoke(route, "audit-affine-readouts", path.name)
                    self.assertEqual((out.returncode, out.stderr), (0, b""))
                    self.assertNotIn(b"private", out.stdout)
                    report = json.loads(out.stdout)
                    self.assertEqual(report["schema_id"], "mlcalc.affine_readout_audit")
                    self.assertIs(report["shared_affine_pass"], expected)
                    self.assertIs(report["interior_pass"], expected)
                    self.assertEqual(report["reference_interior_error"], "0")
                    self.assertEqual(report["low_path_difference"], "1/100")
                    self.assertEqual(path.read_text(encoding="utf-8"), text)
            for invalid in (
                    valid.replace("0,1,3", "0,3,1", 1),
                    valid.replace('"schema_version":1', '"schema_version":true'),
                    valid.replace('"schema_version":1', '"schema_version":1,"schema_version":1'),
                    valid.replace("3.01", "1e99999999999999999999999999999"),
            ):
                path.write_text(invalid, encoding="utf-8")
                out = invoke(routes[0], "audit-affine-readouts", path.name)
                self.assertEqual((out.returncode, out.stdout), (2, b""))
                self.assertIn(b"affine readout audit rejected", out.stderr)
                self.assertNotIn(b"private", out.stderr)
                self.assertNotIn(b"Traceback", out.stderr)
