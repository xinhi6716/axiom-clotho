"""Exact, read-only polarity diagnostics through the actual CLI entry points."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json


class PolarityMirrorCliTests(unittest.TestCase):
    def test_frozen_roster_and_exact_decimal_physical_values(self):
        with tempfile.TemporaryDirectory() as cwd:
            path = Path(cwd)
            readings, roster = path / "private readings.json", path / "private pairs.json"
            valid = ('{"schema_version":1,"rows":[["private-block",0.1,0.4,-0.1,-0.2],'
                     '["private-block",0.2,0,-0.2,0]]}')
            readings.write_text(valid)
            roster.write_text('{"schema_version":1,"expected_pairs":[["private-block",0.2],'
                              '["private-block",0.1]]}')
            before = {p.name: p.read_bytes() for p in path.iterdir()}
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                       PYTHONDONTWRITEBYTECODE="1")
            routes = [("groupbench",), ("mlworkflow", "tabular")]
            def invoke(route, *args):
                return subprocess.run([sys.executable, "-S", "-m", *route, *args],
                                      cwd=cwd, env=env, capture_output=True, timeout=15)
            for route in routes:
                with self.subTest(route=route):
                    help_result = invoke(route, "audit-polarity-mirror", "--help")
                    self.assertEqual((help_result.returncode, help_result.stderr), (0, b""))
                    self.assertIn(b"frozen", help_result.stdout)
                    out = invoke(route, "audit-polarity-mirror", readings.name, roster.name)
                    self.assertEqual((out.returncode, out.stderr), (0, b""))
                    self.assertNotIn(b"private", out.stdout)
                    data = json.loads(out.stdout)
                    self.assertEqual(data["schema_id"], "mlcalc.polarity_mirror_audit")
                    self.assertEqual(data["n_pairs"], 2)
                    self.assertEqual(data["n_both_zero_pairs"], 1)
                    self.assertEqual(data["equal_block_odd_mirror_mae"], "1/10")
                    self.assertEqual(data["worst_absolute_even_component"], "1/10")
                    self.assertEqual(data["worst_relative_even_component"], "1/3")
            self.assertEqual({p.name: p.read_bytes() for p in path.iterdir()}, before)

            # Strict decimals must not collapse this voltage mismatch to 0.1.
            bad_voltage = valid.replace("-0.1", "-0.10000000000000000001")
            omitted = valid.replace(',["private-block",0.2,0,-0.2,0]', "")
            for invalid in (bad_voltage, omitted, valid.replace('"schema_version":1', '"schema_version":true'),
                            valid.replace('"schema_version":1', '"schema_version":1,"schema_version":1')):
                readings.write_text(invalid)
                out = invoke(routes[0], "audit-polarity-mirror", readings.name, roster.name)
                self.assertEqual((out.returncode, out.stdout), (2, b""))
                self.assertIn(b"polarity mirror audit rejected:", out.stderr)
                self.assertNotIn(b"private", out.stderr)
