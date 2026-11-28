"""Exact independent-roster anchor controls through actual no-site entrypoints."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class AnchorControlCliTests(unittest.TestCase):
    def test_both_cli_routes_check_a_complete_frozen_roster(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        env = dict(os.environ, PYTHONPATH=source, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "PRIVATE readings.json", root / "PRIVATE roster.json"
            a, b = ('["PRIVATE-A",0,0.3,0,0.1,0,0.2]',
                    '["PRIVATE-B",0,0.5,0,0.2,0,0.4]')
            full = '{"schema_version":1,"rows":[' + a + "," + b + "]}"
            roster = '{"schema_version":1,"expected_blocks":["PRIVATE-B","PRIVATE-A"]}'
            inputs[1].write_text(roster, encoding="utf-8")
            for route in (("groupbench",), ("mlworkflow", "tabular")):
                def invoke(*args):
                    return subprocess.run([sys.executable, "-S", "-m", *route, *args],
                                          cwd=root, env=env, capture_output=True, timeout=12)
                help_result = invoke("audit-anchor-change-controls", "--help")
                self.assertEqual((help_result.returncode, help_result.stderr), (0, b""))
                self.assertIn(b"frozen", help_result.stdout)
                inputs[0].write_text(full, encoding="utf-8")
                args = ["audit-anchor-change-controls", *(p.name for p in inputs)]
                before = {p.name: p.read_bytes() for p in root.iterdir()}
                out = invoke(*args)
                self.assertEqual((out.returncode, out.stderr), (0, b""))
                self.assertNotIn(b"PRIVATE", out.stdout)
                result = json.loads(out.stdout)
                self.assertEqual(result["schema_id"], "mlcalc.anchor_change_controls_audit")
                self.assertEqual(result["n_blocks"], 2)
                self.assertEqual(result["candidate_minus_reference"]["signed_bias"], "1/10")
                self.assertEqual(result["candidate_minus_sham"], {
                    "signed_bias": "1/4", "mean_absolute_disagreement": "1/4",
                    "centered_mean_square_disagreement": "1/400"})
                self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

                for bad in (full.replace("," + b, ""), full.replace(b, a),
                            full.replace("0.3", "1e99999999999999999999999999999"),
                            full.replace('"schema_version":1', '"schema_version":true')):
                    inputs[0].write_text(bad, encoding="utf-8")
                    error = invoke(*args)
                    self.assertEqual((error.returncode, error.stdout), (2, b""))
                    self.assertNotIn(b"PRIVATE", error.stderr)
                    self.assertNotIn(b"Traceback", error.stderr)
                    self.assertIn(b"anchor change controls audit rejected", error.stderr)
