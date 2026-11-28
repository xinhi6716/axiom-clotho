"""Exact adjacent cadence check through both dependency-free CLI routes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ScheduleGapsCliTests(unittest.TestCase):
    def test_frozen_threshold_is_exact_and_no_schedule_values_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule, contract = root / "private schedule.json", root / "private contract.json"
            good = ('{"schema_version":1,"time_unit":"private-tick","rows":['
                    '["private-d","private-X",9007199254740994.000000000000001],'
                    '["private-c","private-X",9007199254740993],'
                    '["private-a","private-X",9007199254740992],'
                    '["private-b","private-X",9007199254740992],'
                    '["private-e","private-Y",-999]]}')
            valid_contract = ('{"schema_version":1,"time_unit":"private-tick","max_gap":1,'
                              '"associations":[["private-c","private-X"],["private-d","private-X"],'
                              '["private-e","private-Y"],["private-b","private-X"],'
                              '["private-a","private-X"]]}')
            base = str(Path(__file__).resolve().parents[1] / "src")
            env = dict(os.environ, PYTHONPATH=base, PYTHONDONTWRITEBYTECODE="1")
            for route in (("groupbench",), ("mlworkflow", "tabular")):
                def invoke(*args):
                    return subprocess.run([sys.executable, "-S", "-m", *route, *args],
                                          cwd=root, env=env, capture_output=True, timeout=12)
                out = invoke("audit-schedule-gaps", "--help")
                self.assertEqual((out.returncode, out.stderr), (0, b""))
                self.assertIn(b"adjacent", out.stdout)
                args = ["audit-schedule-gaps", schedule.name, contract.name]
                schedule.write_text(good)
                contract.write_text(valid_contract)
                before = {p.name: p.read_bytes() for p in root.iterdir()}
                out = invoke(*args)
                self.assertEqual((out.returncode, out.stderr), (0, b""))
                self.assertNotIn(b"private", out.stdout)
                self.assertNotIn(b"9007199254740994", out.stdout)
                summary = json.loads(out.stdout)
                self.assertEqual(summary["n_groups"], 2)
                self.assertEqual(summary["n_singleton_groups"], 1)
                self.assertEqual(summary["n_adjacent_gaps"], 3)
                self.assertEqual(summary["n_gaps_equal_maximum"], 1)
                self.assertEqual(summary["n_gaps_above_maximum"], 1)
                self.assertEqual(summary["n_zero_gaps"], 1)
                self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)
                for bad_schedule, bad_contract in (
                    (good.replace('["private-e","private-Y",-999]', '["private-e","private-X",-999]'),
                     valid_contract),
                    (good, valid_contract.replace("private-tick", "other-unit")),
                    (good, valid_contract.replace('"max_gap":1', '"max_gap":true')),
                    (good, valid_contract.replace('"max_gap":1', '"max_gap":-1')),
                    (good, valid_contract.replace('"max_gap":1', '"max_gap":1e999999999999999999999999')),
                ):
                    schedule.write_text(bad_schedule)
                    contract.write_text(bad_contract)
                    out = invoke(*args)
                    self.assertEqual((out.returncode, out.stdout), (2, b""))
                    self.assertIn(b"schedule gap audit rejected:", out.stderr)
                    self.assertNotIn(b"private", out.stderr)
