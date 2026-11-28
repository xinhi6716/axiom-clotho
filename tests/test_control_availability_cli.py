"""CLI contract for comparing declared control arrival without outcomes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ControlAvailabilityCliTests(unittest.TestCase):
    def test_fixed_and_update_policies_on_same_frozen_associations(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        env = dict(os.environ, PYTHONPATH=source, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows, roster = path / "private schedule.json", path / "private roster.json"
            good = ('{"schema_version":1,"time_unit":"tick","rows":['
                    '["private-b","private-group",0,2,3,0.1],'
                    '["private-a","private-group",0,1,2,-0.1],'
                    '["private-c","private-group",0,3,4,3],'
                    '["private-d","private-single",10,10,11,10]]}')
            roster.write_text('{"schema_version":1,"associations":[["private-d","private-single"],'
                              '["private-c","private-group"],["private-b","private-group"],'
                              '["private-a","private-group"]]}')
            for route in (("groupbench",), ("mlworkflow", "tabular")):
                def invoke(*arguments):
                    return subprocess.run([sys.executable, "-S", "-m", *route, *arguments],
                                          env=env, cwd=path, capture_output=True, timeout=12)
                help_result = invoke("compare-control-availability", "--help")
                self.assertEqual((help_result.returncode, help_result.stderr), (0, b""))
                self.assertIn(b"outcomes", help_result.stdout)
                args = ["compare-control-availability", rows.name, roster.name]
                rows.write_text(good)
                before = {item.name: item.read_bytes() for item in path.iterdir()}
                result = invoke(*args)
                self.assertEqual((result.returncode, result.stderr), (0, b""))
                self.assertNotIn(b"private", result.stdout)
                summary = json.loads(result.stdout)
                self.assertEqual(summary["n_rows"], 4)
                self.assertEqual(summary["n_groups"], 2)
                self.assertEqual(summary["n_available_to_both"], 1)
                self.assertEqual(summary["n_available_only_to_update"], 1)
                self.assertEqual(summary["n_available_to_neither"], 2)
                self.assertEqual({item.name: item.read_bytes() for item in path.iterdir()}, before)
                for bad in (
                    good.replace(',["private-d","private-single",10,10,11,10]', ""),
                    good.replace('private-c","private-group",0,3,4,3', 'private-c","private-group",1,3,4,3'),
                    good.replace('private-c","private-group",0,3,4,3', 'private-c","private-group",0,1,4,3'),
                    good.replace('private-c","private-group",0,3,4,3', 'private-c","private-group",0,3,3,3'),
                    good.replace('private-c","private-group",0,3,4,3', 'private-c","private-group",0,5,4,3'),
                    good.replace('private-c","private-group",0,3,4,3', 'private-c","private-group",0,3,4,3,99'),
                    good.replace("0.1", "1e99999999999999999999999", 1),
                ):
                    rows.write_text(bad)
                    result = invoke(*args)
                    self.assertEqual((result.returncode, result.stdout), (2, b""))
                    self.assertIn(b"control availability comparison rejected:", result.stderr)
                    self.assertNotIn(b"private", result.stderr)
