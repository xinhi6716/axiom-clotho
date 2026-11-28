"""Exact descriptive elapsed-lag comparison through real stdlib CLI entrypoints."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ElapsedLagCliTests(unittest.TestCase):
    def test_both_aliases_preserve_group_assignment_and_lag_multiplicity(self):
        base = str(Path(__file__).resolve().parents[1] / "src")
        env = dict(os.environ, PYTHONPATH=base, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, roster = [root / (name + " private.json") for name in ("reference", "candidate", "roster")]
            reference = ('{"schema_version":1,"time_unit":"tick","rows":['
                         '["private-a","private-X",0],["private-b","private-X",0],["private-c","private-X",1],'
                         '["private-d","private-Y",0],["private-e","private-Y",2]]}')
            translated = ('{"schema_version":1,"time_unit":"tick","rows":['
                          '["private-e","private-Y",30],["private-d","private-Y",28],'
                          '["private-c","private-X",9007199254740993],["private-b","private-X",9007199254740992],'
                          '["private-a","private-X",9007199254740992]]}')
            reassigned_lags = reference.replace('private-c","private-X",1', 'private-c","private-X",2').replace(
                'private-e","private-Y",2', 'private-e","private-Y",1')
            roster.write_text('{"schema_version":1,"associations":[["private-a","private-X"],'
                              '["private-b","private-X"],["private-c","private-X"],'
                              '["private-d","private-Y"],["private-e","private-Y"]]}')
            a.write_text(reference)
            args = ["compare-elapsed-lag-support", a.name, b.name, roster.name]
            for route in (("groupbench",), ("mlworkflow", "tabular")):
                def invoke(*arguments):
                    return subprocess.run([sys.executable, "-S", "-m", *route, *arguments],
                                          cwd=root, env=env, capture_output=True, timeout=12)
                help_result = invoke("compare-elapsed-lag-support", "--help")
                self.assertEqual((help_result.returncode, help_result.stderr), (0, b""))
                self.assertIn(b"multiplicities", help_result.stdout)
                for text, same, unmatched, groups in [
                    (translated, True, 0, 0),
                    (translated.replace("9007199254740993", "9007199254740993.000000000000001"), False, 2, 1),
                    (reassigned_lags, False, 3, 2),
                ]:
                    b.write_text(text)
                    before = {p.name: p.read_bytes() for p in root.iterdir()}
                    out = invoke(*args)
                    self.assertEqual((out.returncode, out.stderr), (0, b""))
                    self.assertNotIn(b"private", out.stdout)
                    report = json.loads(out.stdout)
                    self.assertIs(report["same_within_group_lag_distributions"], same)
                    self.assertEqual(report["n_pairs_per_schedule"], 4)
                    self.assertEqual(report["n_unmatched_lag_occurrences_per_schedule"], unmatched)
                    self.assertEqual(report["n_groups_with_mismatched_distributions"], groups)
                    self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)
                for invalid in (
                    translated.replace('["private-e","private-Y",30],', ""),
                    translated.replace('private-e","private-Y', 'private-e","private-X'),
                    translated.replace('"tick"', '"elsewhere"'),
                    translated.replace('9007199254740993', '1e99999999999999999999999999999'),
                ):
                    b.write_text(invalid)
                    error = invoke(*args)
                    self.assertEqual((error.returncode, error.stdout), (2, b""))
                    self.assertNotIn(b"private", error.stderr)
                    self.assertIn(b"elapsed lag support rejected:", error.stderr)

    def test_pair_limit_before_enumeration(self):
        from groupbench.elapsed_lag_support import _roster
        with self.assertRaisesRegex(ValueError, "above the fixed limit"):
            _roster({"schema_version": 1, "associations": [[str(i), "group"] for i in range(448)]})
        _, pairs = _roster({"schema_version": 1, "associations": [[str(i), "group"] for i in range(447)]})
        self.assertEqual(pairs, 99_681)
