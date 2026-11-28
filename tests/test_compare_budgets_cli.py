"""The calculator entrypoints expose the saved budget command without ML imports."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_budget_comparison import fixture
from test_budget_expected_roster import roster_for


class CompareBudgetsCliTests(unittest.TestCase):
    def test_named_aliases_match_original_and_enforce_frozen_roster(self):
        routes = [
            ["groupbench.budget_comparison"],
            ["groupbench", "compare-budgets"],
            ["mlworkflow", "tabular", "compare-budgets"],
        ]
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                       PYTHONDONTWRITEBYTECODE="1")
            doc = fixture()
            frozen, comparison = root / "frozen roster.json", root / "saved metrics.json"
            frozen.write_text(json.dumps(roster_for(doc)), encoding="utf-8")
            comparison.write_text(json.dumps(doc), encoding="utf-8")

            def invoke(route, *args):
                out = subprocess.run([sys.executable, "-S", "-m", *route, *args],
                                     cwd=root, env=env, capture_output=True, timeout=12)
                return out.returncode, out.stdout, out.stderr

            before = {p.name: p.read_bytes() for p in root.iterdir()}
            args = [comparison.name, "--expected-roster", frozen.name]
            result = invoke(routes[0], *args)
            self.assertEqual((result[0], result[2]), (0, b""))
            report = json.loads(result[1])
            self.assertEqual(report["coverage"]["expected_roster"], "exact_declared_unit_repeat_pairs")
            self.assertEqual(report["by_budget"]["small"]["mean_unit_gain"], 0)
            for route in routes[1:]:
                with self.subTest(route=route):
                    discovered = invoke(route[:-1], "--help")
                    self.assertEqual((discovered[0], discovered[2]), (0, b""))
                    self.assertIn(b"compare-budgets", discovered[1])
                    help_result = invoke(route, "--help")
                    self.assertEqual((help_result[0], help_result[2]), (0, b""))
                    self.assertIn(b"mlcalc compare-budgets", help_result[1])
                    self.assertIn(b"--expected-roster", help_result[1])
                    self.assertEqual(invoke(route, *args), result)
            self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

            doc["rows"] = [row for row in doc["rows"] if row["unit_id"] != "B"]
            comparison.write_text(json.dumps(doc), encoding="utf-8")
            # A complete remaining cohort still scores without independent
            # evidence of what was omitted. The frozen roster rejects it.
            for route in routes:
                with self.subTest(route=route):
                    unchecked = invoke(route, comparison.name)
                    self.assertEqual((unchecked[0], unchecked[2]), (0, b""))
                    self.assertEqual(json.loads(unchecked[1])["by_budget"]["small"]["mean_unit_gain"], 1)
                    failure = invoke(route, *args)
                    self.assertEqual(failure[:2], (2, b""))
                    self.assertIn(b"match the expected unit/repeat roster", failure[2])
