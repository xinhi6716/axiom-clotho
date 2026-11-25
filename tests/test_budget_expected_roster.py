"""Distinguish complete four-cell repeats from a complete declared cohort."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from groupbench.budget_comparison import compare_saved_budgets
from test_budget_comparison import fixture


def roster_for(doc):
    pairs = sorted({(row["unit_id"], row["repeat_id"]) for row in doc["rows"]})
    return {"schema_version": 1, "unit_repeats": [
        {"unit_id": unit, "repeat_id": repeat} for unit, repeat in pairs]}


class ExpectedBudgetRosterTests(unittest.TestCase):
    def test_whole_harmful_unit_omission_changing_mean_is_detected(self):
        doc = fixture()
        original = compare_saved_budgets(doc)
        self.assertEqual(original["by_budget"]["small"]["mean_unit_gain"], 0)
        roster = roster_for(doc)
        doc["rows"] = [row for row in doc["rows"] if row["unit_id"] != "B"]
        self.assertEqual(compare_saved_budgets(doc)["by_budget"]["small"]["mean_unit_gain"], 1)
        with self.assertRaisesRegex(ValueError, "match.*roster"):
            compare_saved_budgets(doc, expected_roster=roster)

    def test_exact_membership_not_counts_and_no_inference(self):
        doc = fixture()
        roster = roster_for(doc)
        roster["unit_repeats"].reverse()
        checked = compare_saved_budgets(doc, expected_roster=roster)
        self.assertEqual(checked["coverage"].pop("expected_roster"),
                         "exact_declared_unit_repeat_pairs")
        self.assertEqual(checked, compare_saved_budgets(doc))
        changed = deepcopy(doc)
        for row in changed["rows"]:
            if (row["unit_id"], row["repeat_id"]) == ("A", "0"):
                row["repeat_id"] = "replacement"
        self.assertEqual(compare_saved_budgets(changed)["coverage"], checked["coverage"])
        with self.assertRaisesRegex(ValueError, "match.*roster"):
            compare_saved_budgets(changed, expected_roster=roster)

    def test_bad_roster_and_partial_cells_do_not_pass(self):
        doc = fixture()
        roster = roster_for(doc)
        for bad in (None, {**roster, "schema_version": True},
                    {**roster, "unit_repeats": []},
                    {**roster, "unit_repeats": roster["unit_repeats"] * 2}):
            with self.subTest(roster=bad), self.assertRaises(ValueError):
                compare_saved_budgets(doc, expected_roster=bad)
        doc["rows"].pop()
        with self.assertRaisesRegex(ValueError, "Missing matched combination"):
            compare_saved_budgets(doc, expected_roster=roster)

    def test_cli_reads_separate_roster_and_fails_without_output(self):
        with TemporaryDirectory() as directory:
            doc = fixture()
            path, frozen = (Path(directory, name) for name in ("metrics.json", "roster.json"))
            frozen.write_text(json.dumps(roster_for(doc)), encoding="utf-8")
            path.write_text(json.dumps(doc), encoding="utf-8")
            command = [sys.executable, "-m", "groupbench.budget_comparison", str(path),
                       "--expected-roster", str(frozen)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn("exact_declared_unit_repeat_pairs", result.stdout)
            doc["rows"] = [row for row in doc["rows"] if row["unit_id"] != "B"]
            path.write_text(json.dumps(doc), encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual((result.returncode, result.stdout), (2, ""))
            self.assertIn("match the expected unit/repeat roster", result.stderr)
