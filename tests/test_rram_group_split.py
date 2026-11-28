import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "preflight_rram_group_split.py"
HEADER = ("chip", "address", "lane", "episode", "condition")


def cli(fit, evaluation, unit):
    with tempfile.TemporaryDirectory() as directory:
        a, b = (Path(directory) / name for name in ("fit.csv", "eval.csv"))
        for path, rows in ((a, fit), (b, evaluation)):
            with path.open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerows((HEADER, *rows))
        return subprocess.run([sys.executable, str(TOOL), "--fit-roster", str(a),
                               "--evaluation-roster", str(b), "--disjoint-at", unit],
                              text=True, capture_output=True, check=False)


class RramGroupSplit(unittest.TestCase):
    def test_held_episodes_do_not_establish_held_physical_cells(self):
        fit = [("secret", 7, 3, "train1", "c1"), ("secret", 9, 1, "train2", "c1")]
        test = [("secret", "007", "03", "held1", "other"), ("secret", 9, 7, "held2", "other"),
                ("new", 7, 3, "held1", "other")]
        episode = cli(fit, test, "episode")
        cell = cli(fit, test, "cell")
        self.assertEqual((episode.returncode, cell.returncode), (0, 1))
        report = json.loads(cell.stdout)
        stats = report["support_by_unit"]
        for unit, shared, rows in (("chip", 1, 2), ("address", 2, 2), ("cell", 1, 1), ("episode", 0, 0)):
            self.assertEqual((stats[unit]["shared_declared_units"], stats[unit]["evaluation_rows_on_fit_unit"]),
                             (shared, rows))
        self.assertNotIn("secret", cell.stdout)
        self.assertIs(report["physical_genealogy_verified"], False)

    def test_completely_held_chip_and_duplicate_declarations(self):
        one = ("C", 7, 1, "ep1", "c")
        result = cli([one], [("D", *one[1:])], "chip")
        self.assertEqual(result.returncode, 0, result.stderr)
        duplicate = cli([one], [one, ("C", "07", "01", "ep1", "new")], "cell")
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("duplicate", duplicate.stderr)
