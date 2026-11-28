import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "preflight_rram_split_read_pairing.py"
HEADER = ("chip", "address", "lane", "episode", "condition")


def run(pre_rows, post_rows, post_header=HEADER, *, alias=False):
    with tempfile.TemporaryDirectory() as directory:
        a, b = (Path(directory) / name for name in ("pre.csv", "post.csv"))
        for path, header, rows in ((a, HEADER, pre_rows), (b, post_header, post_rows)):
            if alias and path == b:
                b.hardlink_to(a)
            else:
                with path.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((header, *rows))
        return subprocess.run([sys.executable, str(TOOL), "--pre-roster", str(a), "--post-roster", str(b)],
                              text=True, capture_output=True, check=False)


class SplitReadPairing(unittest.TestCase):
    def test_row_order_does_not_define_pairing(self):
        rows = [("privateChip", 7, i, "pulse1", "first") for i in range(16)]
        post = [(chip, "07", f"{lane:02d}", episode, condition)
                for chip, _, lane, episode, condition in reversed(rows)]
        result = run(rows, post)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["exact_declared_identity_condition_pairs"], 16)
        self.assertIs(report["physical_pairing_verified"], False)
        self.assertNotIn("privateChip", result.stdout)

    def test_equal_address_counts_do_not_prove_lane_episode_and_condition_agreement(self):
        pre = [("C", 7, i, "e1", "same") for i in range(4)]
        post = [pre[0], ("C", 7, 4, "e1", "same"), ("C", 7, 2, "new", "same"),
                ("C", 7, 3, "e1", "changed")]
        result = run(pre, post)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual((report["pre_declared_rows"], report["post_declared_rows"]), (4, 4))
        self.assertEqual((report["pre_only_identities"], report["post_only_identities"],
                          report["same_identity_conflicting_condition"],
                          report["exact_declared_identity_condition_pairs"]), (2, 2, 1, 1))

    def test_duplicate_join_identity_and_outcome_columns_are_rejected(self):
        row = ("C", 7, 1, "e1", "same")
        for result, error in (
            (run([row], [row, ("C", "07", "01", "e1", "different")]), "duplicate chip"),
            (run([row], [(*row, "redacted")], (*HEADER, "ADC")), "omit read values"),
            (run([row], [row], alias=True), "must be distinct"),
            (run([row], []), "roster is empty"),
        ):
            self.assertEqual(result.returncode, 2)
            self.assertFalse(result.stdout)
            self.assertIn(error, result.stderr)
