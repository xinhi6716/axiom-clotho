import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "verify_rram_lane_signatures.py"
HEADER = ("capture_lane", "signature")


def patterns(route):
    return [(slot, f"{lane:04b}{lane ^ 15:04b}") for slot, lane in enumerate(route)]


def run(early, late, *, late_header=HEADER, alias=False):
    with tempfile.TemporaryDirectory() as tmp:
        a, b = (Path(tmp) / name for name in ("early.csv", "late.csv"))
        for path, header, rows in ((a, HEADER, early), (b, late_header, late)):
            if alias and path == b:
                b.hardlink_to(a)
            else:
                with path.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((header, *rows))
        return subprocess.run([sys.executable, str(TOOL), "--early-signatures", str(a),
                               "--late-signatures", str(b)], capture_output=True, text=True, check=False)


class LaneSignatureVerifier(unittest.TestCase):
    def test_identity_and_route_permutation_separate_same_reference_from_same_slot(self):
        identity = list(range(16))
        positive = run(patterns(identity), list(reversed(patterns(identity))))
        self.assertEqual(positive.returncode, 0, positive.stderr)
        route = [*range(1, 16), 0]
        negative = run(patterns(identity), patterns(route))
        self.assertEqual(negative.returncode, 1, negative.stderr)
        report = json.loads(negative.stdout)
        self.assertEqual(report["late_capture_lane_to_declared_reference_lane"], route)
        self.assertEqual(report["early_to_late_slot_for_same_declared_reference_lane"], [15, *range(15)])
        self.assertEqual(report["lanes_reassigned_between_routes"], 16)
        self.assertIs(report["reference_physical_identity_verified"], False)
        stable_wrong = run(patterns(route), patterns(route))
        self.assertEqual(stable_wrong.returncode, 1)
        summary = json.loads(stable_wrong.stdout)
        self.assertEqual(summary["lanes_reassigned_between_routes"], 0)
        self.assertEqual(summary["early_nonidentity_lanes"], 16)

    def test_missing_duplicate_or_corrupt_signatures_do_not_return_a_mapping(self):
        raw = patterns(range(16))
        for result, error in (
            (run(raw, raw[:-1]), "cover all sixteen"),
            (run(raw, [("00", raw[1][1])] + raw[:-1]), "duplicate capture lane"),
            (run(raw, [(0, raw[1][1])] + raw[1:]), "duplicate decoded"),
            (run(raw, [(0, "00001110")] + raw[1:]), "signature complement"),
            (run(raw, [(0, "0000111")] + raw[1:]), "eight binary"),
            (run(raw, raw, alias=True), "distinct capture exports"),
            (run(raw, [(*row, 1) for row in raw], late_header=(*HEADER, "adc")), "no outcome"),
        ):
            self.assertEqual(result.returncode, 2)
            self.assertFalse(result.stdout)
            self.assertIn(error, result.stderr)
