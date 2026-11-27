from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples/quantized-rank-enclosure/run.py"


class QuantizedRankEnclosureTests(unittest.TestCase):
    def test_cli_and_ambiguity(self):
        proc = subprocess.run([sys.executable, str(RUNNER)], cwd=ROOT, check=True,
                              capture_output=True, text=True)
        out = json.loads(proc.stdout)
        self.assertEqual(out["coarse_box_k2"], {
            "score": ["0", "5"], "p": ["1/2", "1"], "status": "unresolved",
        })
        self.assertEqual(out["geometrically_tightened_k2"]["status"], "alert")
        self.assertEqual(out["coarse_box_k1"]["status"], "unresolved")
        self.assertEqual(out["k1_feasible_no_alert"]["status"], "no_alert")
        self.assertEqual(out["k1_feasible_alert"]["status"], "alert")
        self.assertEqual(out["score_tie_no_alert"]["status"], "no_alert")

    def test_exact_grid_and_invalid_box(self):
        bound = runpy.run_path(str(RUNNER))["score_bounds"]
        refs = [(-1, 0), (1, 0)]
        lo, hi = bound(refs, [(-1, 1), (0, 1)], 2)
        scores = [
            max((x-a)**2+(y-b)**2 for a, b in refs)
            for x, y in product([Fraction(j, 8) for j in range(-8, 9)],
                                [Fraction(j, 8) for j in range(9)])
        ]
        self.assertEqual((min(scores), max(scores)), (1, 5))
        self.assertTrue(all(lo <= s <= hi for s in scores))
        with self.assertRaisesRegex(ValueError, "box"):
            bound(refs, [(1, -1), (0, 1)], 2)
