"""Exercise the exact summary and sign nonidentification witness."""

from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = (Path(__file__).resolve().parents[1]
           / "examples/derived-bead-geometry-alias/run.py")


class DerivedBeadGeometryAliasTests(unittest.TestCase):
    def test_cli(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "height change=+56; derived summaries reconcile=True",
            "height change=-56; derived summaries reconcile=True",
        ])

    def test_independent_constraint_identities_and_area_bounds(self):
        m = runpy.run_path(str(EXAMPLE))
        for d, h, t in m["REPORTS"]:
            for height in (72, 128):
                H, D, W, A, B = m["geometry"]((d, h, t), height)
                self.assertEqual(H + D, 200)
                self.assertEqual(W**2 * D, h**2 * 200)
                self.assertEqual(A, t*h)
                self.assertEqual(B * (1-d), A*d)
                self.assertTrue(0 < A <= W * H and 0 < B <= W * D)
