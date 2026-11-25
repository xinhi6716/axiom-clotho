"""Check equal fixed-cadence observations and a distinct prospective contrast."""

from pathlib import Path
import runpy
import subprocess
import sys
import unittest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/read-cadence-alias/run.py"


class ReadCadenceAliasTests(unittest.TestCase):
    def test_documented_cli(self):
        output = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(output.splitlines(), [
            "held chip, same cadence: time-only/read-only loss=(4, 4)",
            "same age, no intermediate reads: time-only/read-only loss=(4, 0)",
            "same-cadence determinant=0",
            "matched-age contrast determinant=-16",
        ])

    def test_many_same_cadence_ages_do_not_break_alias(self):
        mod = runpy.run_path(str(EXAMPLE))
        for t in range(1, 21):
            self.assertEqual(mod["losses"](t, t), (t, t))
            for u in range(1, 21):
                self.assertEqual(mod["determinant"]((t, t), (u, u)), 0)

    def test_matched_age_changes_read_world_only(self):
        mod = runpy.run_path(str(EXAMPLE))
        for t in range(1, 21):
            self.assertEqual(mod["losses"](t, 0), (t, 0))
            self.assertEqual(mod["determinant"]((t, 0), (t, t)), t*t)
