"""Check pairwise equivalence and an unequal path endpoint."""

from pathlib import Path
import runpy
import subprocess
import sys
import unittest

EXAMPLE = (Path(__file__).resolve().parents[1]
           / "examples/pairwise-tables-and-trajectories/run.py")


class PairwiseTablesAndTrajectoriesTests(unittest.TestCase):
    def test_cli(self):
        out = subprocess.check_output([sys.executable, str(EXAMPLE)], text=True)
        self.assertEqual(out.splitlines(), [
            "all three pair tables agree: True",
            "conditional recovery, even vs odd: 0 vs 1",
        ])

    def test_exact_cells_in_all_pairs_and_roster_order(self):
        mod = runpy.run_path(str(EXAMPLE))
        for parity in (0, 1):
            rows = mod["roster"](parity)
            expected = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
            self.assertEqual(mod["pair_tables"](rows[::-1]), (expected,) * 3)

    def test_independent_path_enumeration_and_no_eligible_devices(self):
        mod = runpy.run_path(str(EXAMPLE))
        self.assertEqual(mod["recovery"](((0, 0, 0), (0, 1, 1),
                                          (1, 0, 1), (1, 1, 0))), 0)
        self.assertEqual(mod["recovery"](((0, 0, 1), (0, 1, 0),
                                          (1, 0, 0), (1, 1, 1))), 1)
        self.assertIsNone(mod["recovery"](((1, 0, 0), (0, 0, 1))))
