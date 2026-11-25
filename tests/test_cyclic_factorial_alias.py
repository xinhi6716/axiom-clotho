from pathlib import Path
import runpy
import unittest


run = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                         "examples/cyclic-factorial-alias/run.py"))


class CyclicFactorialAliasTest(unittest.TestCase):
    def test_all_pairs_balanced_but_no_conditional_contrast(self):
        a = run["audit"]()
        self.assertEqual([len(c) for c in a["pair_margins"]], [9, 9, 9])
        self.assertTrue(all(set(c.values()) == {1} for c in a["pair_margins"]))
        self.assertTrue(a["observed_worlds_equal"])
        self.assertEqual(a["in_design_contrasts"], ())

    def test_prespecified_new_cell_contrast_can_distinguish_worlds(self):
        a = run["audit"]()
        self.assertEqual(a["augmented_contrasts"], ((0, 0),))
        self.assertEqual(a["conditional_deltas"], (2, 0))
        self.assertEqual(run["conditional_support"](run["DESIGN"] * 2), ())
