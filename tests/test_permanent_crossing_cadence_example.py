"""Verify exact finite-roster contrasts without imputing failure times."""

from fractions import Fraction as F
from itertools import product
from pathlib import Path
import runpy
import unittest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/permanent-crossing-inspection-cadence/run.py"
MOD = runpy.run_path(str(EXAMPLE))
ARMS = (MOD["ARM_A"], MOD["ARM_B"])
summarize = MOD["summarize"]


class PermanentCrossingCadenceTests(unittest.TestCase):
    def test_imputations_reverse_ranking_but_both_compatible_signs_exist(self):
        result = summarize(*ARMS)
        self.assertEqual((result["first_detection_difference"], result["midpoint_difference"]),
                         (F(1), F(-1, 2)))
        contrast = lambda a, b: ((a + 10) - (b + 10)) / F(2)
        self.assertEqual((contrast(1, 6), contrast(8, 5)), (F(-5, 2), F(3, 2)))
        self.assertEqual((result["lower"], result["upper"]), (F(-3), F(2)))
        self.assertFalse(result["lower_included"])
        self.assertFalse(result["upper_included"])
        self.assertEqual(summarize(((5, 8), ARMS[0][1]), ARMS[1])["lower"], F(-1, 2))

    def test_independent_dense_grid_approaches_sharp_bounds(self):
        denom = 19
        for a, b in (ARMS, (((5, 8), (10, None)), ARMS[1]),
                     (((10, None),), ARMS[1]), (ARMS[0], ((10, None),))):
            def means(arm):
                grid = [range(denom * l + 1, denom * r + 1) if r is not None
                        else [10 * denom] for l, r in arm]
                return [F(sum(points), denom * len(arm)) for points in product(*grid)]
            differences = [x - y for x in means(a) for y in means(b)]
            r = summarize(a, b)
            self.assertEqual(min(differences), r["lower"] + (
                F(0) if r["lower_included"] else F(1, denom * len(a))))
            self.assertEqual(max(differences), r["upper"] - (
                F(0) if r["upper_included"] else F(1, denom * len(b))))

    def test_unknown_followup_and_nonbrackets_fail(self):
        for arm in ((), ((4, None),), ((5, 5),), ((1, 11),), ((False, 4),)):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                summarize(arm, ARMS[1])
        known = summarize(((10, None),), ((10, None),))
        self.assertEqual((known["lower"], known["upper"]), (F(0), F(0)))
        self.assertTrue(known["lower_included"] and known["upper_included"])
