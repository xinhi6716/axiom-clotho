import unittest
from decimal import Decimal
from fractions import Fraction
from itertools import combinations_with_replacement, product

from groupbench.quantity_contract import bound_paired_interval_score_difference


class BoundsTest(unittest.TestCase):
    def test_target_corners_can_miss_both_rank_reversals(self):
        bound = bound_paired_interval_score_difference((1, 3), (0, 4),
                                                        target_cell=(0, 4), alpha=.5)
        self.assertEqual((bound.minimum, bound.maximum), (-2, 2))
        self.assertIsNone(bound.strict_preference)
        bound = bound_paired_interval_score_difference((1, 3), (0, 4),
                                                        target_cell=(1, 3), alpha=.5)
        self.assertEqual((bound.minimum, bound.maximum, bound.strict_preference),
                         (-2, -2, "first"))

    def test_exhaustive_small_grid_with_independent_piecewise_formula(self):
        intervals = tuple(combinations_with_replacement(range(-1, 3), 2))

        def score(q, target):
            l, r = q
            if target < l:
                return r-l + 4*(l-target)
            if target > r:
                return r-l + 4*(target-r)
            return r-l

        for a,b,(lo,hi) in product(intervals, repeat=3):
            d = [score(a, Fraction(x, 4)) - score(b, Fraction(x, 4))
                 for x in range(4*lo, 4*hi+1)]
            res = bound_paired_interval_score_difference(a,b,target_cell=(lo,hi),alpha=.5)
            self.assertEqual((res.minimum, res.maximum), (min(d), max(d)))

    def test_precision_sign_and_validation(self):
        small = Decimal('0.1000000000000000000000000001')
        large = Decimal('0.1000000000000000000000000002')
        res = bound_paired_interval_score_difference((0,large),(0,small),
                                                       target_cell=(0,0),alpha=Decimal('0.1'))
        self.assertEqual(res.minimum, Fraction(1, 10**28))
        self.assertEqual(res.strict_preference, "second")
        tie = bound_paired_interval_score_difference((0,1),(0,1),target_cell=(-5,5),alpha=.5)
        self.assertIsNone(tie.strict_preference)
        for bad in (0, 1, True, float('nan')):
            with self.assertRaises(ValueError):
                bound_paired_interval_score_difference((0,1),(0,0),target_cell=(0,1),alpha=bad)
        for a,b,y in (((1,0),(0,1),(0,1)),((0,1),(0,), (0,1)),
                      ((0,1),(0,1),(1,0)),((0,1),(0,1),(True,5))):
            with self.assertRaises(ValueError):
                bound_paired_interval_score_difference(a,b,target_cell=y,alpha=.5)
