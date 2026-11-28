"""Outcome-free exact comparisons of measured timestamp support."""
from decimal import Decimal
from fractions import Fraction as F
import unittest

from groupbench import quantity_contract as q


class ElapsedCaptureSupportTests(unittest.TestCase):
    def test_equal_counts_can_span_different_clocks(self):
        one, two = (0, 1, 2, 3, 4), (0, 2, 4, 6, 8)
        a, = q.audit_elapsed_capture_support(one, two, windows=(("g", 0, 9),))
        self.assertEqual((a.reference_count, a.comparison_count), (5, 5))
        self.assertEqual((a.reference_span, a.comparison_span), (F(4), F(8)))
        self.assertEqual(a.shared_exact_timestamps, 3)
        self.assertFalse(a.identical_observed_grid)

    def test_boundaries_missingness_and_exact_decimal_positions(self):
        x = Decimal("2.000000000000000000000000000001")
        rows = q.audit_elapsed_capture_support((1,2,3,6), (1,x,3,7),
                    windows=(("initial",0,2),("main",2,4),("empty",4,6),("tail",6,7)))
        self.assertTrue(rows[0].identical_observed_grid)
        self.assertEqual(rows[1].shared_exact_timestamps,1)
        self.assertEqual(rows[1].comparison_span,F(3)-F(x))
        self.assertFalse(rows[2].identical_observed_grid)
        self.assertIsNone(rows[2].reference_span)
        self.assertEqual((rows[3].reference_count,rows[3].comparison_count),(1,0))
        self.assertEqual(rows[3].reference_span,0)
        self.assertIsNone(rows[3].comparison_span)

    def test_rejects_reset_overlap_duplication_or_missing_schema(self):
        for one, windows in [
            ((0,2,1),(("g",0,4),)),
            ((0,0,1),(("g",0,4),)),
            ((0,1,2),(("X",0,2),("X",2,4))),
            ((0,1,2),(("X",0,3),("Y",2,4))),
            ((0,1,2),(("X",1,1),)),
            ((0,1,2),((None,0,4),)),
            ((0,1,2),()),
        ]:
            with self.subTest(one=one,windows=windows),self.assertRaises(ValueError):
                q.audit_elapsed_capture_support(one, (0,1,2), windows=windows)
