import unittest
from decimal import Decimal
from fractions import Fraction

from groupbench.quantity_contract import compare_affine_readouts

class AffineReadoutTests(unittest.TestCase):
    def compare(self, comparison, *, interior=0, difference=0):
        return compare_affine_readouts([2, 6, 10], [3.5, 8.5, 13.5], comparison,
                                       max_interior_error=interior, max_path_difference=difference)

    def test_matching_transfer_and_strict_bound(self):
        def levels(offset):
            return [Decimal("3.50")+offset, Decimal("8.50")+offset, Decimal("13.50")+offset]
        at = self.compare(levels(Decimal(".125")), difference=Decimal(".125"))
        above = self.compare(levels(Decimal(".12501")), difference=Decimal(".125"))
        self.assertEqual(at.max_affine_path_difference, Fraction(1, 8))
        self.assertTrue(at.shared_affine_pass)
        self.assertFalse(above.shared_affine_pass)
        self.assertTrue(above.interior_pass)

    def test_extra_levels_reveal_bend_hidden_by_matching_endpoints(self):
        bent = self.compare([3.5, 9.5, 13.5])
        self.assertFalse(bent.interior_pass)
        self.assertFalse(bent.shared_affine_pass)
        self.assertEqual((bent.comparison_interior_error, bent.max_affine_path_difference), (1, 0))
        self.assertTrue(self.compare([3.5, 9.5, 13.5], interior=1).shared_affine_pass)

    def test_affine_but_not_shared(self):
        out = self.compare([4, 10, 16], difference=2)
        self.assertTrue(out.interior_pass)
        self.assertEqual((out.low_path_difference, out.high_path_difference), (Fraction(1,2), Fraction(5,2)))
        self.assertFalse(out.shared_affine_pass)

    def test_offcenter_interior_check_uses_physical_position(self):
        out = compare_affine_readouts([0, 2, 10], [3, 7, 23], [3, 7, 23],
                                      max_interior_error=0, max_path_difference=0)
        self.assertTrue(out.shared_affine_pass)

    def test_invalid_or_decreasing_inputs_fail_closed(self):
        for bad in ([True, 4, 5], [float("nan"), 4, 5], [1, 2, 1],
                    [1, 2, Decimal("1e201")], [0, 2, 10**200]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.compare(bad)
        with self.assertRaises(ValueError):
            self.compare([1, 4, 5], difference=-1)
