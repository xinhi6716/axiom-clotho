"""Exact same-block electrical polarity comparisons; invented quantities only."""
from decimal import Decimal
from fractions import Fraction
import unittest

from groupbench import quantity_contract as contract


class QuantityPolarityMirrorTests(unittest.TestCase):
    def audit(self, rows, expected):
        return contract.audit_polarity_mirror(rows, expected_pairs=expected)

    def test_odd_passive_and_rectifying_passive_models_differ(self):
        roster = [("same_block", a) for a in (1, 2)]
        symmetric = [(b, a, 2*a, -a, -2*a) for b, a in roster]
        shifted = [(b, a, 2*a, -a, -a) for b, a in roster]
        left, right = self.audit(symmetric, roster), self.audit(shifted, roster)
        self.assertEqual(left.n_pairs, 2)
        self.assertEqual(left.equal_block_odd_mirror_mae, 0)
        self.assertEqual(right.equal_block_odd_mirror_mae, Fraction(3, 2))
        self.assertEqual(right.worst_absolute_even_component, 1)
        self.assertEqual(right.worst_relative_even_component, Fraction(1, 3))
        self.assertEqual(right.n_both_zero_pairs, 0)

    def test_equal_block_weight_and_exact_near_cancellation(self):
        roster = [("A", 1), ("A", 2), ("A", 3), ("B", 1)]
        huge = 2**60
        rows = [("A", a, huge + a, -a, -huge) for a in (1, 2, 3)]
        rows += [("B", 1, 0, -1, 0)]
        for ordered in (rows, rows[::-1]):
            result = self.audit(ordered, roster)
            self.assertEqual((result.n_blocks, result.n_pairs, result.n_both_zero_pairs), (2, 4, 1))
            self.assertEqual(result.equal_block_odd_mirror_mae, 1)
            self.assertEqual(result.worst_absolute_even_component, Fraction(3, 2))
            self.assertEqual(result.worst_relative_even_component, Fraction(3, 2*huge+3))
        no_signal = self.audit([("a", 1, 0, -1, -0.)], [("a", 1)])
        self.assertEqual(no_signal.equal_block_odd_mirror_mae, 0)
        self.assertIsNone(no_signal.worst_relative_even_component)

    def test_exact_decimal_mirror_and_full_roster(self):
        a = Decimal("0.100000000000000000000000000001")
        opposite = a.copy_negate()
        row = ("block", a, Decimal("1.2"), opposite, Decimal("-1.1"))
        self.assertEqual(self.audit([row], [("block", a)]).worst_absolute_even_component, Fraction(1, 20))
        for rows, expected in (
                ([row], [("block", a), ("other", a)]),
                ([row], [("replacement", a)]),
                ([row, row], [("block", a)]),
                ([row], [("block", a), ("block", a)]),
                ([("block", a, 1, Decimal("-0.1"), -1)], [("block", a)]),
                ([("block", a, 1, a, -1)], [("block", a)]),
                ([("block", a, True, opposite, -1)], [("block", a)]),
                ([("block", 0, 1, 0, -1)], [("block", 0)])):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                self.audit(rows, expected)


if __name__ == "__main__":
    unittest.main()
