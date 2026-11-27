from fractions import Fraction
import unittest

from groupbench.metrics import paired_group_mae_upper_tail


class PairedGroupMaeUpperTailTests(unittest.TestCase):
    def test_mean_gain_hides_tail_harm_with_fractional_tied_group(self):
        rows = [("a", 0, 0, 4), ("b", 0, 0, 1), ("c", 0, 0, 1)]
        rows += [(str(i), 0, 15, 0) for i in range(12)]
        roster = [(key, key if key in {"a", "b", "c"} else "large") for key, *_ in rows]
        out = paired_group_mae_upper_tail(rows, expected_roster=roster,
                                          tail_mass=Fraction(3, 8))
        self.assertEqual(out["equal_group_mae_b_minus_a"], Fraction(-9, 4))
        self.assertEqual(out["upper_tail_mean_mae_b_minus_a"], 3)
        self.assertEqual(out["boundary_mae_b_minus_a"], 1)
        self.assertEqual(out["n_groups_above_boundary"], 1)
        self.assertEqual(out["n_groups_at_boundary"], 2)
        self.assertEqual(out, paired_group_mae_upper_tail(rows[::-1], expected_roster=roster[::-1],
                                                         tail_mass=Fraction(3, 8)))

    def test_full_and_tiny_tail_and_one_group(self):
        rows = [("a", 0, 0, 4), ("b", 0, 3, 0)]
        roster = [("a", "a"), ("b", "b")]
        for mass, expected in ((Fraction(1), Fraction(1, 2)),
                               (Fraction(1, 9), 4), (Fraction(1, 2), 4)):
            out = paired_group_mae_upper_tail(rows, expected_roster=roster, tail_mass=mass)
            self.assertEqual(out["upper_tail_mean_mae_b_minus_a"], expected)
        one = paired_group_mae_upper_tail(rows, expected_roster=[("a", "g"), ("b", "g")],
                                          tail_mass=Fraction(7, 17))
        self.assertEqual(one["upper_tail_mean_mae_b_minus_a"], Fraction(1, 2))

    def test_fails_no_shiftable_roster_or_inexact_tail(self):
        rows = [("a", 0, 0, 4), ("b", 0, 3, 0)]
        roster = [("a", "a"), ("b", "b")]
        for bad, mass in ((rows[:1], Fraction(1)), (rows[:1] * 2, Fraction(1)),
                          (rows, Fraction()), (rows, Fraction(2)), (rows, 0.5)):
            with self.subTest(bad=bad, mass=mass), self.assertRaises(ValueError):
                paired_group_mae_upper_tail(bad, expected_roster=roster, tail_mass=mass)
