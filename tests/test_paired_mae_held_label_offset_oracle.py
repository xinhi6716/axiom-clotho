from fractions import Fraction
from itertools import product
import unittest

from groupbench.metrics import paired_mae_held_label_offset_oracle


class PairedMAEHeldLabelOffsetOracleTests(unittest.TestCase):
    def test_group_refitting_can_reverse_native_ranking(self):
        rows = [("x", 0, 0, 6), ("y", 0, 2, 6), ("z", 0, 0, -4)]
        roster = [("z", "small"), ("x", "big"), ("y", "big")]
        out = paired_mae_held_label_offset_oracle(rows, expected_roster=roster)
        self.assertEqual(out["b"]["native_mae"], Fraction(16, 3))
        self.assertEqual(out["b"]["held_global_offset_oracle_mae"], Fraction(10, 3))
        self.assertEqual(out["b"]["held_group_offset_oracle_mae"], 0)
        self.assertEqual(out["b_minus_a"]["native_mae"], Fraction(14, 3))
        self.assertEqual(out["b_minus_a"]["held_group_offset_oracle_mae"], Fraction(-2, 3))
        self.assertEqual(out, paired_mae_held_label_offset_oracle(rows[::-1],
                                                                 expected_roster=roster[::-1]))

    def test_oracles_equal_exhaustive_constant_search(self):
        roster = [("p", "a"), ("q", "a"), ("r", "b"), ("s", "b")]
        for values in product((-2, -1, 0, 1), repeat=4):
            rows = [(row, 0, x, 0) for (row, _), x in zip(roster, values)]
            out = paired_mae_held_label_offset_oracle(rows, expected_roster=roster)["a"]
            shifts = range(-2, 3)
            gl = min(sum(abs(x - s) for x in values) for s in shifts)
            gp = min(sum(abs(x - s) for x in values[:2]) + sum(abs(x - t) for x in values[2:])
                     for s, t in product(shifts, repeat=2))
            self.assertEqual(out["held_global_offset_oracle_mae"], Fraction(gl, 4))
            self.assertEqual(out["held_group_offset_oracle_mae"], Fraction(gp, 4))

    def test_rejects_partial_duplicate_or_unpaired_cohort(self):
        rows = [("x", 0, 2, 1), ("y", 0, 1, 2)]
        roster = [("x", "a"), ("y", "b")]
        for bad in (rows[:1], rows[:1] * 2, [rows[0], ("y", 0, None, 2)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired_mae_held_label_offset_oracle(bad, expected_roster=roster)
