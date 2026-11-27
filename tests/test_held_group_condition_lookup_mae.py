from fractions import Fraction
import unittest

from groupbench.metrics import held_group_condition_lookup_mae


class HeldGroupConditionLookupTests(unittest.TestCase):
    train = [("a", 0), ("b", 2), ("c", 9), ("d", 9)]
    tr = [("a", "t1", "low"), ("b", "t2", "low"),
          ("c", "t1", "high"), ("d", "t2", "high")]
    held = [("x", 1), ("y", 9), ("z", 8)]
    hr = [("x", "h1", "low"), ("y", "h1", "high"), ("z", "h2", "new")]

    def test_gain_exclusively_on_seen_conditions(self):
        out = held_group_condition_lookup_mae(self.train, self.held, expected_train=self.tr,
                                              expected_held=self.hr)
        self.assertEqual(out["micro"]["lookup_minus_global_mae"], Fraction(-8, 3))
        self.assertEqual(out["equal_held_group"]["lookup_minus_global_mae"], -2)
        seen, unseen = (out["by_condition_support"][key] for key in ("seen", "unseen"))
        self.assertEqual(seen, {"n_held_rows": 2, "micro": {
            "lookup_mae": 0, "global_mean_mae": 4, "lookup_minus_global_mae": 4 * -1}})
        self.assertEqual(unseen, {"n_held_rows": 1, "micro": {
            "lookup_mae": 3, "global_mean_mae": 3, "lookup_minus_global_mae": 0}})
        self.assertEqual(out, held_group_condition_lookup_mae(
            self.train[::-1], self.held[::-1], expected_train=self.tr[::-1], expected_held=self.hr[::-1]))

    def test_exact_training_fraction_and_empty_seen_stratum(self):
        out = held_group_condition_lookup_mae([("a", 0), ("b", 0), ("c", 1)], [("e", 0)],
                                              expected_train=self.tr[:3],
                                              expected_held=[("e", "held", "novel")])
        self.assertIsNone(out["by_condition_support"]["seen"]["micro"])
        self.assertEqual(out["micro"]["lookup_mae"], Fraction(1, 3))

    def test_rejects_shared_group_and_incomplete_duplicate_or_reused_ids(self):
        for rows, declared in (
            (self.held, [("x", "t1", "low"), *self.hr[1:]]),
            (self.held[1:], self.hr),
            (self.held[:1] * 3, self.hr),
            ([("a", 0), *self.held[1:]], [("a", "h1", "low"), *self.hr[1:]]),
        ):
            with self.subTest(rows=rows, declared=declared), self.assertRaises(ValueError):
                held_group_condition_lookup_mae(self.train, rows, expected_train=self.tr,
                                                expected_held=declared)
