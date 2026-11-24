"""Invented metadata, no outcome or modeling."""
from collections import Counter
import unittest
from groupbench.splitting import plan_crossed_holdout


class CrossedHoldoutTests(unittest.TestCase):
    def test_all_75_folds_preserve_exact_partition_and_double_holdout_support(self):
        rows = [(f"{d}/{r}/{i}", str(d), str(r))
                for d in range(3) for r in range(25) for i in range(9)]
        doubly_held = Counter()
        for d in range(3):
            for r in range(25):
                p = plan_crossed_holdout(rows, str(d), str(r))
                q = [p.fit, p.new_recipe, p.new_device, p.both]
                self.assertEqual([len(x) for x in q], [432, 18, 216, 9])
                self.assertEqual(Counter(x for arm in q for x in arm),
                                 Counter(row[0] for row in rows))
                for index, arm in enumerate(q):
                    self.assertEqual(arm, tuple(k for k, dev, recipe in rows
                                                if (dev == str(d)) == bool(index & 2)
                                                and (recipe == str(r)) == bool(index & 1)))
                doubly_held.update(p.both)
        self.assertEqual(doubly_held, Counter(row[0] for row in rows))

    def test_rejects_missing_quadrant_and_duplicate_row_without_echo(self):
        rows = [("00", "0", "0"), ("01", "0", "1"),
                ("10", "1", "0"), ("11", "1", "1")]
        for roster, device, recipe in ((rows[:-1], "1", "1"),
                                       (rows, "unseen", "1"),
                                       (rows, "1", "unseen")):
            with self.assertRaisesRegex(ValueError, "quadrant"):
                plan_crossed_holdout(roster, device, recipe)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            plan_crossed_holdout(rows + [("00", "1", "1")], "1", "1")
        with self.assertRaisesRegex(ValueError, "three valid identity"):
            plan_crossed_holdout(rows + [("extra", "1", "1", 1.0)], "1", "1")


if __name__ == "__main__":
    unittest.main()
