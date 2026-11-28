"""Exact cumulative event path sensitivity on a frozen roster."""
from collections import Counter
from decimal import Decimal as D
from fractions import Fraction as F
from itertools import product
import random
import unittest

from groupbench.interval_evaluation import bound_paired_onset_brier_paths as audit


class PairedOnsetPathTests(unittest.TestCase):
    def test_impossible_reversal_changes_ranking(self):
        rows = [("a", 0, 3, (D(".5"), D(".5")), (D(".25"), D(".75"))),
                ("b", 2, None, (D(".5"), D(".5")), (0, D(".5")))]
        roster = [("a", "chip"), ("b", "chip")]
        o = audit(rows, expected_roster=roster[::-1], horizons=(1, 2))
        self.assertEqual(o["row_weighted_b_minus_a_brier_bounds"], (F(-5, 32), F(-1, 32)))
        self.assertEqual(o["n_episodes_with_ambiguous_path"], 1)

    def test_exact_random_weighted_brute_force_and_event_boundaries(self):
        choices = [(None, None), (None, 0), (0, 1), (1, 2), (2, None),
                   (None, 2), (0, None), (0, 2), (1, None), (None, 1)]
        rng = random.Random(6716)
        for _ in range(160):
            ticks = sorted(rng.sample(range(3), rng.randrange(1, 4)))
            rows, possibilities, roster = [], [], []
            for n in range(rng.randrange(1, 4)):
                left, right = rng.choice(choices)
                a, b = (sorted(D(rng.randrange(5))/4 for _ in ticks) for _ in range(2))
                key = str(n)
                roster.append((key, rng.choice(("chip-a", "chip-b"))))
                rows.append((key, left, right, a, b))
                event_times = [F(z, 2) for z in range(7)
                               if (left is None or F(z, 2) > left)
                               and (right is None or F(z, 2) <= right)]
                possibilities.append({sum((F(y) - int(t <= h))**2 - (F(x) - int(t <= h))**2
                                          for x, y, h in zip(a, b, ticks)) / len(ticks)
                                      for t in event_times})
            actual = audit(rows, expected_roster=roster, horizons=ticks)
            reverse = audit([(k, l, r, b, a) for k, l, r, a, b in rows[::-1]],
                            expected_roster=roster[::-1], horizons=ticks)
            counts = Counter(g for _, g in roster)
            for key in ("row_weighted", "equal_group"):
                w = [F(1, len(rows)) if key == "row_weighted" else F(1, len(counts)*counts[g])
                     for _, g in roster]
                observed = [sum(x*y for x, y in zip(w, options)) for options in product(*possibilities)]
                lo, hi = min(observed), max(observed)
                self.assertEqual(actual[key + "_b_minus_a_brier_bounds"], (lo, hi))
                self.assertEqual(reverse[key + "_b_minus_a_brier_bounds"], (-hi, -lo))

    def test_rejects_roster_and_path_incoherence(self):
        row = ("j", None, None, (0, 1), (0, 1))
        for rows, roster, horizons in (
            ([row], [("wrong", "g")], (0, 1)),
            ([row], [("j", "g"), ("k", "g")], (0, 1)),
            ([row, row], [("j", "g")], (0, 1)),
            ([row], [("j", "g")], (1, 1)),
            ([row], [("j", "g")], (0, float("inf"))),
            ([("j", None, None, (1, 0), (0, 1))], [("j", "g")], (0, 1)),
            ([("j", None, None, (0,), (0, 1))], [("j", "g")], (0, 1)),
            ([("j", None, None, (0, True), (0, 1))], [("j", "g")], (0, 1)),
            ([("j", 1, 1, (0, 1), (0, 1))], [("j", "g")], (0, 1)),
        ):
            with self.assertRaises(ValueError):
                audit(rows, expected_roster=roster, horizons=horizons)
