"""Exact checks for the common-offset interval-score sensitivity model."""
from collections import Counter
from decimal import Decimal as D
from fractions import Fraction as F
import random
import unittest

from groupbench.interval_evaluation import (
    bound_paired_interval_scores,
    bound_paired_interval_scores_common_offset as audit,
)


class CommonOffsetTests(unittest.TestCase):
    def test_common_calibration_is_not_the_product_of_rows(self):
        rows = [
            ("j", 0, 0, 1, 1, 2),
            ("k", 0, 1, 3, 0, 1),
            ("l", 0, D(".125"), D(".875"), 0, 1),
        ]
        roster = [("j", "g"), ("k", "g"), ("l", "g")]
        calc = audit(rows, expected_roster=roster, alpha=D(".5"), offset=(0, 1))
        self.assertEqual(calc["row_weighted_b_minus_a_bounds"], (F(-5, 12), F(-1, 4)))
        self.assertEqual(calc["row_weighted_extremizing_offsets"], (0, F(1, 8)))
        uncoupled = bound_paired_interval_scores(
            [(k, 0, 1, *p) for k, _, *p in rows], expected_roster=roster, alpha=.5)
        self.assertEqual(uncoupled["row_weighted_b_minus_a_bounds"], (F(-7, 4), F(13, 12)))

    def test_dyadic_enumeration_and_witnesses_with_two_weightings(self):
        rng = random.Random(1972)
        for _ in range(320):
            rows, roster = [], []
            for i in range(rng.randrange(1, 9)):
                k = str(i)
                x = D(rng.randrange(-8, 9)) / 4
                a = sorted(D(rng.randrange(-8, 9)) / 4 for _ in range(2))
                b = sorted(D(rng.randrange(-8, 9)) / 4 for _ in range(2))
                rows.append((k, x, *a, *b))
                roster.append((k, rng.choice(("one", "two", "three"))))
            alpha = rng.choice((D(".2"), D(".5"), D(".9")))
            lo, hi = sorted(rng.randrange(-9, 10) for _ in range(2))
            out = audit(rows, expected_roster=roster[::-1], alpha=alpha, offset=(D(lo)/4, D(hi)/4))
            groups = dict(roster)
            counts = Counter(groups.values())

            def direct(d):
                res = [F(0), F(0)]
                for k, x, al, au, bl, bu in rows:
                    y = F(x) + d
                    costs = []
                    for a, b in ((al, au), (bl, bu)):
                        gap = abs(y - max(F(a), min(y, F(b))))
                        costs.append(F(b-a) + 2 * gap / F(alpha))
                    delta = costs[1] - costs[0]
                    res[0] += delta / len(rows)
                    res[1] += delta / (len(counts) * counts[groups[k]])
                return res

            grid = [direct(F(t, 8)) for t in range(2 * lo, 2 * hi + 1)]
            swapped = audit([(k, x, bl, bu, al, au) for k, x, al, au, bl, bu in rows[::-1]],
                            expected_roster=roster, alpha=alpha, offset=(D(lo)/4, D(hi)/4))
            for i, key in enumerate(("row_weighted", "equal_group")):
                a, b = min(y[i] for y in grid), max(y[i] for y in grid)
                self.assertEqual(out[key + "_b_minus_a_bounds"], (a, b))
                self.assertEqual(swapped[key + "_b_minus_a_bounds"], (-b, -a))
                t1, t2 = out[key + "_extremizing_offsets"]
                self.assertEqual((direct(t1)[i], direct(t2)[i]), (a, b))

    def test_invalid_rows_or_support_do_not_subset_score(self):
        j = ("j", 1, 1, 1, 2, 2)
        for rows, roster, alpha, offset in (
            ([j], [("x", "g")], .5, (0, 1)),
            ([j], [("j", "g"), ("x", "g")], .5, (0, 1)),
            ([j, j], [("j", "g")], .5, (0, 1)),
            ([], [("j", "g")], .5, (0, 1)),
            ([("j", 0, 1, 0, 2, 3)], [("j", "g")], .5, (0, 1)),
            ([j], [("j", "g")], 1, (0, 1)),
            ([j], [("j", "g")], .5, (1, 0)),
            ([j], [("j", "g")], .5, (False, 0)),
            ([j], [("j", "g")], .5, (0, float("inf"))),
            ([j], [("j", "g")], .5, (0,)),
        ):
            with self.subTest(rows=rows, offset=offset), self.assertRaises(ValueError):
                audit(rows, expected_roster=roster, alpha=alpha, offset=offset)
        collapsed = audit([j], expected_roster=[("j", "g")], alpha=.5, offset=(0, 0))
        self.assertEqual(collapsed["row_weighted_b_minus_a_bounds"], (4, 4))
