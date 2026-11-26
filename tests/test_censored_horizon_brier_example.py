from fractions import Fraction
from itertools import product
from pathlib import Path
from runpy import run_path
from unittest import TestCase

fixture = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                       / "censored-horizon-brier" / "run.py"))


class CensoredHorizonBrierTests(TestCase):
    def test_fixed_roster_rank_can_reverse(self):
        self.assertEqual(fixture["demonstrate"](), {
            "resolved_only_gap": "-3/16", "full_roster_gap_range": ["-3/16", "1/16"],
        })

    def test_bounds_match_exhaustive_completions(self):
        pairs = ((Fraction(1, 4), Fraction(3, 4)), (Fraction(3, 4), Fraction(1, 2)))
        losses = [sum((b - y)**2 - (a - y)**2
                      for (a, b), y in zip(pairs, labels)) / 2
                  for labels in product((0, 1), repeat=2)]
        self.assertEqual(fixture["bounds"]([(*p, None) for p in pairs]),
                         (min(losses), max(losses)))
