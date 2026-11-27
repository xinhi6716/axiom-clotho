from fractions import Fraction
from itertools import product
import unittest

from groupbench.binary.metrics import paired_auc_tie_audit


class PairedAUCTieAuditTests(unittest.TestCase):
    def test_pooled_gain_does_not_imply_within_group_discrimination(self):
        rows = [("p1", 1, 0, 2), ("p2", 1, 0, 2), ("n1", 0, 0, 2),
                ("p3", 1, 0, 1), ("n2", 0, 0, 1), ("n3", 0, 0, 1)]
        roster = [(r[0], "high" if i < 3 else "low") for i, r in enumerate(rows)]
        result = paired_auc_tie_audit(rows, expected_roster=roster)
        self.assertEqual(result["pooled"]["a"], {
            "wins": 0, "ties": 9, "losses": 0, "auc": Fraction(1, 2),
            "tie_refinement_auc_bounds": (0, 1)})
        self.assertEqual(result["pooled"]["b"], {
            "wins": 4, "ties": 4, "losses": 1, "auc": Fraction(2, 3),
            "tie_refinement_auc_bounds": (Fraction(4, 9), Fraction(8, 9))})
        self.assertEqual(result["equal_group"]["auc_b_minus_a"], 0)
        self.assertEqual(result, paired_auc_tie_audit(rows[::-1], expected_roster=roster[::-1]))

    def test_exact_counts_concur_with_all_pairs_including_zero_and_subnormal(self):
        roster = [(str(i), "all") for i in range(4)]
        for scores in product((-0.0, 5e-324, 1), repeat=4):
            rows = [(str(i), i % 2, score, -score) for i, score in enumerate(scores)]
            pooled = paired_auc_tie_audit(rows, expected_roster=roster)["pooled"]
            comparisons = [(scores[p] > scores[n]) - (scores[p] < scores[n])
                           for p in (1, 3) for n in (0, 2)]
            for model, desired in (("a", 1), ("b", -1)):
                self.assertEqual(pooled[model]["wins"], comparisons.count(desired))
                self.assertEqual(pooled[model]["ties"], comparisons.count(0))
                self.assertEqual(pooled[model]["losses"], comparisons.count(-desired))

    def test_single_class_group_is_not_silently_dropped(self):
        rows = [("n", 0, 0, 0), ("p", 1, 1, 1), ("u", 1, 2, 2)]
        roster = [("n", "both"), ("p", "both"), ("u", "positive_only")]
        result = paired_auc_tie_audit(rows, expected_roster=roster)
        self.assertEqual(result["pooled"]["a"]["auc"], 1)
        self.assertIsNone(result["equal_group"])
        self.assertEqual(result["groups_without_comparable_classes"], ("positive_only",))
        self.assertIsNone(result["per_group"]["positive_only"])
        with self.assertRaises(ValueError):
            paired_auc_tie_audit(rows[1:], expected_roster=roster[1:])

    def test_incomplete_duplicate_and_invalid_pair_fail(self):
        rows = [("n", 0, 0, 0), ("p", 1, 1, 1)]
        roster = [("n", "g"), ("p", "g")]
        for bad in [rows[:1], rows[:1] * 2,
                    [rows[0], ("p", True, 0, 0)],
                    [rows[0], ("p", 1, float("nan"), 0)]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired_auc_tie_audit(bad, expected_roster=roster)
