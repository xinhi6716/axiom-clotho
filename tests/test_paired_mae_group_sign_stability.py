import unittest

from groupbench.metrics import paired_mae_group_sign_stability


class PairedMaeGroupSignStabilityTests(unittest.TestCase):
    def test_opposite_full_winners_different_reversals_and_tie(self):
        rows = [("a1", 0, 0, 2), ("a2", 0, 0, 2), ("a3", 0, 0, 2),
                ("b", 0, 3, 0), ("c", 0, 2, 0)]
        roster = [(r[0], r[0][0]) for r in rows]
        result = paired_mae_group_sign_stability(rows, expected_roster=roster)
        self.assertEqual(result["winner"], {"micro": "A", "equal_group": "B"})
        self.assertEqual(result["without_group"], (
            {"omitted_group": "a", "n_omitted_rows": 3, "micro": "B", "equal_group": "B"},
            {"omitted_group": "b", "n_omitted_rows": 1, "micro": "A", "equal_group": "tie"},
            {"omitted_group": "c", "n_omitted_rows": 1, "micro": "A", "equal_group": "B"},
        ))
        self.assertEqual(result["changes"]["micro"]["strict_reversals"], ("a",))
        self.assertEqual(result["changes"]["equal_group"]["strict_reversals"], ())
        self.assertEqual(result["changes"]["equal_group"]["ties_created"], ("b",))
        self.assertEqual(result, paired_mae_group_sign_stability(
            rows[::-1], expected_roster=roster[::-1]))

    def test_smallest_positive_residual_decides_after_huge_cancellation(self):
        rows = [("x", 0, 0, 1e300), ("y", 0, 1e300, 0), ("z", 0, 0, 5e-324)]
        roster = [(r[0], r[0]) for r in rows]
        result = paired_mae_group_sign_stability(rows, expected_roster=roster)
        for weighting in ("micro", "equal_group"):
            self.assertEqual(result["winner"][weighting], "A")
            self.assertEqual(result["changes"][weighting]["strict_reversals"], ("x",))
            self.assertEqual(result["changes"][weighting]["ties_created"], ("z",))

    def test_original_tie_and_single_group(self):
        rows = [("a", 0, 0, 2), ("b", 0, 2, 0)]
        result = paired_mae_group_sign_stability(rows, expected_roster=[("a", "a"), ("b", "b")])
        self.assertEqual(result["changes"]["micro"]["ties_resolved"], ("a", "b"))
        one = paired_mae_group_sign_stability(rows, expected_roster=[("a", "a"), ("b", "a")])
        self.assertEqual(one["winner"], {"micro": "tie", "equal_group": "tie"})
        self.assertEqual(one["without_group"], ())

    def test_cannot_trim_frozen_roster_or_repeat_one_model_pair(self):
        rows = [("x", 0, 0, 1), ("y", 0, 1, 0)]
        roster = [("x", "chip1"), ("y", "chip2")]
        for bad in (rows[:1], rows[:1] * 2, rows[:1] + [("y", 0, 1, None)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired_mae_group_sign_stability(bad, expected_roster=roster)
