import unittest
from decimal import Decimal
from fractions import Fraction

from groupbench.quantity_contract import audit_read_qualification


class QualificationTest(unittest.TestCase):
    def setUp(self):
        self.rows = [("a", (1, 1), 1), ("b", (0, 1), 0),
                     ("c", (1, 0), 1), ("d", (0, 0), 0)]
        self.args = dict(expected_episodes=["a", "b", "c", "d"], streak=2, band=(1, 1))

    def test_selection_changes_rates_not_common_denominator(self):
        result = audit_read_qualification(self.rows, **self.args)
        self.assertEqual((result.episodes, result.followup_passes), (4, 2))
        self.assertEqual((result.last_qualified, result.streak_qualified), (2, 1))
        self.assertEqual(result.roster_rate, Fraction(1, 2))
        self.assertEqual(result.last_selected_rate, Fraction(1, 2))
        self.assertEqual(result.streak_selected_rate, 1)
        self.assertEqual(result, audit_read_qualification(self.rows[::-1], **self.args))

    def test_complete_attempted_roster_and_fresh_reads_required(self):
        bad = [self.rows[:-1], self.rows + [self.rows[1]],
               self.rows[:-1] + [("e", (0, 0), 0)],
               self.rows[:-1] + [("d", (0,), 0)],
               self.rows[:-1] + [("d", (0, 0), None)],
               self.rows[:-1] + [("d", (0, True), 0)]]
        for rows in bad:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                audit_read_qualification(rows, **self.args)
        for change in (dict(streak=True), dict(band=(2, 1)),
                       dict(expected_episodes=["a", "b", "c", "c"])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_read_qualification(self.rows, **(self.args | change))

    def test_exact_closed_band_and_empty_selection(self):
        x = Decimal("0.10000000000000000000000000000000001")
        y = Decimal("0.10000000000000000000000000000000002")
        result = audit_read_qualification([("a", (x, y), x)], expected_episodes=["a"],
                                          streak=2, band=(x, x))
        self.assertEqual(result.roster_rate, 1)
        self.assertIsNone(result.last_selected_rate)
        self.assertIsNone(result.streak_selected_rate)
