from pathlib import Path
from runpy import run_path
from unittest import TestCase

fixture = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                       / "series-path-drift-sign" / "run.py"))


class SeriesPathDriftSignTests(TestCase):
    def test_same_terminal_data_allow_opposite_cell_drift(self):
        output = fixture["demonstrate"]()
        self.assertEqual(output["path_change_plus_or_minus_three"], ["-1", "5"])
        self.assertEqual(output["path_change_plus_or_minus_one"], ["1", "3"])

    def test_conflicting_bounds_rejected(self):
        f = fixture["cell_change_bounds"]
        self.assertEqual(f(1, 2, -10, 10), (-1, 2))
        with self.assertRaisesRegex(ValueError, "conflict"):
            f(1, 2, 3, 4)
