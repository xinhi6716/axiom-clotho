from pathlib import Path
from runpy import run_path
from unittest import TestCase

fixture = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                       / "read-spacing-recovery" / "run.py"))


class ReadSpacingRecoveryTests(TestCase):
    def test_same_count_different_spacing(self):
        result = fixture["demonstrate"]()
        self.assertEqual(result["early"]["count_only"], result["late"]["count_only"])
        self.assertEqual(result["early"]["spacing_sensitive"], "3/32")
        self.assertEqual(result["late"]["spacing_sensitive"], "3/4")
        self.assertEqual(result["dense"]["spacing_sensitive"], "31/32")
        self.assertEqual(result["none"]["spacing_sensitive"], "0")

    def test_terminal_observation_is_not_prior_exposure(self):
        with self.assertRaisesRegex(ValueError, "before endpoint"):
            fixture["summarize"]((1, 6))
