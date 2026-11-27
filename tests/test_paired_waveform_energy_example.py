from pathlib import Path
from runpy import run_path
from unittest import TestCase

fixture = run_path(str(Path(__file__).resolve().parents[1] / "examples"
                       / "paired-waveform-energy" / "run.py"))


class PairedWaveformEnergyTests(TestCase):
    def test_same_marginals_opposite_reference_order(self):
        a, b = fixture["demonstrate"]().values()
        self.assertEqual(a["product_of_channel_means_times_duration"], "8")
        self.assertEqual(b["product_of_channel_means_times_duration"], "8")
        self.assertEqual((a["paired_energy"], b["paired_energy"]), (10, 6))
        self.assertEqual((a["contrast_to_fixed_reference"], b["contrast_to_fixed_reference"]), (3, -1))

    def test_zip_cannot_silently_truncate_misaligned_channels(self):
        with self.assertRaisesRegex(ValueError, "common bin grid"):
            fixture["energy"]((1, 3), (1,))
