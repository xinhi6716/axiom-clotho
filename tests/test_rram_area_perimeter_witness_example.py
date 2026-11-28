"""Exact checks for the fixed-state area/perimeter example."""

from fractions import Fraction as F
import importlib.util
from itertools import permutations, product
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "examples" / "rram-area-perimeter-witness" / "run.py"
spec = importlib.util.spec_from_file_location("rram_area_perimeter_example", path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class AreaPerimeterWitnessTest(unittest.TestCase):
    def test_held_shape_and_frozen_baseline(self):
        self.assertEqual(example.witness(), {
            "training_area_perimeter_conductance": [["1", "4", "9"], ["4", "8", "22"], ["4", "10", "24"]],
            "training_determinant": "6",
            "fitted_intercept_area_perimeter": ["2", "3", "1"],
            "held_truth": "41",
            "held_area_only": "139/3",
            "held_area_only_abs_error": "16/3",
            "held_area_perimeter_abs_error": "0",
        })

    def test_exact_recovery_independent_of_row_order(self):
        for shapes in permutations([(1, 1), (2, 2), (1, 4)]):
            rows = [example.design(*s) for s in shapes]
            for coef in product((F(0), F(1, 7), F(11, 2)), repeat=3):
                y = [example.predict(r, coef) for r in rows]
                self.assertEqual(example.area_perimeter_fit(rows, y), coef)

    def test_relabeling_and_rotation_do_not_supply_identifying_contrast(self):
        rows = [example.design(1, 1), example.design(1, 4), example.design(4, 1)]
        self.assertEqual(rows[1], rows[2])
        self.assertEqual(example.determinant(rows), 0)
        with self.assertRaisesRegex(ValueError, "does not identify"):
            example.area_perimeter_fit(rows, (9, 24, 24))
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                example.design(value, 1)
