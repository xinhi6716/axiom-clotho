"""The optional Rust port never becomes a hidden Python runtime dependency."""

import importlib.util
from copy import deepcopy
import os
from pathlib import Path
import shutil
import unittest


ROOT = Path(__file__).resolve().parents[1]
PARITY_PATH = ROOT / "tools/check_rust_parity.py"


class RustSupportParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("clotho_rust_parity", PARITY_PATH)
        cls.parity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.parity)

    def test_golden_inputs_can_be_evaluated_without_rust(self):
        source, cases = self.parity.generated_tests()
        self.assertEqual(len(cases), 43)
        self.assertEqual(len(set(cases)), len(cases))
        self.assertIn("all_missing_type_accommodation", cases)
        self.assertIn("large_inventory", cases)
        self.assertEqual(source.count("#[test]"), len(cases))

    def test_rust_matches_every_python_report_field(self):
        compiler = os.environ.get("RUSTC") or shutil.which("rustc")
        if not compiler:
            if os.environ.get("CLOTHO_REQUIRE_RUST") == "1":
                self.fail("Rust compiler required by CLOTHO_REQUIRE_RUST=1 but unavailable.")
            self.skipTest("rustc unavailable; Rust parity was not run")
        result = self.parity.check_parity(compiler)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["golden_cases"], 43)

    def test_python_oracle_schema_drift_fails_closed(self):
        _, training, incoming = list(self.parity.golden_cases())[1]
        original = self.parity.prediction_input_support(self.parity.validate_features(training), incoming)
        mutations = []
        for location in ((), ("row_flag_counts",), ("features", 0), ("features", 1)):
            changed = deepcopy(original)
            target = changed
            for key in location:
                target = target[key]
            target["new_unreviewed_field"] = 1
            mutations.append(changed)
        for key, value in (("kind", "future_kind"), ("training_support", "future_support")):
            changed = deepcopy(original)
            changed["features"][0][key] = value
            mutations.append(changed)
        for location, key in (((), "reference"), (("row_flag_counts",), "missing_value"),
                              (("features", 0), "below_training_min_count"),
                              (("features", 1), "unseen_category_count")):
            changed = deepcopy(original)
            target = changed
            for part in location:
                target = target[part]
            del target[key]
            mutations.append(changed)
        for number, report in enumerate(mutations):
            with self.subTest(mutation=number), self.assertRaises(ValueError):
                self.parity._expected(report)
        self.assertEqual(original, self.parity.prediction_input_support(
            self.parity.validate_features(training), incoming))


if __name__ == "__main__":
    unittest.main()
