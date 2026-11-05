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

    def test_numeric_byte_golden_coverage_is_separate(self):
        source, cases = self.parity.generated_numeric_byte_tests()
        self.assertGreaterEqual(len(cases), 24)
        self.assertEqual(source.count("NumericSupportReference::from_training"), len(cases))
        self.assertEqual(source.count("reference.compare(&incoming)"), 2 * len(cases))
        self.assertEqual(len(set(cases)), len(cases))
        self.assertEqual(source.count("#[test]"), len(cases))
        for name in ("bytes_all_missing", "bytes_all_missing_query", "bytes_row_limit",
                     "bytes_nan_payloads", "bytes_extreme_binary64", "bytes_signed_zero_subnormal",
                     "bytes_feature_limit", "bytes_random_bits_15"):
            self.assertIn(name, cases)

    def test_surrogate_literals_fail_clearly_without_narrowing_python(self):
        import pandas as pd
        for surrogate in ("\ud800", "\udfff"):
            frame = pd.DataFrame({surrogate: [surrogate]})
            validated = self.parity.validate_features(frame)
            report = self.parity.prediction_input_support(validated, frame)
            self.assertEqual(report["features"][0]["name"], surrogate)
            self.assertEqual(report["features"][0]["unseen_category_count"], 0)
            with self.assertRaisesRegex(ValueError, "Unpaired surrogate"):
                self.parity._rust_string(surrogate)
            with self.assertRaisesRegex(ValueError, "Unpaired surrogate"):
                self.parity._columns(validated)
            with self.assertRaisesRegex(ValueError, "Unpaired surrogate"):
                self.parity._expected(report)
            with self.assertRaisesRegex(ValueError, "Unpaired surrogate"):
                self.parity._columns(pd.DataFrame({"valid_name": [surrogate]}))
        self.assertEqual(self.parity._rust_string("𝄞"), '\"𝄞\"')

    def test_rust_matches_every_python_report_field(self):
        compiler = os.environ.get("RUSTC") or shutil.which("rustc")
        if not compiler:
            if os.environ.get("CLOTHO_REQUIRE_RUST") == "1":
                self.fail("Rust compiler required by CLOTHO_REQUIRE_RUST=1 but unavailable.")
            self.skipTest("rustc unavailable; Rust parity was not run")
        result = self.parity.check_parity(compiler)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["golden_cases"], 43)
        self.assertEqual(result["numeric_byte_cases"], len(list(self.parity.numeric_byte_cases())))
        self.assertEqual(result["total_cases"], 43 + result["numeric_byte_cases"])

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
