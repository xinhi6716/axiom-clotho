"""Synthetic ALD declaration conflicts only; no paper data or model fits."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from groupbench.process_contract import (
    ALDExperimentalContract, ALDObservation, ProcessFeatureDeclaration,
    ThicknessCycleDerivation, MAX_RECORDS, MAX_CYCLES, MAX_ISSUES, preview_ald_contract,
)


def contract(**changes):
    return replace(ALDExperimentalContract(
        temperature_role="substrate", temperature_basis="setpoint",
        target_method="total_thickness_div_cycles", target_column="growth",
        features=(ProcessFeatureDeclaration("substrate_temperature", "process_setting", "before_process"),)
    ), **changes)


def observation(**changes):
    # Invented exact arithmetic fixture, not a published experimental claim.
    recipe = ThicknessCycleDerivation("12.5", "nm", 250, "synthetic-source", "v1", "synthetic table 1", "condition-1")
    row = ALDObservation(
        source_id="synthetic-source", source_version="v1", source_locator="synthetic table 1",
        condition_id="condition-1", temperature="200", temperature_unit="degC",
        temperature_role="substrate", temperature_basis="setpoint", temperature_condition_id="condition-1",
        target="0.5", target_unit="angstrom/cycle", target_method="total_thickness_div_cycles",
        target_condition_id="condition-1", observation_kind="experimental", evidence_scope="full_text",
        batch_id="batch-1", replicate_id="sample-1", derivation=recipe,
    )
    return replace(row, **changes)


def codes(report):
    return set(report["issue_counts"])


class ProcessContractTests(unittest.TestCase):
    def test_exact_average_and_equivalent_units(self):
        row = observation()
        for candidate in (row, replace(row, target="0.05", target_unit="nm/cycle"),
                          replace(row, derivation=replace(row.derivation, thickness="125", thickness_unit="angstrom"))):
            result = preview_ald_contract([candidate], contract=contract())
            self.assertEqual(result["status"], "no_declared_conflicts")
            self.assertEqual(result["issue_count"], 0)
            self.assertEqual(result["training_admission"], "not_performed")

    def test_exact_setpoint_consistency_across_replicates(self):
        issue = "conflicting_temperature_setpoints"
        row = observation()
        equal = replace(row, replicate_id="sample-2", temperature_unit="K", temperature="473.15")
        different = replace(equal, temperature="473.150000000000000001")
        other_version = replace(different, source_version="v2",
                                derivation=replace(row.derivation, source_version="v2"))
        self.assertEqual(preview_ald_contract([row, different], contract=contract())["issue_counts"], {})
        cases = ((equal, {}), (different, {issue: 2}), (other_version, {}))
        for second, expected in cases:
            with self.subTest(second=second), localcontext() as context:
                context.prec = 1
                r = preview_ald_contract([row, second], contract=contract(),
                                         check_consistent_temperature_setpoints=True)
                self.assertEqual(r["issue_counts"], expected)
                if expected:
                    self.assertEqual([i["row_position"] for i in r["issues"]], [0, 1])
                    self.assertEqual({i["field"] for i in r["issues"]}, {"temperature"})

        measured = [replace(item, temperature_basis="measured") for item in (row, different)]
        r = preview_ald_contract(measured, contract=contract(temperature_basis="measured"),
                                 check_consistent_temperature_setpoints=True)
        self.assertEqual(r["issue_counts"], {})
        with self.assertRaisesRegex(ValueError, "explicit bool"):
            preview_ald_contract([row], contract=contract(), check_consistent_temperature_setpoints=1)

    def test_wrong_temperature_role_basis_and_condition_are_distinct_conflicts(self):
        row = observation(temperature="80", temperature_role="precursor_vessel",
                          temperature_basis="measured", temperature_condition_id="other-condition")
        result = preview_ald_contract([row], contract=contract())
        self.assertEqual(result["issue_counts"]["declared_role_or_method_mismatch"], 2)
        self.assertIn("condition_identity_mismatch", codes(result))
        self.assertEqual(row.temperature, "80")  # No automatic temperature substitution.

    def test_absolute_temperature_bound_uses_exact_declared_units(self):
        # Below-bound deltas can disappear when Celsius is narrowed to float64
        # or rounded under a low-precision global Decimal context.
        cases = (
            ("K", "-1", True), ("K", "-1e-200", True),
            ("degC", "-273.150000000000000000000000000000000001", True),
            ("degC", "-2.731500000000000000000000000000000001e2", True),
            ("K", "-0.0", False), ("K", "0", False), ("K", "1e-200", False),
            ("degC", "-273.15", False),
            ("degC", "-273.149999999999999999999999999999999999", False),
            ("degC", "-1", False), ("degC", "200", False),
        )
        for basis in ("setpoint", "measured"):
            for unit, token, below in cases:
                with self.subTest(basis=basis, unit=unit, token=token), localcontext() as context:
                    context.prec, context.Emax, context.Emin = 1, 1, -1
                    row = observation(temperature_basis=basis, temperature_unit=unit, temperature=token)
                    report = preview_ald_contract(
                        [row], contract=contract(temperature_basis=basis), check_absolute_temperature=True,
                    )
                    self.assertEqual(report["issue_counts"], {"temperature_below_absolute_zero": 1} if below else {})
                    self.assertEqual(report["status"], "declaration_issues_found" if below else "no_declared_conflicts")
                    self.assertEqual(row.temperature, token)
                    if below:
                        self.assertEqual(report["issues"], [{
                            "code": "temperature_below_absolute_zero", "field": "temperature",
                            "row_position": 0, "feature_position": None,
                        }])

    def test_invalid_units_and_tokens_are_not_reinterpreted_for_temperature_bound(self):
        for unit, token, expected in (
                ("C", "-500", {"unsupported_temperature_unit": 1}),
                ("K", "-1e-201", {"invalid_decimal_token": 1}),
                ([], "-500", {"unsupported_temperature_unit": 1}),
                ("F", "NaN", {"unsupported_temperature_unit": 1, "invalid_decimal_token": 1})):
            with self.subTest(unit=unit, token=token):
                report = preview_ald_contract(
                    [observation(temperature_unit=unit, temperature=token)], contract=contract(),
                    check_absolute_temperature=True,
                )
                self.assertEqual(report["issue_counts"], expected)

    def test_temperature_check_opt_in_is_explicit(self):
        for choice in (None, 0, 1, "true"):
            with self.subTest(choice=choice), self.assertRaisesRegex(ValueError, "explicit bool"):
                preview_ald_contract([observation()], contract=contract(), check_absolute_temperature=choice)

    def test_opt_in_derived_absolute_thickness_and_average_sign(self):
        row = observation()
        for unit, magnitude, target in (
                ("nm", "-12.5", "-0.5"),
                ("angstrom", "-1e-200", "-1e-200"),
        ):
            recipe = replace(row.derivation, thickness_unit=unit, thickness=magnitude)
            candidate = replace(row, derivation=recipe, target=target)
            with self.subTest(unit=unit), localcontext() as ctx:
                ctx.prec, ctx.Emax, ctx.Emin = 1, 1, -1
                # Use a valid text bound for the target and accept the tiny
                # arithmetic difference explicitly without hiding its sign.
                selected = contract(derived_target_tolerance_angstrom_per_cycle="1e-200")
                self.assertEqual(preview_ald_contract([candidate], contract=selected)["issue_counts"], {})
                r = preview_ald_contract([candidate], contract=selected,
                                         check_nonnegative_derived_thickness=True)
                self.assertEqual(r["issue_counts"], {"negative_derived_average": 1,
                                                      "negative_total_thickness": 1})
                self.assertEqual(r["n_rows_with_issues"], 1)
                self.assertEqual({i["field"] for i in r["issues"]}, {"target", "derivation.thickness"})
        for token in ("0", "-0.0", "-0e-200"):
            candidate = replace(row, target=token, derivation=replace(row.derivation, thickness=token))
            r = preview_ald_contract([candidate], contract=contract(),
                                     check_nonnegative_derived_thickness=True)
            self.assertEqual(r["issue_counts"], {})
        malformed = replace(row, target="-1", target_unit="not-a-growth-unit",
                            derivation=replace(row.derivation, thickness="-1", thickness_unit="not-a-length"))
        self.assertEqual(preview_ald_contract([malformed], contract=contract(),
                                             check_nonnegative_derived_thickness=True)["issue_counts"],
                         {"unsupported_growth_unit": 1, "unsupported_thickness_unit": 1})
        slope = replace(row, target_method="steady_growth_slope", target="-0.1", derivation=None)
        self.assertEqual(preview_ald_contract([slope], contract=contract(target_method="steady_growth_slope"),
                                             check_nonnegative_derived_thickness=True)["issue_counts"], {})
        for switch in (None, 0, 1, "true"):
            with self.subTest(switch=switch), self.assertRaisesRegex(ValueError, "explicit bool"):
                preview_ald_contract([row], contract=contract(), check_nonnegative_derived_thickness=switch)

    def test_steady_slope_is_not_proved_by_average_thickness(self):
        declared = contract(target_method="steady_growth_slope")
        clean = observation(target_method="steady_growth_slope", target="-0.1", derivation=None)
        self.assertEqual(preview_ald_contract([clean], contract=declared)["issue_count"], 0)
        wrong = replace(clean, derivation=observation().derivation)
        self.assertIn("derivation_not_applicable_to_method", codes(preview_ald_contract([wrong], contract=declared)))
        self.assertIn("declared_role_or_method_mismatch", codes(preview_ald_contract([observation()], contract=declared)))

    def test_reported_unspecified_temperature_basis_remains_unknown_even_when_expected(self):
        report = preview_ald_contract([observation(temperature_basis="reported_unspecified")],
                                      contract=contract(temperature_basis="reported_unspecified"))
        self.assertIn("unknown_temperature_basis", codes(report))
        self.assertEqual(report["status"], "declaration_issues_found")

    def test_nonexperimental_rows_and_limited_source_scope_are_flagged(self):
        for kind in ("simulation", "theory", "unknown", "invented_type", None, []):
            result = preview_ald_contract([observation(observation_kind=kind)], contract=contract())
            self.assertTrue(codes(result) & {"nonexperimental_or_unknown_observation", "unsupported_observation_kind"})
        for scope in ("abstract", "snippet", "unknown", "not_a_scope", None, []):
            result = preview_ald_contract([observation(evidence_scope=scope)], contract=contract())
            self.assertTrue(codes(result) & {"limited_or_unknown_source_scope", "unsupported_evidence_scope"})

    def test_missing_derivation_and_cross_source_recipe_fail(self):
        row = observation()
        for recipe in (None, {}, "guess"):
            self.assertIn("missing_or_invalid_derivation", codes(preview_ald_contract([replace(row, derivation=recipe)], contract=contract())))
        for field in ("source_id", "source_version", "condition_id"):
            recipe = replace(row.derivation, **{field: "different"})
            self.assertIn("derivation_identity_mismatch", codes(preview_ald_contract([replace(row, derivation=recipe)], contract=contract())))
        # A different explicit locator within the same source/version/condition is allowed.
        other_page = replace(row, derivation=replace(row.derivation, source_locator="synthetic figure 2"))
        self.assertEqual(preview_ald_contract([other_page], contract=contract())["issue_count"], 0)

    def test_tolerance_is_explicit_exact_and_context_independent(self):
        row = observation(target="0.49")
        self.assertIn("derived_target_arithmetic_mismatch", codes(preview_ald_contract([row], contract=contract())))
        self.assertEqual(preview_ald_contract([row], contract=contract(derived_target_tolerance_angstrom_per_cycle="0.01"))["issue_count"], 0)
        self.assertIn("derived_target_arithmetic_mismatch", codes(preview_ald_contract([row], contract=contract(derived_target_tolerance_angstrom_per_cycle="0.009999"))))
        with localcontext() as context:
            context.prec, context.Emax, context.Emin = 1, 1, -1
            self.assertEqual(preview_ald_contract([observation()], contract=contract())["issue_count"], 0)
        for value in ("-0.01", "NaN", "1e201", 0.0, Decimal("0"), None):
            with self.subTest(tolerance_type=type(value).__name__), self.assertRaises(ValueError):
                contract(derived_target_tolerance_angstrom_per_cycle=value)

    def test_strict_decimal_tokens_and_units(self):
        invalid = [None, True, 1, 1., Decimal("1"), "", " 1", "1 ", "NaN", "Infinity", "1e201", ".1e-200", "1" * 65, "１", "1,0"]
        for field in ("temperature", "target"):
            for value in invalid:
                with self.subTest(field=field, value_type=type(value).__name__):
                    result = preview_ald_contract([observation(**{field: value})], contract=contract())
                    self.assertIn("invalid_decimal_token", codes(result))
        for field, value, expected in (("temperature_unit", "C", "unsupported_temperature_unit"),
                                       ("target_unit", "A/cycle", "unsupported_growth_unit"),
                                       ("target_unit", [], "unsupported_growth_unit")):
            self.assertIn(expected, codes(preview_ald_contract([observation(**{field: value})], contract=contract())))
        row = observation()
        for value in invalid:
            result = preview_ald_contract([replace(row, derivation=replace(row.derivation, thickness=value))], contract=contract())
            self.assertIn("invalid_decimal_token", codes(result))
        self.assertIn("unsupported_thickness_unit", codes(preview_ald_contract(
            [replace(row, derivation=replace(row.derivation, thickness_unit="um"))], contract=contract())))

    def test_cycle_counts_are_bounded_positive_exact_integers(self):
        row = observation()
        for value in (True, False, 0, -1, 250., "250", None, MAX_CYCLES + 1, 10 ** 200):
            result = preview_ald_contract([replace(row, derivation=replace(row.derivation, cycle_count=value))], contract=contract())
            self.assertIn("invalid_cycle_count", codes(result))
        edge = replace(row, target="0.000000125", derivation=replace(row.derivation, cycle_count=MAX_CYCLES))
        self.assertEqual(preview_ald_contract([edge], contract=contract())["issue_count"], 0)

    def test_selected_target_postprocess_and_unknown_features_are_blocked(self):
        features = (
            ProcessFeatureDeclaration("growth", "process_setting", "before_process"),
            ProcessFeatureDeclaration("thickness", "postprocess_measurement", "before_process"),
            ProcessFeatureDeclaration("composition", "process_setting", "after_process"),
            ProcessFeatureDeclaration("target_formula", "computed_target", "before_process"),
            ProcessFeatureDeclaration("sample", "identifier", "before_process"),
            ProcessFeatureDeclaration("unknown", "unknown", "unknown"),
        )
        result = preview_ald_contract([observation()], contract=contract(features=features))
        self.assertIn("target_selected_as_feature", codes(result))
        self.assertEqual(result["issue_counts"]["feature_role_not_preprocess"], 4)
        self.assertEqual(result["issue_counts"]["feature_not_available_before_process"], 2)
        self.assertEqual(result["n_rows_with_issues"], 0)
        self.assertEqual(result["n_feature_declarations_with_issues"], 6)
        self.assertEqual(result["status"], "declaration_issues_found")

    def test_unknown_provenance_and_independence_are_not_filled(self):
        for value in (None, "unknown", "UNKNOWN", "", " padded", "bad\0text", "bad\ud800text", 0):
            result = preview_ald_contract([observation(source_id=value)], contract=contract())
            self.assertIn("missing_or_invalid_provenance", codes(result))
        row = observation(batch_id=None, replicate_id=None)
        result = preview_ald_contract([row], contract=contract())
        self.assertEqual(result["issue_counts"]["unknown_independence_metadata"], 2)
        self.assertIsNone(row.batch_id)
        self.assertIsNone(row.replicate_id)
        self.assertIn("invalid_independence_metadata", codes(preview_ald_contract([observation(batch_id=0)], contract=contract())))

    def test_duplicate_declared_identity_ignores_version_and_locator_but_scopes_replicates_by_batch(self):
        first = observation()
        second = replace(first, source_version="v2", source_locator="synthetic supplementary table",
                         derivation=replace(first.derivation, source_version="v2"))
        result = preview_ald_contract([first, second], contract=contract())
        self.assertEqual(result["issue_counts"]["repeated_declared_observation_identity"], 2)
        self.assertEqual(result["n_rows_with_issues"], 2)
        self.assertEqual(preview_ald_contract([first, replace(first, batch_id="batch-2")], contract=contract())["issue_count"], 0)
        self.assertEqual(preview_ald_contract([first, replace(first, replicate_id="sample-2")], contract=contract())["issue_count"], 0)

    def test_recognized_unknown_independence_keys_share_a_diagnostic_bucket(self):
        first = observation(batch_id=None, replicate_id=None)
        rows = [first, replace(first, batch_id="Unreported", replicate_id="unknown"),
                replace(first, batch_id="not_reported", replicate_id="UNREPORTED"),
                replace(first, batch_id="batch-2"), replace(first, replicate_id="sample-2"),
                replace(first, batch_id="", replicate_id="unknown")]
        before = deepcopy(rows)
        result = preview_ald_contract(rows, contract=contract())
        repeated = [issue["row_position"] for issue in result["issues"]
                    if issue["code"] == "repeated_declared_observation_identity"]
        self.assertEqual(repeated, [0, 1, 2])
        self.assertEqual(result["issue_counts"]["unknown_independence_metadata"], 9)
        self.assertEqual(result["issue_counts"]["invalid_independence_metadata"], 1)
        self.assertEqual(result["training_admission"], "not_performed")
        self.assertEqual(rows, before)

    def test_reports_do_not_echo_values_identifiers_or_feature_names(self):
        private = "SYNTHETIC_PRIVATE_SENTINEL"
        row = observation(source_id=private, source_locator=private, temperature=private,
                          target=private, batch_id=private, replicate_id=private)
        report = preview_ald_contract([row], contract=contract(features=(ProcessFeatureDeclaration(private, "computed_target", "after_process"),)))
        self.assertNotIn(private, json.dumps(report, allow_nan=False))
        self.assertEqual(report["issues"][0]["feature_position"], 0)
        self.assertTrue(any(issue["row_position"] == 0 for issue in report["issues"]))

    def test_no_mutation_and_frozen_contracts_are_rechecked(self):
        rows = [observation()]
        before = deepcopy(rows)
        preview_ald_contract(rows, contract=contract())
        self.assertEqual(rows, before)
        declared = contract()
        object.__setattr__(declared, "target_method", "simulation")
        object.__setattr__(declared, "__post_init__", lambda: None)
        with self.assertRaises(ValueError):
            preview_ald_contract(rows, contract=declared)
        row = observation()
        object.__setattr__(row, "target", "NaN")
        self.assertIn("invalid_decimal_token", codes(preview_ald_contract([row], contract=contract())))

    def test_input_and_feature_bounds_do_not_consume_arbitrary_iterators(self):
        for rows in ([], (), iter([observation()]), "row", [object()], [observation()] * (MAX_RECORDS + 1)):
            with self.assertRaises(ValueError):
                preview_ald_contract(rows, contract=contract())
        with self.assertRaises(ValueError):
            preview_ald_contract([observation()], contract={})
        for features in ([], [ProcessFeatureDeclaration("x", "process_setting", "before_process")],
                         (ProcessFeatureDeclaration("x", "process_setting", "before_process"),) * 2,
                         tuple(ProcessFeatureDeclaration(f"x{i}", "process_setting", "before_process") for i in range(101))):
            with self.assertRaises(ValueError):
                contract(features=features)

    def test_detail_cap_does_not_hide_later_conflicts_or_incomplete_status(self):
        rows = [observation() for _ in range(MAX_ISSUES + 2)]
        rows[-1] = replace(rows[-1], target="wrong")
        report = preview_ald_contract(rows, contract=contract())
        self.assertEqual(len(report["issues"]), MAX_ISSUES)
        self.assertTrue(report["issues_truncated"])
        self.assertGreater(report["issue_count"], MAX_ISSUES)
        self.assertEqual(report["n_rows_with_issues"], len(rows))
        self.assertIn("invalid_decimal_token", report["issue_counts"])

    def test_no_physical_plausibility_or_declared_source_truth_is_inferred(self):
        # Explicitly outside this helper's scope: a numerically well-formed
        # impossible temperature still needs a physical/source review.
        report = preview_ald_contract([observation(temperature="-1", temperature_unit="K")], contract=contract())
        self.assertEqual(report["issue_count"], 0)
        self.assertEqual(report["training_admission"], "not_performed")

    def test_standard_library_only_no_model_imports(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root / "src"))
        code = "import sys; import groupbench.process_contract; assert not ({'numpy','pandas','sklearn'} & set(sys.modules))"
        completed = subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_synthetic_example_and_report_shape(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root / "src"))
        completed = subprocess.run([sys.executable, "-S", str(root / "examples/ald-contract-preview/preview.py")],
                                   env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        example = json.loads(completed.stdout)
        self.assertEqual(example["fixture"], "invented_not_measured")
        self.assertEqual(example["model_fits"], 0)
        self.assertEqual(len(example["reports"]), 6)
        expected_keys = {"schema_version", "kind", "action", "training_admission", "status", "n_rows",
                         "n_selected_features", "n_rows_with_issues", "n_feature_declarations_with_issues",
                         "issue_count", "issue_counts", "issues", "issues_truncated", "max_returned_issues", "limitations"}
        for report in example["reports"].values():
            self.assertEqual(set(report), expected_keys)
            for issue in report["issues"]:
                self.assertEqual(set(issue), {"code", "field", "row_position", "feature_position"})



class TwoFactorDesignSupportTests(unittest.TestCase):
    def test_replication_does_not_connect_a_disconnected_design(self):
        from groupbench.process_contract import audit_two_factor_additive_design

        rows = [('same', 'same'), ('b', 'high'), ('b', 'high')]
        report = audit_two_factor_additive_design(
            rows, a_levels=['same', 'b', 'unused'], b_levels=['same', 'high'])
        self.assertEqual((report.n_records, report.n_distinct_cells), (3, 2))
        self.assertEqual((report.design_rank, report.full_design_rank), (2, 4))
        self.assertEqual((report.within_cell_residual_df, report.distinct_cell_additivity_df), (1, 0))
        self.assertEqual(report.components, (((0,), (0,)), ((1,), (1,)), ((2,), ())))
        self.assertEqual(report.cell_counts, ((0, 0, 1), (1, 1, 2)))
        self.assertEqual(report, audit_two_factor_additive_design(
            rows[::-1], a_levels=['same', 'b', 'unused'], b_levels=['same', 'high']))

    def test_distinct_cells_and_repeats_contribute_different_residual_df(self):
        from groupbench.process_contract import audit_two_factor_additive_design

        levels = ('low', 'high')
        square = [(a, b) for a in levels for b in levels]
        tree = audit_two_factor_additive_design(
            square[:-1] * 10, a_levels=levels, b_levels=levels)
        cycle = audit_two_factor_additive_design(
            square + [square[0]] * 2, a_levels=levels, b_levels=levels)
        self.assertEqual((tree.design_rank, tree.distinct_cell_additivity_df,
                          tree.within_cell_residual_df), (3, 0, 27))
        self.assertEqual((cycle.design_rank, cycle.distinct_cell_additivity_df,
                          cycle.within_cell_residual_df), (3, 1, 2))

    def test_exact_graph_rank_matches_independent_matrix_elimination(self):
        from fractions import Fraction
        from groupbench.process_contract import audit_two_factor_additive_design

        a_levels, b_levels = ('a0', 'a1', 'a2'), ('b0', 'b1', 'b2')
        grid = [(a, b) for a in a_levels for b in b_levels]
        for mask in range(1, 1 << len(grid)):
            rows = [pair for index, pair in enumerate(grid) if mask & (1 << index)]
            matrix = [[Fraction(left == level) for level in a_levels] +
                      [Fraction(right == level) for level in b_levels] for left, right in rows]
            pivot = 0
            for column in range(len(a_levels) + len(b_levels)):
                pick = next((i for i in range(pivot, len(rows)) if matrix[i][column]), None)
                if pick is None:
                    continue
                matrix[pivot], matrix[pick] = matrix[pick], matrix[pivot]
                for row in matrix[pivot + 1:]:
                    ratio = row[column] / matrix[pivot][column]
                    row[:] = [x - ratio * y for x, y in zip(row, matrix[pivot])]
                pivot += 1
            audit = audit_two_factor_additive_design(rows, a_levels=a_levels, b_levels=b_levels)
            self.assertEqual(audit.design_rank, pivot, mask)
            self.assertEqual(audit.distinct_cell_additivity_df, len(rows) - pivot, mask)

    def test_rejects_undeclared_or_ambiguous_records_and_levels(self):
        from groupbench.process_contract import audit_two_factor_additive_design

        for levels in ([], ['a', 'a'], ['unknown'], {'a'}, [' a'], ['a\0'], [1]):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                audit_two_factor_additive_design([('a', 'b')], a_levels=levels, b_levels=['b'])
        for rows in ([], [('a', 'b', 'response')], [('A', 'b')], [('a', [])],
                     {'a': 'b'}, [('a', 'b')] * 20_001):
            with self.assertRaises(ValueError):
                audit_two_factor_additive_design(rows, a_levels=['a'], b_levels=['b'])

if __name__ == "__main__":
    unittest.main()
