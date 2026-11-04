"""Invented metadata only; no research quantity values or model fits."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from groupbench.process_contract import (
    MAX_ISSUES, MAX_RECORDS, MAX_QUARANTINE_SCOPES, MAX_SCOPE_REFERENCES,
    ObservationEvidenceKey as Key, QuarantineScope as Scope, preview_evidence_quarantine,
)


def preview(rows, scopes):
    return preview_evidence_quarantine(rows, unresolved_scopes=scopes)


class EvidenceQuarantineTests(unittest.TestCase):
    def setUp(self):
        self.blocked = Key("invented-source", "recipe-2", "peak_ratio")
        self.other = replace(self.blocked, endpoint_name="peak_area")
        self.scope = Scope("invented-source", ("recipe-2",), ("peak_ratio",))

    def test_exact_quantity_local_isolation_and_positive_mutation(self):
        clean = preview([self.other], [self.scope])
        self.assertEqual(clean["status"], "no_declared_quarantine_matches")
        bad = preview([self.other, self.blocked], [self.scope])
        self.assertEqual(bad["status"], "quarantine_violation_found")
        self.assertEqual(bad["n_rows_with_quarantine_matches"], 1)
        self.assertEqual(bad["issues"][0]["row_position"], 1)
        self.assertEqual(bad["training_admission"], "not_performed")

    def test_source_group_and_endpoint_are_all_required_without_alias_guessing(self):
        rows = [replace(self.blocked, source_id="other-source"),
                replace(self.blocked, recipe_group_id="recipe-3"), self.other,
                replace(self.blocked, endpoint_name="Peak_Ratio"),
                replace(self.blocked, endpoint_name="peak_ratio_SD")]
        self.assertEqual(preview(rows, [self.scope])["issue_count"], 0)

    def test_multiple_groups_and_endpoints_have_explicit_rectangular_scope(self):
        scope = Scope("invented-source", ("recipe-1", "recipe-2"), ("peak_ratio", "other_ratio"))
        rows = [Key(scope.source_id, group, endpoint) for group in scope.recipe_group_ids
                for endpoint in scope.endpoint_names] + [self.other]
        report = preview(rows, [scope])
        self.assertEqual(report["n_rows_with_quarantine_matches"], 4)
        self.assertEqual(report["issue_count"], 4)
        # Separate declarations express a nonrectangular mapping.
        narrow = [Scope(scope.source_id, ("recipe-1",), ("peak_ratio",)),
                  Scope(scope.source_id, ("recipe-2",), ("other_ratio",))]
        self.assertEqual(preview(rows, narrow)["n_rows_with_quarantine_matches"], 2)

    def test_unknown_endpoint_mapping_never_becomes_a_clean_result(self):
        scope = replace(self.scope, endpoint_names=None)
        report = preview([self.other], [scope])
        self.assertEqual(report["status"], "mapping_not_established")
        self.assertEqual(report["n_unmapped_scopes"], 1)
        self.assertEqual(report["n_rows_with_quarantine_matches"], 0)
        self.assertEqual(report["n_rows_in_unmapped_scope_groups"], 1)
        self.assertIsNone(report["issues"][0]["row_position"])
        self.assertEqual(preview([], [scope])["status"], "mapping_not_established")
        self.assertEqual(preview([replace(self.other, source_id="elsewhere")], [scope])["status"], "mapping_not_established")

    def test_known_violation_and_unknown_mapping_are_both_retained(self):
        report = preview([self.blocked], [self.scope, replace(self.scope, endpoint_names=None)])
        self.assertEqual(report["status"], "quarantine_violation_found")
        self.assertEqual(report["n_unmapped_scopes"], 1)
        self.assertEqual(report["issue_counts"], {"quarantined_observation_present": 1,
                                                 "scope_endpoint_mapping_not_established": 1})

    def test_explicit_empty_registry_is_not_claimed_complete(self):
        report = preview([self.blocked], [])
        self.assertEqual(report["n_unresolved_scopes"], 0)
        self.assertEqual(report["status"], "no_declared_quarantine_matches")
        self.assertIn("registry completeness", " ".join(report["limitations"]))
        self.assertEqual(preview([], [self.scope])["n_observation_rows"], 0)
        with self.assertRaises(TypeError):
            preview_evidence_quarantine([self.blocked])

    def test_duplicate_observations_and_overlapping_scopes_do_not_inflate_unique_row_count(self):
        report = preview([self.blocked, self.blocked], [self.scope, self.scope])
        self.assertEqual(report["n_observation_rows"], 2)
        self.assertEqual(report["n_rows_with_quarantine_matches"], 2)
        self.assertEqual(report["issue_count"], 4)  # Declared scope/row matches, not specimens.

    def test_reordering_preserves_counts_and_uses_current_row_positions(self):
        a, b = preview([self.other, self.blocked], [self.scope]), preview([self.blocked, self.other], [self.scope])
        for field in ("status", "n_rows_with_quarantine_matches", "issue_count", "issue_counts"):
            self.assertEqual(a[field], b[field])
        self.assertEqual(a["issues"][0]["row_position"], 1)
        self.assertEqual(b["issues"][0]["row_position"], 0)

    def test_capped_details_retain_all_counts_and_later_mapping_uncertainty(self):
        rows = [self.blocked] * (MAX_ISSUES + 2)
        report = preview(rows, [self.scope, replace(self.scope, endpoint_names=None)])
        self.assertEqual(len(report["issues"]), MAX_ISSUES)
        self.assertTrue(report["issues_truncated"])
        self.assertEqual(report["n_rows_with_quarantine_matches"], len(rows))
        self.assertEqual(report["n_rows_in_unmapped_scope_groups"], len(rows))
        self.assertEqual(report["issue_count"], len(rows) + 1)
        self.assertEqual(report["n_unmapped_scopes"], 1)

    def test_no_input_mutation_and_no_private_identity_echo(self):
        sentinel = "INVENTED_PRIVATE_SENTINEL"
        rows = [Key(sentinel, sentinel, sentinel)]
        scopes = [Scope(sentinel, (sentinel,), (sentinel,))]
        before = deepcopy((rows, scopes))
        report = preview(rows, scopes)
        self.assertEqual((rows, scopes), before)
        self.assertNotIn(sentinel, json.dumps(report, allow_nan=False))
        invalid = replace(rows[0], source_id=" " + sentinel)
        with self.assertRaises(ValueError) as caught:
            preview([invalid], scopes)
        self.assertNotIn(sentinel, str(caught.exception))

    def test_exact_containers_and_record_types_are_required(self):
        class SubKey(Key):
            pass
        class SubScope(Scope):
            pass
        for rows in (None, iter([self.blocked]), {"row": self.blocked}, [object()],
                     [SubKey("source", "group", "endpoint")], [self.blocked] * (MAX_RECORDS + 1)):
            with self.assertRaises(ValueError):
                preview(rows, [self.scope])
        for scopes in (None, iter([self.scope]), {"scope": self.scope}, [object()],
                       [SubScope("source", ("group",), ("endpoint",))], [self.scope] * (MAX_QUARANTINE_SCOPES + 1)):
            with self.assertRaises(ValueError):
                preview([self.blocked], scopes)

    def test_unknown_or_malformed_identity_is_not_coerced(self):
        bad = (None, 0, True, [], {}, "", "unknown", "UNKNOWN", "unreported", "not_reported", " x", "x ", "a\0b", "a\ud800b", "x" * 257)
        for value in bad:
            for field in ("source_id", "recipe_group_id", "endpoint_name"):
                with self.subTest(field=field, value_type=type(value).__name__), self.assertRaises(ValueError):
                    preview([replace(self.blocked, **{field: value})], [self.scope])
            with self.assertRaises(ValueError):
                preview([self.blocked], [replace(self.scope, source_id=value)])

    def test_scope_lists_require_nonempty_unique_explicit_identity_tuples(self):
        for field in ("recipe_group_ids", "endpoint_names"):
            for value in ((), [], ["x"], "x", ("x", "x"), (None,), ("unknown",), (1,), ({},)):
                with self.subTest(field=field, value_type=type(value).__name__), self.assertRaises(ValueError):
                    preview([self.blocked], [replace(self.scope, **{field: value})])
        # None is allowed only for endpoint mapping, even when both fields are None.
        for endpoints in (("peak_ratio",), None):
            with self.assertRaises(ValueError):
                preview([self.blocked], [Scope("invented-source", None, endpoints)])

    def test_total_scope_reference_bound_and_sparse_cartesian_case(self):
        groups = tuple(f"g{i}" for i in range(MAX_SCOPE_REFERENCES // 2))
        endpoints = tuple(f"e{i}" for i in range(MAX_SCOPE_REFERENCES // 2))
        # 100 million possible pairs are not constructed; only one observed key.
        scope = Scope("source", groups, endpoints)
        self.assertEqual(preview([Key("source", groups[-1], endpoints[-1])], [scope])["issue_count"], 1)
        with self.assertRaises(ValueError):
            preview([], [scope, Scope("source", ("g",), ("e",))])

    def test_frozen_declarations_are_rechecked(self):
        row, scope = deepcopy(self.blocked), deepcopy(self.scope)
        object.__setattr__(row, "endpoint_name", None)
        with self.assertRaises(ValueError):
            preview([row], [scope])
        object.__setattr__(scope, "recipe_group_ids", None)
        with self.assertRaises(ValueError):
            preview([self.blocked], [scope])

    def test_missing_fields_and_uninitialized_records_raise_fixed_value_errors(self):
        for field in ("source_id", "recipe_group_id", "endpoint_name"):
            row = deepcopy(self.blocked)
            object.__delattr__(row, field)
            with self.subTest(record="observation", field=field), self.assertRaises(ValueError):
                preview([row], [self.scope])
        for field in ("source_id", "recipe_group_ids", "endpoint_names"):
            scope = deepcopy(self.scope)
            object.__delattr__(scope, field)
            with self.subTest(record="scope", field=field), self.assertRaises(ValueError):
                preview([self.blocked], [scope])
        with self.assertRaises(ValueError):
            preview([object.__new__(Key)], [self.scope])
        with self.assertRaises(ValueError):
            preview([self.blocked], [object.__new__(Scope)])

    def test_standard_library_only_and_fixed_report_schema(self):
        root = Path(__file__).resolve().parents[1]
        code = "import sys; from groupbench.process_contract import preview_evidence_quarantine; assert not ({'numpy','pandas','sklearn'} & set(sys.modules)); assert preview_evidence_quarantine([], unresolved_scopes=[])['training_admission'] == 'not_performed'"
        completed = subprocess.run([sys.executable, "-S", "-c", code], env=dict(os.environ, PYTHONPATH=str(root / "src")), capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = preview([self.blocked], [self.scope])
        self.assertEqual(set(report), {"schema_version", "kind", "action", "training_admission", "status",
            "n_observation_rows", "n_unresolved_scopes", "n_unmapped_scopes", "n_rows_with_quarantine_matches",
            "n_rows_in_unmapped_scope_groups", "issue_count", "issue_counts", "issues", "issues_truncated",
            "max_returned_issues", "limitations"})
        self.assertEqual(set(report["issues"][0]), {"code", "field", "scope_position", "row_position"})

    def test_invented_example_runs_without_site_packages(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run([sys.executable, "-S", str(root / "examples/evidence-quarantine/preview.py")],
            env=dict(os.environ, PYTHONPATH=str(root / "src")), capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        example = json.loads(completed.stdout)
        self.assertEqual(example["fixture"], "invented_metadata_no_measurements")
        self.assertEqual(example["model_fits"], 0)
        self.assertEqual(len(example["reports"]), 4)


if __name__ == "__main__":
    unittest.main()
