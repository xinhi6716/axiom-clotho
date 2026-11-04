"""No-fit conformance tests for retained, partial temporal result bundles.

All results are explicitly invented software fixtures. No evaluator, preview,
source observer, estimator, RNG, external dataset, or scientific run builds
these records. Independent scalar arithmetic supplies the saved metric oracle.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import builtins
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import pickle
import random
import socket
import struct
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import groupbench
import groupbench.group_time_evaluation as temporal
import groupbench.group_time_saved as saved
import groupbench.model as model
import groupbench.package_source as package_source
import groupbench.report as report
import groupbench.verification as verification
from groupbench.group_time_json import _document
from groupbench.group_time_plan import (
    MAX_TICK, MIN_TICK, GroupSpan, TimeWindow, plan_group_forward_windows,
)
from groupbench.group_time_saved import (
    GroupForwardBundleError, GroupForwardBundleValidation,
    validate_group_forward_evaluation_bundle as validate,
    write_group_forward_evaluation_bundle as write,
)


BUNDLE_ID = "mlcalc.group_forward_baseline_bundle"
RESULT_ID = "mlcalc.group_forward_baseline_evaluation"
RECIPES = ("dummy_mean", "ridge_alpha_1")
# A literal historical inventory, not current SOURCE_FILES minus an omission.
HISTORIC_37_FILES = (
    "__init__.py", "__main__.py", "baselines.py", "binary/__init__.py", "binary/cli.py",
    "binary/data.py", "binary/metrics.py", "binary/model.py", "binary/report.py",
    "binary/saved_evaluation.py", "binary/workflow.py", "budget_comparison.py",
    "categorical.py", "cli.py", "data.py", "domain_selection.py", "evaluation.py",
    "evaluation_schema.py", "feature_manifest.py", "group_time_evaluation.py",
    "group_time_json.py", "group_time_plan.py", "input_support.py", "interval_evaluation.py",
    "kernel.py", "metrics.py", "model.py", "observed_history.py", "package_source.py",
    "planning.py", "prediction.py", "prediction_schema.py", "quantity_contract.py",
    "report.py", "saved_scoring.py", "splitting.py", "verification.py",
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def envelope(document):
    return {"schema_id": BUNDLE_ID, "schema_version": 1, "result": document}


def source_declaration(names=("invented/not_installed.py",)):
    contract = {"schema_version": "1.0", "algorithm": "sha256",
                "scope": "fixed_groupbench_python_sources_v1"}
    files = {name: hashlib.sha256(name.encode("utf-8")).hexdigest() for name in names}
    # This is the source identity's specified encoding, independent of observer.
    raw = json.dumps({**contract, "files": files}, sort_keys=True,
                     separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return {**contract, "status": "available", "sha256": hashlib.sha256(raw).hexdigest(),
            "files": files}


def scalar_scores(rows, prediction_key):
    """Tiny independent arithmetic; deliberately does not call package metrics."""
    observed = [row["observed"] for row in rows]
    residual = [row[prediction_key] - row["observed"] for row in rows]
    count = len(rows)
    squares = sum(value * value for value in residual)
    center = sum(observed) / count
    total = sum((value - center) ** 2 for value in observed)
    return {"n_rows": count, "mae": float(sum(map(abs, residual)) / count),
            "rmse": float(math.sqrt(squares / count)),
            "r2": float(1.0 - squares / total) if total else None,
            "bias": float(sum(residual) / count)}


def scalar_grouped(rows, prediction_key):
    identities = list(dict.fromkeys((type(row["group"]), row["group"]) for row in rows))
    groups = [{"group": identity[1], **scalar_scores(
        [row for row in rows if (type(row["group"]), row["group"]) == identity],
        prediction_key)} for identity in identities]
    pooled = scalar_scores(rows, prediction_key)
    pooled.update(n_groups=len(groups),
                  group_mae=float(sum(row["mae"] for row in groups) / len(groups)),
                  group_rmse=float(sum(row["rmse"] for row in groups) / len(groups)),
                  worst_group_mae=float(max(row["mae"] for row in groups)))
    return pooled, groups


def refresh_metrics(document):
    result = []
    rows = document["predictions"]
    for recipe in RECIPES:
        key = "prediction_" + recipe
        pooled, groups = scalar_grouped(rows, key)
        windows = []
        for fold in document["plan"]["folds"]:
            scores, _ = scalar_grouped(
                [row for row in rows if row["window_index"] == fold["index"]], key)
            windows.append({"window_index": fold["index"], **scores})
        result.append({"recipe": recipe, "pooled": pooled,
                       "per_window": windows, "per_group": groups})
    document["metrics"] = result


def rebind(document):
    bindings = document["bindings"]
    bindings["plan_sha256"] = temporal._digest("plan", document["plan"])
    bindings["execution_sha256"] = temporal._digest("execution", {
        "plan_sha256": bindings["plan_sha256"], "input_sha256": bindings["input_sha256"],
        "protocol": document["protocol"], "coverage_mode": document["coverage_mode"],
        "recipes": document["recipes"], "environment": document["environment"],
        "resource_limits": document["fit_budget"]["resource_limits"],
    })


def invented_document(plan=None, *, zero=False):
    if plan is None:
        groups = [1, "1", "a", "warm2", "b", "cross", 1, "a", "late"]
        plan = plan_group_forward_windows(
            groups, group_spans=[
                GroupSpan(1, 0, 1, 1), GroupSpan("warm2", 2, 3, 3),
                GroupSpan("1", 10, 10, 12), GroupSpan("a", 11, 12, 12),
                GroupSpan("b", 20, 21, 21), GroupSpan("cross", 8, 12, 12),
                GroupSpan("late", 40, 41, 41)],
            windows=[TimeWindow(10, 15), TimeWindow(20, 25)],
            time_unit="day", clock_id="invented-clock", gap=2)
    count = 2 * len(plan.folds)
    cells = 2 * sum(fold.train.n_rows + fold.test.n_rows for fold in plan.folds)
    observations = {1: 3.0, 2: 4.0, 4: 5.0, 7: 6.0}
    ridge = {1: 2.5, 2: 4.0, 4: 4.5, 7: 5.0}
    windows = {row: fold.index for fold in plan.folds for row in fold.test.rows}
    rows = [{"row_position": row, "group": plan.row_groups[row],
             "window_index": windows[row], "observed": -0.0 if zero else observations[row],
             "prediction_dummy_mean": -0.0 if zero else (2.0 if windows[row] == 0 else 4.0),
             "prediction_ridge_alpha_1": -0.0 if zero else ridge[row]}
            for row in plan.tested.rows]
    document = {
        "schema_id": RESULT_ID, "schema_version": 1, "protocol": temporal.PROTOCOL,
        "coverage_mode": temporal.COVERAGE_MODE,
        "bindings": {"encoding": "mlcalc.temporal-binding-v1", "plan_sha256": "0" * 64,
                     "input_sha256": "a" * 64, "execution_sha256": "0" * 64},
        "plan": _document(plan),
        "feature_schema": [{"name": "invented_feature", "source_dtype": "float64",
                            "normalized_dtype": "float64"}],
        "metadata_columns": ["group", "start", "end", "label_available_at", "row_key"],
        "target_name": "outcome", "group_name": "group", "recipes": temporal._recipes(),
        "fit_budget": {"unit": "pipeline.fit calls", "expected_total": count,
                       "attempted_total": count, "completed_total": count,
                       "per_window": [{"window_index": fold.index, "expected": 2,
                                       "attempted": 2, "completed": 2} for fold in plan.folds],
                       "inner_selection_fits": 0, "full_data_refits": 0,
                       "selected_model_refits": 0, "bootstraps": 0, "retries": 0,
                       "resource_limits": temporal._limits(cells)},
        "coverage": temporal._coverage(plan), "predictions": rows, "metrics": [],
        "environment": {"package_version": "invented-version", "package_source": source_declaration(),
                        "python": "invented-python", "numpy": "invented-numpy",
                        "pandas": "invented-pandas", "sklearn": "invented-sklearn",
                        "scipy": "invented-scipy", "threadpoolctl": "invented-threadpoolctl"},
        "limitations": list(temporal._LIMITATIONS),
    }
    refresh_metrics(document)
    rebind(document)
    return document


def exact_result(document):
    return temporal.GroupForwardEvaluationResult(
        **{key: temporal._freeze(value) for key, value in document.items()})


def typed_document():
    ids = [1, "1", "01", MIN_TICK, MAX_TICK, "  padded  ", "\u00e9", "e\u0301",
           "nul\x00id", "surrogate\ud800", 1]
    groups = ["warmup", *ids]
    distinct = list(dict.fromkeys(ids))
    plan = plan_group_forward_windows(
        groups, group_spans=[GroupSpan("warmup", -2, -1, -1),
                             *(GroupSpan(group, 10, 10, 10) for group in distinct)],
        windows=[TimeWindow(10, 11)], time_unit="tick", clock_id="invented-typed")
    return invented_document(plan, zero=True)


def put(path, keys, value):
    for key in keys[:-1]:
        path = path[key]
    path[keys[-1]] = value


def node_count(value):
    if type(value) is dict:
        return 1 + sum(node_count(item) for item in value.values())
    if type(value) is list:
        return 1 + sum(node_count(item) for item in value)
    return 1


def container_depth(value):
    if type(value) is dict:
        return 1 + max((container_depth(item) for item in value.values()), default=0)
    if type(value) is list:
        return 1 + max((container_depth(item) for item in value), default=0)
    return 0


class SavedFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Every test, including fixture creation, is under the no-fit traps.
        for owner, name in ((temporal, "_prepare"), (temporal, "_validate_result"),
                            (temporal, "preview_group_forward_evaluation"),
                            (temporal, "evaluate_group_forward_baselines"),
                            (temporal, "make_pipeline"), (model, "make_pipeline"),
                            (Pipeline, "fit"), (Pipeline, "predict"),
                            (temporal, "observe_package_source"),
                            (package_source, "observe_package_source")):
            self.stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
        self.temp = tempfile.TemporaryDirectory(prefix="temporal-no-fit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.document = invented_document()
        self.result = exact_result(self.document)
        self.sequence = 0

    def fresh(self, label="bundle"):
        self.sequence += 1
        return self.root / f"{label}-{self.sequence}"

    def install(self, document=None, *, raw=None):
        directory = self.fresh("raw")
        directory.mkdir()
        payload = canonical(envelope(self.document if document is None else document)) if raw is None else raw
        (directory / "temporal_result.json").write_bytes(payload)
        (directory / "checksums.json").write_bytes(canonical({
            "algorithm": "sha256", "files": {"temporal_result.json": hashlib.sha256(payload).hexdigest()}}))
        return directory

    def rejected_document(self, document, *, both=True):
        directory = self.install(document)
        self.assertEqual(verification.verify_bundle(directory).bundle_type, "temporal_evaluation")
        with self.assertRaises(GroupForwardBundleError):
            validate(directory)
        if both and document.keys() == self.document.keys():
            target = self.fresh("invalid")
            with self.assertRaises(GroupForwardBundleError):
                write(exact_result(document), target)
            self.assertFalse(target.exists())

    def mutate_reject(self, path, value, *, bind=False):
        document = deepcopy(self.document)
        put(document, path, value)
        if bind:
            rebind(document)
        self.rejected_document(document)


class TypedRoundTripAndPartialPlanTests(SavedFixtureTestCase):
    """Acceptance groups 1 and 2: identities and complete partial coverage."""
    def test_exact_typed_round_trip_preserves_signed_zero_and_input(self):
        document = typed_document()
        result = exact_result(document)
        detached = result.to_dict()
        detached_before = deepcopy(detached)
        directory = self.fresh()
        returned = write(result, directory)
        self.assertEqual(returned, directory / "temporal_result.json")
        raw = returned.read_bytes()
        self.assertEqual(raw, canonical(envelope(document)))
        self.assertIn(b"\\u0000", raw)
        self.assertIn(b"\\ud800", raw)
        decoded = json.loads(raw)["result"]
        self.assertTrue(temporal._same_typed(decoded, document))
        self.assertTrue(temporal._same_typed(result.to_dict(), document))
        self.assertTrue(temporal._same_typed(detached, detached_before))
        for row in decoded["predictions"]:
            for name in ("observed", "prediction_dummy_mean", "prediction_ridge_alpha_1"):
                self.assertEqual(struct.pack("<d", row[name]), struct.pack("<d", -0.0))
        ids = decoded["plan"]["row_groups"]
        self.assertEqual([(type(x), x) for x in ids],
                         [(type(x), x) for x in document["plan"]["row_groups"]])
        summary = validate(directory)
        self.assertIs(type(summary), GroupForwardBundleValidation)
        self.assertEqual(summary.n_groups, 11)
        self.assertEqual(summary.n_tested_groups, 10)
        self.assertEqual(summary.n_untested_rows, 1)
        self.assertEqual(summary.validation_fits_performed, 0)
        self.assertEqual(summary.declared_pipeline_fits, 2)
        self.assertEqual(summary.temporal_result_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(summary.payload_bytes, len(raw))
        self.assertIs(type(summary.checked_scope), tuple)
        self.assertIs(type(summary.unchecked_scope), tuple)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            summary.n_rows = 999
        detached["predictions"][0]["group"] = "changed-copy"
        self.assertNotEqual(result.to_dict()["predictions"][0]["group"], "changed-copy")

    def test_nine_row_partial_plan_and_later_training_are_retained(self):
        plan = self.document["plan"]
        self.assertEqual(plan["tested"]["rows"], [1, 2, 4, 7])
        self.assertEqual(plan["untested"]["rows"], [0, 3, 5, 6, 8])
        self.assertEqual(plan["folds"][0]["train"]["rows"], [0, 3, 6])
        self.assertEqual(plan["folds"][1]["train"]["rows"], [0, 1, 2, 3, 5, 6, 7])
        self.assertIn("cross", plan["folds"][0]["unused"]["groups"])
        self.assertIn("cross", plan["folds"][1]["train"]["groups"])
        self.assertNotIn("cross", plan["tested"]["groups"])
        directory = self.fresh()
        write(self.result, directory)
        summary = validate(directory)
        self.assertEqual((summary.n_rows, summary.n_groups, summary.n_tested_rows,
                          summary.n_tested_groups, summary.n_untested_rows,
                          summary.n_untested_groups, summary.n_windows, summary.n_features,
                          summary.n_recipes, summary.declared_pipeline_fits),
                         (9, 7, 4, 3, 5, 4, 2, 1, 2, 4))
        self.assertEqual((summary.bundle_schema_version, summary.result_schema_version), (1, 1))
        self.assertEqual(summary.coverage_mode, "partial_predeclared")
        self.assertEqual(summary.protocol, "group-forward-fixed-regression-v1")
        self.assertEqual(plan["fits_performed"], 0)
        self.assertEqual(self.document["fit_budget"]["resource_limits"]["matrix_cell_visits"], 28)

    def test_full_coverage_rewrite_cannot_turn_training_into_testing(self):
        document = deepcopy(self.document)
        document["plan"]["tested"] = deepcopy(document["plan"]["total"])
        document["plan"]["untested"] = {"rows": [], "groups": [], "n_rows": 0, "n_groups": 0}
        document["coverage"]["tested"] = deepcopy(document["plan"]["tested"])
        document["coverage"]["untested"] = deepcopy(document["plan"]["untested"])
        rebind(document)
        self.rejected_document(document)

    def test_singleton_test_window_and_constant_groups_have_null_r2(self):
        self.assertIsNone(self.document["metrics"][0]["per_window"][1]["r2"])
        self.assertIsNone(self.document["metrics"][1]["per_group"][0]["r2"])
        summary = validate(self.install())
        self.assertEqual(summary.n_windows, 2)
        typed = typed_document()
        for metric in typed["metrics"]:
            self.assertIsNone(metric["pooled"]["r2"])
            self.assertTrue(all(row["r2"] is None for row in metric["per_group"]))
        self.assertEqual(validate(self.install(typed)).n_tested_groups, 10)


class SchemaAndProjectionTests(SavedFixtureTestCase):
    """Acceptance group 3: closed schemas and exact immutable result input."""
    def test_envelope_identity_version_and_exact_keys(self):
        for key, value in (("schema_id", "wrong"), ("schema_version", True),
                           ("schema_version", 2), ("schema_version", 1.0), ("result", None)):
            with self.subTest(key=key, value=value):
                outer = envelope(deepcopy(self.document))
                outer[key] = value
                with self.assertRaises(GroupForwardBundleError):
                    validate(self.install(raw=canonical(outer)))
        for kind in ("missing", "extra"):
            outer = envelope(deepcopy(self.document))
            if kind == "missing":
                del outer["schema_id"]
            else:
                outer["extra"] = None
            with self.subTest(kind=kind), self.assertRaises(GroupForwardBundleError):
                validate(self.install(raw=canonical(outer)))

    def test_result_and_plan_markers_are_exact(self):
        cases = [(("schema_id",), "wrong"), (("schema_version",), True),
                 (("schema_version",), 2), (("protocol",), "full-oof"),
                 (("coverage_mode",), "complete"), (("plan", "schema_id"), "wrong"),
                 (("plan", "schema_version"), True), (("plan", "protocol"), "wrong"),
                 (("plan", "coverage_mode"), "complete"), (("plan", "gap"), True),
                 (("plan", "folds", 0, "window", "lower"), 10.0)]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                self.mutate_reject(path, value)

    def test_extra_missing_and_null_records_at_each_major_schema_level(self):
        locations = [(), ("bindings",), ("plan",), ("plan", "folds", 0),
                     ("plan", "folds", 0, "window"), ("plan", "group_spans", 0),
                     ("plan", "folds", 0, "exclusions", 0), ("plan", "total"),
                     ("coverage",), ("coverage", "tested"), ("feature_schema", 0),
                     ("recipes", 0), ("recipes", 0, "preprocessing", 0),
                     ("fit_budget",), ("fit_budget", "per_window", 0),
                     ("fit_budget", "resource_limits"), ("predictions", 0),
                     ("metrics", 0), ("metrics", 0, "pooled"),
                     ("metrics", 0, "per_window", 0), ("metrics", 0, "per_group", 0),
                     ("environment",), ("environment", "package_source")]
        for location in locations:
            for mode in ("extra", "missing"):
                document = deepcopy(self.document)
                record = document
                for key in location:
                    record = record[key]
                if mode == "extra":
                    record["unexpected_private_field"] = "secret-marker"
                else:
                    del record[next(iter(record))]
                with self.subTest(location=location, mode=mode):
                    self.rejected_document(document)
        for location in (("plan",), ("coverage",), ("predictions", 0), ("metrics", 0)):
            with self.subTest(null=location):
                self.mutate_reject(location, None)

    def test_all_record_schemas_reject_before_plan_or_metric_computation(self):
        cases = [(("metrics",), None), (("metrics", 0, "pooled"), None),
                 (("coverage",), None), (("coverage", "tested"), None),
                 (("fit_budget",), None), (("fit_budget", "per_window"), {}),
                 (("predictions",), None), (("predictions", 0), None)]
        for path, value in cases:
            document = deepcopy(self.document)
            put(document, path, value)
            with self.subTest(path=path), \
                    patch.object(saved, "plan_group_forward_windows", side_effect=AssertionError("planned malformed schema")), \
                    patch.object(saved, "_metrics", side_effect=AssertionError("scored malformed schema")), \
                    self.assertRaises(GroupForwardBundleError):
                saved._check_document(document)

    def test_reordered_protocol_sequences_reject(self):
        for location in (("recipes",), ("metrics",), ("plan", "folds"),
                         ("predictions",), ("fit_budget", "per_window"),
                         ("metrics", 0, "per_window"), ("metrics", 0, "per_group")):
            document = deepcopy(self.document)
            sequence = document
            for key in location:
                sequence = sequence[key]
            sequence.reverse()
            with self.subTest(location=location):
                self.rejected_document(document)

    def test_writer_requires_exact_result_not_mapping_subclass_or_preview(self):
        class SubResult(temporal.GroupForwardEvaluationResult):
            pass
        preview = object.__new__(temporal.GroupForwardEvaluationPreview)
        subclass = SubResult(**{key: getattr(self.result, key) for key in self.document})
        for value in (self.document, MappingProxyType(self.document), subclass, preview, None):
            target = self.fresh()
            with self.subTest(type=type(value)), self.assertRaises(GroupForwardBundleError):
                write(value, target)
            self.assertFalse(target.exists())

    def test_bounded_projection_does_not_call_to_dict_or_object_conversion(self):
        with patch.object(temporal.GroupForwardEvaluationResult, "to_dict",
                          side_effect=AssertionError("arbitrary conversion")):
            projected = saved._project_result(self.result)
            self.assertTrue(temporal._same_typed(projected, self.document))
            write(self.result, self.fresh())
        class Dangerous:
            def __iter__(self):
                raise AssertionError("iterated foreign object")
            def __str__(self):
                raise AssertionError("stringified foreign object")
            def to_dict(self):
                raise AssertionError("converted foreign object")
        for field, value in (("plan", dict(self.result.plan)), ("predictions", []),
                             ("environment", Dangerous()), ("schema_version", np.int64(1))):
            forged = replace(self.result, **{field: value})
            target = self.fresh()
            with self.subTest(field=field), self.assertRaises(GroupForwardBundleError):
                write(forged, target)
            self.assertFalse(target.exists())

    def test_proxy_over_custom_mapping_or_dict_subclass_cannot_run_hooks(self):
        from collections.abc import Mapping
        class DangerousMapping(Mapping):
            def __getitem__(self, key):
                raise AssertionError("custom getitem invoked")
            def __len__(self):
                raise AssertionError("custom length invoked")
            def __iter__(self):
                raise AssertionError("custom iterator invoked")
            def items(self):
                raise AssertionError("custom items invoked")
        class DangerousDict(dict):
            def __len__(self):
                raise AssertionError("subclass length invoked")
            def items(self):
                raise AssertionError("subclass items invoked")
        for mapping in (DangerousMapping(), DangerousDict()):
            forged = replace(self.result, plan=MappingProxyType(mapping))
            with self.subTest(kind=type(mapping)), self.assertRaises(GroupForwardBundleError):
                saved._project_result(forged)
            target = self.fresh()
            with self.assertRaises(GroupForwardBundleError):
                write(forged, target)
            self.assertFalse(target.exists())

    def test_cycles_and_overdeep_or_oversized_frozen_projection_reject(self):
        backing = {}
        cycle = MappingProxyType(backing)
        backing["self"] = cycle
        nested = ()
        for _ in range(saved.MAX_DEPTH + 1):
            nested = (nested,)
        for value in (cycle, MappingProxyType({"deep": nested})):
            forged = replace(self.result, plan=value)
            with self.assertRaises(GroupForwardBundleError):
                saved._project_result(forged)
        with patch.object(saved, "MAX_VALUES", 5), self.assertRaises(GroupForwardBundleError):
            saved._project_result(self.result)
        with patch.object(saved, "MAX_PAYLOAD_BYTES", 10), self.assertRaises(GroupForwardBundleError):
            saved._project_result(self.result)


class StrictTransportTests(SavedFixtureTestCase):
    """Acceptance group 4: one strict bounded UTF-8 JSON document."""
    def test_duplicate_keys_at_every_record_and_escaped_equivalents(self):
        # Duplicate a first key in every individual dictionary, retaining all
        # other data, so a permissive decoder would otherwise recover valid JSON.
        def paths(value, path=()):
            if type(value) is dict:
                yield path
                for key, item in value.items():
                    yield from paths(item, path + (key,))
            elif type(value) is list:
                for index, item in enumerate(value):
                    yield from paths(item, path + (index,))
        outer = envelope(self.document)
        for location in paths(outer):
            record = outer
            for key in location:
                record = record[key]
            if not record:
                continue
            first = next(iter(record))
            prefix = json.dumps(first) + ":" + json.dumps(record[first], ensure_ascii=True)
            duplicate = "{" + prefix + "," + json.dumps(record, ensure_ascii=True)[1:]
            # Replace the selected record with a unique marker before injecting.
            changed = deepcopy(outer)
            if location:
                put(changed, location, "__duplicate_record_marker__")
                raw = json.dumps(changed, ensure_ascii=True).replace(
                    '"__duplicate_record_marker__"', duplicate).encode("utf-8")
            else:
                raw = duplicate.encode("utf-8")
            with self.subTest(location=location), self.assertRaises(GroupForwardBundleError):
                saved._decode(raw)
        with self.assertRaises(GroupForwardBundleError):
            saved._decode(b'{"group":1,"gr\\u006fup":2}')

    def test_bad_utf8_bom_trailing_document_and_invalid_json_are_categorical(self):
        raw = canonical(envelope(self.document))
        for payload in (b"\xef\xbb\xbf" + raw, b"\xff" + raw, raw + b"{}", raw + b"SECRET",
                        b'{"private-source-value":', b"", b"{,}"):
            with self.subTest(payload=payload[:12]), self.assertRaises(GroupForwardBundleError) as error:
                saved._decode(payload)
            self.assertNotIn("private-source-value", str(error.exception))
            self.assertNotIn("SECRET", str(error.exception))
            self.assertIsNone(error.exception.__cause__)

    def test_integer_token_length_range_and_exact_type(self):
        for value in (MIN_TICK, MAX_TICK, 0, -1):
            outer = envelope(deepcopy(self.document))
            outer["result"]["plan"]["row_groups"][0] = value
            decoded = saved._decode(canonical(outer))
            actual = decoded["result"]["plan"]["row_groups"][0]
            self.assertIs(type(actual), int)
            self.assertEqual(actual, value)
        for token in (b"9223372036854775808", b"-9223372036854775809", b"10000000000000000000"):
            with self.subTest(token=token), self.assertRaises(GroupForwardBundleError):
                saved._decode(b'{"integer":' + token + b'}')
        for value in (True, 1.0):
            self.mutate_reject(("plan", "row_groups", 0), value)
            self.mutate_reject(("predictions", 0, "row_position"), value)

    def test_floating_tokens_length_finiteness_underflow_and_negative_zero(self):
        valid = ["-0.0", "0e-999", "5e-324", "1.7976931348623157e308",
                 "1." + "0" * 62, "0e" + "9" * 62, "-0e-" + "9" * 60]
        base = canonical(envelope(self.document))
        self.assertIn(b'"observed":3.0', base)
        for token in valid:
            raw = base.replace(b'"observed":3.0', b'"observed":' + token.encode(), 1)
            parsed = saved._decode(raw)["result"]["predictions"][0]["observed"]
            self.assertIs(type(parsed), float)
            self.assertEqual(struct.pack("<d", parsed), struct.pack("<d", float(token)))
        invalid = ["NaN", "Infinity", "-Infinity", "1e309", "-1e309", "1e-999",
                   "-1e-999", "2e-324", "1." + "0" * 63]
        for token in invalid:
            raw = base.replace(b'"observed":3.0', b'"observed":' + token.encode(), 1)
            with self.subTest(token=token), self.assertRaises(GroupForwardBundleError):
                saved._decode(raw)
        self.mutate_reject(("predictions", 0, "observed"), 3)
        self.mutate_reject(("predictions", 0, "prediction_dummy_mean"), 2)

    def test_exact_byte_value_and_depth_boundaries_precede_semantics(self):
        outer = envelope(self.document)
        raw = canonical(outer)
        for constant, boundary in (("MAX_PAYLOAD_BYTES", len(raw)),
                                   ("MAX_VALUES", node_count(outer)),
                                   ("MAX_DEPTH", container_depth(outer))):
            with self.subTest(constant=constant):
                with patch.object(saved, constant, boundary):
                    self.assertEqual(saved._decode(raw), outer)
                with patch.object(saved, constant, boundary - 1), \
                        patch.object(saved, "plan_group_forward_windows", side_effect=AssertionError("planned")), \
                        patch.object(saved, "_metrics", side_effect=AssertionError("scored")), \
                        self.assertRaises(GroupForwardBundleError):
                    saved._decode(raw)
        with self.assertRaises(GroupForwardBundleError):
            saved._decode(b"[" * (saved.MAX_DEPTH + 1) + b"0" + b"]" * (saved.MAX_DEPTH + 1))

    def test_strings_and_escaped_brackets_do_not_consume_container_depth(self):
        outer = envelope(deepcopy(self.document))
        outer["result"]["environment"]["python"] = '[{\\"}]' * 20
        raw = canonical(outer)
        with patch.object(saved, "MAX_DEPTH", container_depth(outer)):
            self.assertEqual(saved._decode(raw), outer)

    def test_noncanonical_whitespace_and_key_order_are_valid_retained_bytes(self):
        raw = (" \n\t" + json.dumps(envelope(self.document), sort_keys=False,
                                  ensure_ascii=True, indent=2) + "\n \t").encode()
        summary = validate(self.install(raw=raw))
        self.assertEqual(summary.payload_bytes, len(raw))
        self.assertEqual(summary.temporal_result_sha256, hashlib.sha256(raw).hexdigest())


class PlanAndPredictionTamperingTests(SavedFixtureTestCase):
    """Acceptance groups 5 and 6, always with refreshed byte checksums."""
    def test_every_complete_plan_corner_is_reconstructed(self):
        cases = [(("group_spans", 0, "start"), 2),
                 (("group_spans", 2, "label_available_at"), 19),
                 (("gap",), 7), (("folds", 0, "window", "upper"), 12),
                 (("folds", 1, "index"), 0),
                 (("folds", 0, "train", "rows"), [0, 3]),
                 (("folds", 0, "train", "n_rows"), 2),
                 (("folds", 0, "unused", "groups"), ["b", "late"]),
                 (("folds", 0, "exclusions", 0, "train"), "before_window"),
                 (("folds", 0, "exclusions", 5, "test"), "before_window"),
                 (("tested", "rows"), [1, 2, 4]), (("untested", "rows"), [0, 3, 6, 8]),
                 (("total", "n_groups"), True), (("fits_performed",), 4),
                 (("limitations", 0), "invented replacement")]
        for path, value in cases:
            with self.subTest(path=path):
                # Rebinding deliberately removes stale-digest rejection as a
                # substitute for checking the full declared plan.
                self.mutate_reject(("plan", *path), value, bind=True)
        for field in ("row_groups", "group_spans"):
            document = deepcopy(self.document)
            document["plan"][field].reverse()
            rebind(document)
            self.rejected_document(document)

    def test_coverage_must_equal_plan_and_never_claim_untested_predictions(self):
        for path, value in ((("tested", "n_rows"), 9), (("untested", "rows"), []),
                            (("total", "n_groups"), 8), (("untested_predictions_present",), True),
                            (("untested_predictions_present",), 0)):
            with self.subTest(path=path):
                self.mutate_reject(("coverage", *path), value)

    def test_prediction_roster_requires_every_and_only_tested_position(self):
        for action in ("missing", "duplicate", "extra", "reverse"):
            document = deepcopy(self.document)
            rows = document["predictions"]
            if action == "missing":
                rows.pop()
            elif action == "duplicate":
                rows[1] = deepcopy(rows[0])
            elif action == "extra":
                rows.append(deepcopy(rows[0]))
            else:
                rows.reverse()
            with self.subTest(action=action):
                self.rejected_document(document)
        for position in (0, 3, 5, 6, 8, -1, 9, True, 1.0):
            with self.subTest(position=position):
                self.mutate_reject(("predictions", 0, "row_position"), position)
        self.mutate_reject(("predictions", 0, "group"), 1)
        self.mutate_reject(("predictions", 0, "window_index"), 1)
        self.mutate_reject(("predictions", 0, "window_index"), False)

    def test_prediction_values_are_finite_floats_with_exact_recipe_fields(self):
        for field in ("observed", "prediction_dummy_mean", "prediction_ridge_alpha_1"):
            for value in (True, None, "1.0", 1, float("nan"), float("inf"), -float("inf")):
                document = deepcopy(self.document)
                document["predictions"][0][field] = value
                with self.subTest(field=field, value=value):
                    # Nonfinite values cannot be encoded by the approved writer.
                    with self.assertRaises(GroupForwardBundleError):
                        write(exact_result(document), self.fresh())
        document = deepcopy(self.document)
        row = document["predictions"][0]
        row["prediction_selected"] = row.pop("prediction_ridge_alpha_1")
        self.rejected_document(document)

    def test_dummy_constant_is_bitwise_per_window_even_after_metric_refresh(self):
        document = deepcopy(self.document)
        document["predictions"][1]["prediction_dummy_mean"] = 2.1
        refresh_metrics(document)
        self.rejected_document(document)
        zero = typed_document()
        zero["predictions"][0]["prediction_dummy_mean"] = 0.0
        refresh_metrics(zero)
        self.rejected_document(zero)

    def test_observation_edits_are_detected_by_metrics_not_original_target(self):
        document = deepcopy(self.document)
        document["predictions"][0]["observed"] = 9.0
        self.rejected_document(document)
        refresh_metrics(document)
        self.assertEqual(validate(self.install(document)).n_tested_rows, 4)
        write(exact_result(document), self.fresh())


class RecipeBudgetRoleTests(SavedFixtureTestCase):
    """Acceptance group 7: exact v1 declarations, not merely plausible limits."""
    def test_shared_pure_recipe_and_limit_helpers_keep_literal_v1_meanings(self):
        preprocessing = [
            {"step": "SimpleImputer", "strategy": "median", "keep_empty_features": True},
            {"step": "StandardScaler", "with_mean": True, "with_std": True},
        ]
        expected = [
            {"name": "dummy_mean", "preprocessing": preprocessing,
             "estimator": "DummyRegressor", "parameters": {"strategy": "mean"},
             "training_weight": "unweighted_rows"},
            {"name": "ridge_alpha_1", "preprocessing": preprocessing,
             "estimator": "Ridge", "parameters": {"alpha": 1.0, "solver": "lsqr",
                 "fit_intercept": True, "tol": 1e-8, "max_iter": 1000,
                 "copy_X": True, "positive": False},
             "training_weight": "unweighted_rows", "iteration_cap_policy": "reject_at_cap"},
        ]
        self.assertTrue(temporal._same_typed(temporal._recipes(), expected))
        self.assertTrue(temporal._same_typed(temporal._limits(28), {
            "max_rows": 20000, "max_features": 100, "max_windows": 32,
            "max_groups": 20000, "max_window_row_group_work": 250000,
            "max_plan_document_bytes": 16 * 1024 * 1024,
            "max_matrix_cell_visits": 20000000, "matrix_cell_visits": 28,
            "max_pipeline_fits": 64, "threads": 1, "ridge_max_iter": 1000,
        }))

    def test_population_and_work_bounds_reject_before_reconstruction_or_metrics(self):
        for name, limit in (("MAX_ROWS", 8), ("MAX_FEATURES", 0), ("MAX_WINDOWS", 1),
                            ("MAX_WORK", 31), ("MAX_PLAN_BYTES", 10)):
            with self.subTest(name=name), patch.object(saved, name, limit), \
                    patch.object(saved, "plan_group_forward_windows", side_effect=AssertionError("planned")), \
                    patch.object(saved, "_metrics", side_effect=AssertionError("scored")), \
                    self.assertRaises(GroupForwardBundleError):
                saved._check_document(deepcopy(self.document))
        with patch.object(saved, "MAX_MATRIX_CELLS", 27), \
                patch.object(saved, "_metrics", side_effect=AssertionError("scored")), \
                self.assertRaises(GroupForwardBundleError):
            saved._check_document(deepcopy(self.document))

    def test_fixed_recipe_every_material_parameter_and_order(self):
        cases = [((1, "parameters", "solver"), "auto"), ((1, "parameters", "alpha"), 2.0),
                 ((1, "parameters", "alpha"), 1), ((1, "parameters", "tol"), 1e-7),
                 ((1, "parameters", "max_iter"), 999), ((1, "parameters", "fit_intercept"), False),
                 ((1, "parameters", "copy_X"), False), ((1, "parameters", "positive"), True),
                 ((1, "iteration_cap_policy"), "accept_at_cap"),
                 ((0, "parameters", "strategy"), "median"),
                 ((0, "training_weight"), "group_balanced"),
                 ((0, "preprocessing", 0, "strategy"), "mean"),
                 ((0, "preprocessing", 0, "keep_empty_features"), False),
                 ((1, "preprocessing", 1, "with_mean"), False),
                 ((1, "preprocessing", 1, "with_std"), False)]
        for path, value in cases:
            with self.subTest(path=path):
                self.mutate_reject(("recipes", *path), value, bind=True)
        document = deepcopy(self.document)
        document["recipes"].append({"name": "selected"})
        rebind(document)
        self.rejected_document(document)

    def test_budget_totals_window_counts_and_operation_counts_are_exact(self):
        for field in ("expected_total", "attempted_total", "completed_total"):
            for value in (3, 5, True, 4.0):
                with self.subTest(field=field, value=value):
                    self.mutate_reject(("fit_budget", field), value)
        for field in ("expected", "attempted", "completed", "window_index"):
            self.mutate_reject(("fit_budget", "per_window", 0, field), 7)
        for field in ("inner_selection_fits", "full_data_refits", "selected_model_refits", "bootstraps", "retries"):
            self.mutate_reject(("fit_budget", field), 1)
        self.mutate_reject(("fit_budget", "unit"), "estimators")

    def test_resource_record_is_recomputed_not_trusted_because_below_cap(self):
        for field, value in (("matrix_cell_visits", 26), ("matrix_cell_visits", 30),
                             ("threads", 2), ("max_pipeline_fits", 65),
                             ("max_rows", 19999), ("ridge_max_iter", 1001),
                             ("max_matrix_cell_visits", 20000001)):
            self.mutate_reject(("fit_budget", "resource_limits", field), value, bind=True)
        document = deepcopy(self.document)
        document["feature_schema"].append({"name": "second", "source_dtype": "int32",
                                           "normalized_dtype": "float64"})
        self.rejected_document(document)
        document["fit_budget"]["resource_limits"]["matrix_cell_visits"] = 56
        rebind(document)
        self.assertEqual(validate(self.install(document)).n_features, 2)

    def test_feature_dtype_admission_is_bounded_real_numpy_declaration(self):
        for dtype in ("float16", "float32", "float64", "int8", "int16", "int32", "int64",
                      "uint8", "uint16", "uint32", "uint64", ">f8", "<f8", ">i4"):
            document = deepcopy(self.document)
            document["feature_schema"][0]["source_dtype"] = dtype
            with self.subTest(dtype=dtype):
                self.assertEqual(validate(self.install(document)).n_features, 1)
        for dtype in ("object", "bool", "complex128", "datetime64[ns]", "timedelta64[ns]",
                      "Int64", "Float64", "float128", "longdouble", "S8", "U4", "void", "", None, 64):
            with self.subTest(dtype=dtype):
                self.mutate_reject(("feature_schema", 0, "source_dtype"), dtype)
        self.mutate_reject(("feature_schema", 0, "normalized_dtype"), "float32")

    def test_declared_names_and_roles_cannot_conflict(self):
        for path, value in ((("feature_schema", 0, "name"), "group"),
                            (("feature_schema", 0, "name"), "outcome"),
                            (("feature_schema", 0, "name"), " padded"),
                            (("feature_schema", 0, "name"), ""),
                            (("target_name",), "group"), (("target_name",), "row_key"),
                            (("group_name",), "absent"), (("group_name",), " padded"),
                            (("metadata_columns",), ["group", "group"]),
                            (("metadata_columns",), ["outcome", "group"]),
                            (("metadata_columns",), ["group", "invented_feature"]),
                            (("metadata_columns",), [])):
            with self.subTest(path=path, value=value):
                self.mutate_reject(path, value)
        document = deepcopy(self.document)
        document["feature_schema"] *= 2
        self.rejected_document(document)


class IndependentMetricTests(SavedFixtureTestCase):
    """Acceptance group 8: independent arithmetic and null/type semantics."""
    def test_literal_small_metric_expectations_and_unequal_group_sizes(self):
        dummy, ridge = self.document["metrics"]
        expected_dummy = {"n_rows": 4, "mae": 2.0, "rmse": math.sqrt(5.5), "r2": -3.4,
                          "bias": -2.0, "n_groups": 3, "group_mae": 5.0 / 3.0,
                          "group_rmse": (2.0 + math.sqrt(10.0)) / 3.0, "worst_group_mae": 3.0}
        expected_ridge = {"n_rows": 4, "mae": 0.5, "rmse": math.sqrt(0.375), "r2": 0.7,
                          "bias": -0.5, "n_groups": 3, "group_mae": 0.5,
                          "group_rmse": (1.0 + math.sqrt(0.5)) / 3.0, "worst_group_mae": 0.5}
        for actual, expected in ((dummy["pooled"], expected_dummy), (ridge["pooled"], expected_ridge)):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assertAlmostEqual(actual[key], expected[key], places=13)
        self.assertEqual([row["group"] for row in dummy["per_group"]], ["1", "a", "b"])
        self.assertEqual([row["n_rows"] for row in dummy["per_group"]], [1, 2, 1])
        self.assertAlmostEqual(dummy["per_window"][0]["mae"], 7.0 / 3.0)
        self.assertAlmostEqual(dummy["per_window"][0]["rmse"], math.sqrt(7.0))
        self.assertAlmostEqual(dummy["per_window"][0]["r2"], -3.5)
        self.assertAlmostEqual(ridge["per_window"][0]["r2"], 41.0 / 56.0)
        self.assertEqual(dummy["per_group"][1]["r2"], -9.0)
        self.assertEqual(ridge["per_group"][1]["r2"], 0.5)
        self.assertNotAlmostEqual(dummy["pooled"]["mae"],
                                  sum(row["mae"] for row in dummy["per_window"]) / 2)
        self.assertNotAlmostEqual(dummy["pooled"]["group_rmse"], math.sqrt(12.0 / 3.0))
        self.assertEqual(validate(self.install()).n_tested_rows, 4)

    def test_every_metric_field_is_checked_outside_tolerance(self):
        for recipe in range(2):
            for location in (("pooled",), ("per_window", 0), ("per_window", 1),
                             ("per_group", 0), ("per_group", 1), ("per_group", 2)):
                source = self.document["metrics"][recipe]
                for key in location:
                    source = source[key]
                for field, value in source.items():
                    if type(value) is float:
                        replacement = value + 0.1
                    elif value is None:
                        replacement = 0.0
                    elif field == "group":
                        replacement = "wrong-group"
                    else:
                        replacement = value + 1
                    with self.subTest(recipe=recipe, location=location, field=field):
                        self.mutate_reject(("metrics", recipe, *location, field), replacement)

    def test_metric_floats_counts_nulls_and_tolerance_are_distinct(self):
        for value in (2, True, None, "2.0"):
            self.mutate_reject(("metrics", 0, "pooled", "mae"), value)
        for value in (4.0, True):
            self.mutate_reject(("metrics", 0, "pooled", "n_rows"), value)
        self.mutate_reject(("metrics", 0, "per_group", 0, "r2"), 0.0)
        self.mutate_reject(("metrics", 0, "per_group", 1, "r2"), None)
        document = deepcopy(self.document)
        document["metrics"][0]["pooled"]["mae"] += 1e-11
        self.assertEqual(validate(self.install(document)).n_recipes, 2)
        document["metrics"][0]["pooled"]["mae"] += 1e-5
        self.rejected_document(document)

    def test_checked_metric_arithmetic_overflow_and_underflow_reject(self):
        for observed, prediction in ((1e308, -1e308), (1e-200, 0.0)):
            document = deepcopy(self.document)
            for row in document["predictions"]:
                row["observed"] = observed
                row["prediction_dummy_mean"] = prediction
                row["prediction_ridge_alpha_1"] = prediction
            with self.subTest(observed=observed):
                self.rejected_document(document)


class BindingsSourceAndHonestLimitsTests(SavedFixtureTestCase):
    """Acceptance groups 9 and 10: internal consistency, not authentication."""
    def test_binding_digest_syntax_and_recomputed_relations(self):
        for key in ("plan_sha256", "input_sha256", "execution_sha256"):
            for value in ("G" * 64, "a" * 63, "", None):
                with self.subTest(key=key, value=value):
                    self.mutate_reject(("bindings", key), value)
        self.mutate_reject(("bindings", "plan_sha256"), "b" * 64)
        self.mutate_reject(("bindings", "execution_sha256"), "c" * 64)
        self.mutate_reject(("bindings", "encoding"), "other")

    def test_source_aggregate_markers_counts_and_unsafe_inert_names(self):
        for field, value in (("schema_version", 1), ("algorithm", "sha512"),
                             ("scope", "other"), ("status", "unavailable"),
                             ("sha256", "b" * 64), ("files", {})):
            self.mutate_reject(("environment", "package_source", field), value, bind=True)
        for name in ("/absolute.py", "../parent.py", "a/../b.py", "./a.py", "a//b.py",
                     "a\\b.py", "nul\x00.py", "a.txt", "x" * 257 + ".py", "https://example.org/a.py"):
            document = deepcopy(self.document)
            document["environment"]["package_source"] = source_declaration([name])
            rebind(document)
            with self.subTest(name=name):
                self.rejected_document(document)
        document = deepcopy(self.document)
        document["environment"]["package_source"] = source_declaration(
            [f"invented_{index}.py" for index in range(257)])
        rebind(document)
        self.rejected_document(document)

    def test_old_37_file_source_and_changed_versions_remain_readable(self):
        self.assertEqual(len(HISTORIC_37_FILES), 37)
        self.assertNotIn("group_time_saved.py", HISTORIC_37_FILES)
        document = deepcopy(self.document)
        document["environment"]["package_source"] = source_declaration(HISTORIC_37_FILES)
        for key in document["environment"]:
            if key != "package_source":
                document["environment"][key] = "historical-other-version"
        rebind(document)
        directory = self.install(document)
        with patch.object(package_source, "SOURCE_FILES", ("changed_installation.py",)):
            self.assertEqual(validate(directory).n_windows, 2)
        write(exact_result(document), self.fresh())

    def test_source_map_and_environment_exact_admission_boundaries(self):
        document = deepcopy(self.document)
        document["environment"]["package_source"] = source_declaration(
            ["a" * 253 + ".py", *(f"source_{index}.py" for index in range(255))])
        document["environment"]["python"] = "v" * 256
        rebind(document)
        self.assertEqual(validate(self.install(document)).n_rows, 9)

    def test_environment_strings_are_bounded_exact_and_unpadded(self):
        for key in ("package_version", "python", "numpy", "pandas", "sklearn", "scipy", "threadpoolctl"):
            for value in ("", " version", "version ", "v" * 257, None, 1):
                with self.subTest(key=key, value=value):
                    self.mutate_reject(("environment", key), value, bind=True)

    def test_opaque_input_hash_and_invented_source_map_can_be_consistent(self):
        document = deepcopy(self.document)
        document["bindings"]["input_sha256"] = "e" * 64
        document["environment"]["package_source"] = source_declaration(["never_existed.py"])
        rebind(document)
        summary = validate(self.install(document))
        self.assertEqual(summary.validation_fits_performed, 0)
        unchecked = " ".join(summary.unchecked_scope).lower()
        for concept in ("input", "training", "authentic", "scientific", "atomic"):
            self.assertIn(concept, unchecked)
        checked = " ".join(summary.checked_scope).lower().replace("-", "")
        for concept in ("plan", "metric", "source", "sha256"):
            self.assertIn(concept, checked)
        write(exact_result(document), self.fresh())

    def test_consistent_new_predictions_need_not_come_from_a_fitted_estimator(self):
        document = deepcopy(self.document)
        for row in document["predictions"]:
            row["prediction_dummy_mean"] = 8.0 if row["window_index"] == 0 else -2.0
            row["prediction_ridge_alpha_1"] = row["observed"] + 0.25
        refresh_metrics(document)
        self.assertEqual(validate(self.install(document)).n_recipes, 2)
        write(exact_result(document), self.fresh())

    def test_valid_role_and_feature_order_edits_do_not_reconstruct_input_digest(self):
        document = deepcopy(self.document)
        original_input = document["bindings"]["input_sha256"]
        document["target_name"] = "other_target"
        document["feature_schema"][0]["name"] = "other_feature"
        document["feature_schema"][0]["source_dtype"] = "int64"
        self.assertEqual(validate(self.install(document)).n_features, 1)
        self.assertEqual(document["bindings"]["input_sha256"], original_input)
        # With no original feature array, two valid feature names' order is an
        # opaque input-linkage claim, not independently verifiable provenance.
        document["feature_schema"].append({"name": "second", "source_dtype": "float64",
                                           "normalized_dtype": "float64"})
        document["fit_budget"]["resource_limits"]["matrix_cell_visits"] = 56
        rebind(document)
        document["feature_schema"].reverse()
        self.assertEqual(validate(self.install(document)).n_features, 2)


class ReadOnlyIOTests(SavedFixtureTestCase):
    """Acceptance group 11: pin, bound, retain, and recheck the exact bytes."""
    def test_reads_each_owned_file_once_parses_retained_bytes_and_never_writes(self):
        directory = self.install()
        files = [directory / name for name in ("checksums.json", "temporal_result.json")]
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
        reads, decoded = [], []
        real_read, real_decode = verification._read_file, saved._decode
        retained = {}
        def read(fd, name, limit, **kwargs):
            reads.append((name, kwargs.get("collect")))
            result = real_read(fd, name, limit, **kwargs)
            retained[name] = result[2]
            return result
        def decode(raw):
            self.assertIs(raw, retained["temporal_result.json"])
            decoded.append(raw)
            return real_decode(raw)
        with ExitStack() as stack:
            stack.enter_context(patch.object(verification, "_read_file", read))
            stack.enter_context(patch.object(saved, "_decode", decode))
            for owner, name in ((Path, "write_bytes"), (Path, "write_text"),
                                (os, "mkdir"), (os, "link"), (os, "replace"), (os, "unlink")):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError("read-only")))
            summary = validate(directory)
        self.assertEqual(reads, [("checksums.json", True), ("temporal_result.json", True)])
        self.assertEqual(len(decoded), 1)
        self.assertEqual(summary.payload_bytes, len(decoded[0]))
        self.assertEqual({path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}, before)

    def test_unlisted_source_like_files_and_declared_paths_are_never_opened(self):
        directory = self.install()
        (directory / "invented").mkdir()
        (directory / "invented" / "not_installed.py").write_text("unverified source decoy")
        (directory / "unlisted.py").symlink_to(directory / "does-not-exist")
        real_open = os.open
        calls = []
        def guarded_open(path, flags, *args, **kwargs):
            calls.append(os.fspath(path))
            # Directory components are allowed; all nondirectory reads are the
            # two fixed owned filenames, never anything from JSON or listing.
            if not flags & os.O_DIRECTORY:
                self.assertIn(os.fspath(path), ("checksums.json", "temporal_result.json"))
                self.assertFalse(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            return real_open(path, flags, *args, **kwargs)
        with patch.object(os, "open", guarded_open), \
                patch.object(builtins, "open", side_effect=AssertionError("unexpected source open")):
            self.assertEqual(validate(directory).n_rows, 9)
        self.assertNotIn("unlisted.py", calls)
        self.assertNotIn("not_installed.py", calls)

    def test_symlink_components_directory_and_files_reject(self):
        directory = self.install()
        alias = self.root / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        with self.assertRaises(GroupForwardBundleError):
            validate(alias)
        parent_alias = self.root / "parent-alias"
        parent_alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(GroupForwardBundleError):
            validate(parent_alias / directory.name)
        for name in ("checksums.json", "temporal_result.json"):
            for broken in (False, True):
                changed = self.install()
                payload = changed / name
                old = payload.read_bytes()
                payload.unlink()
                target = self.fresh("target")
                if not broken:
                    target.write_bytes(old)
                payload.symlink_to(target)
                with self.subTest(name=name, broken=broken), self.assertRaises(GroupForwardBundleError):
                    validate(changed)

    def test_nonregular_payloads_are_rejected_without_opening_them(self):
        for kind in ("directory", "fifo"):
            directory = self.install()
            target = directory / "temporal_result.json"
            target.unlink()
            if kind == "directory":
                target.mkdir()
            else:
                os.mkfifo(target)
            real_open = os.open
            def guard(path, flags, *args, **kwargs):
                if os.fspath(path) == "temporal_result.json":
                    raise AssertionError("nonregular payload opened")
                return real_open(path, flags, *args, **kwargs)
            with self.subTest(kind=kind), patch.object(os, "open", guard), \
                    self.assertRaises(GroupForwardBundleError):
                validate(directory)

    def test_device_metadata_rejects_before_open(self):
        import stat
        directory = self.install()
        real_stat, real_open = os.stat, os.open
        def device_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if path == "temporal_result.json" and kwargs.get("dir_fd") is not None:
                fields = list(result)
                fields[0] = stat.S_IFCHR | 0o600
                return os.stat_result(fields)
            return result
        def guard(path, flags, *args, **kwargs):
            if path == "temporal_result.json":
                raise AssertionError("device opened")
            return real_open(path, flags, *args, **kwargs)
        with patch.object(os, "stat", device_stat), patch.object(os, "open", guard), \
                self.assertRaises(GroupForwardBundleError):
            validate(directory)

    def test_payload_byte_limit_and_growing_stream_fail_closed(self):
        directory = self.install()
        size = (directory / "temporal_result.json").stat().st_size
        with patch.object(saved, "MAX_PAYLOAD_BYTES", size - 1), self.assertRaises(GroupForwardBundleError):
            validate(directory)
        with patch.object(saved, "MAX_PAYLOAD_BYTES", size):
            self.assertEqual(validate(directory).payload_bytes, size)
        real_read = os.read
        payload_inode = (directory / "temporal_result.json").stat().st_ino
        extra_sent = False
        def growing_read(fd, count):
            nonlocal extra_sent
            value = real_read(fd, count)
            if not value and os.fstat(fd).st_ino == payload_inode and not extra_sent:
                extra_sent = True
                return b" "
            return value
        with patch.object(saved, "MAX_PAYLOAD_BYTES", size), patch.object(os, "read", growing_read), \
                self.assertRaises(GroupForwardBundleError):
            validate(directory)
        self.assertTrue(extra_sent)

    def test_checksum_size_syntax_inventory_and_path_keys_reject(self):
        payload_hash = hashlib.sha256(canonical(envelope(self.document))).hexdigest()
        invalid = [b" " * (verification.MAX_CHECKSUM_BYTES + 1),
                   b'{"algorithm":"sha256","algorithm":"sha256","files":{}}',
                   canonical({"algorithm": "md5", "files": {"temporal_result.json": payload_hash}}),
                   canonical({"algorithm": "sha256", "files": {"../temporal_result.json": payload_hash}}),
                   canonical({"algorithm": "sha256", "files": {"/temporal_result.json": payload_hash}}),
                   canonical({"algorithm": "sha256", "files": {"temporal_result.json": "F" * 64}}),
                   canonical({"algorithm": "sha256", "files": {
                       "temporal_result.json": payload_hash, "extra.py": payload_hash}})]
        for raw in invalid:
            directory = self.install()
            (directory / "checksums.json").write_bytes(raw)
            with self.subTest(raw=raw[:70]), self.assertRaises(GroupForwardBundleError):
                validate(directory)

    def test_missing_files_hash_mismatch_and_unsupported_platform_reject(self):
        for name in ("checksums.json", "temporal_result.json"):
            directory = self.install()
            (directory / name).unlink()
            with self.assertRaises(GroupForwardBundleError):
                validate(directory)
        directory = self.install()
        (directory / "temporal_result.json").write_bytes(b"wrong")
        with self.assertRaises(GroupForwardBundleError):
            validate(directory)
        with patch.object(verification, "_SAFE_OPEN_SUPPORTED", False), \
                self.assertRaises(GroupForwardBundleError):
            validate(self.install())
        target = self.fresh()
        with patch.object(verification, "_SAFE_OPEN_SUPPORTED", False), \
                self.assertRaises(GroupForwardBundleError):
            write(self.result, target)
        self.assertFalse(target.exists())

    def test_direct_path_restrictions_and_error_redaction(self):
        directory = self.install()
        paths = [directory / ".." / directory.name,
                 self.root / "PRIVATE_SOURCE_PATH_not_present",
                 self.root / ("x" * 4097)]
        for path in paths:
            with self.subTest(path_length=len(os.fspath(path))), \
                    self.assertRaises(GroupForwardBundleError) as caught:
                validate(path)
            self.assertNotIn("PRIVATE_SOURCE_PATH", str(caught.exception))
            self.assertIsNone(caught.exception.__cause__)
        document = deepcopy(self.document)
        document["predictions"][0]["group"] = "PRIVATE_GROUP_LABEL"
        with self.assertRaises(GroupForwardBundleError) as caught:
            validate(self.install(document))
        self.assertNotIn("PRIVATE_GROUP_LABEL", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_replacement_between_lstat_and_open_is_detected(self):
        directory = self.install()
        target = directory / "temporal_result.json"
        replacement = directory / "replacement.unlisted"
        replacement.write_bytes(target.read_bytes())
        real_open = os.open
        replaced = False
        def replacing_open(path, flags, *args, **kwargs):
            nonlocal replaced
            if path == "temporal_result.json" and not replaced:
                replaced = True
                os.replace(replacement, target)
            return real_open(path, flags, *args, **kwargs)
        with patch.object(os, "open", replacing_open), self.assertRaises(GroupForwardBundleError):
            validate(directory)
        self.assertTrue(replaced)

    def test_mutation_during_payload_read_is_detected(self):
        directory = self.install()
        target = directory / "temporal_result.json"
        inode = target.stat().st_ino
        real_read = os.read
        changed = False
        def mutating_read(fd, count):
            nonlocal changed
            raw = real_read(fd, count)
            if raw and not changed and os.fstat(fd).st_ino == inode:
                changed = True
                with target.open("ab") as stream:
                    stream.write(b" ")
            return raw
        with patch.object(os, "read", mutating_read), self.assertRaises(GroupForwardBundleError):
            validate(directory)
        self.assertTrue(changed)

    def test_retained_file_signatures_rechecked_after_semantics(self):
        for name in ("checksums.json", "temporal_result.json"):
            directory = self.install()
            target = directory / name
            replacement = directory / "replacement.unlisted"
            replacement.write_bytes(target.read_bytes())
            real_check = saved._check_document
            def replaced_after_check(document):
                plan = real_check(document)
                os.replace(replacement, target)
                return plan
            with self.subTest(name=name), patch.object(saved, "_check_document", replaced_after_check), \
                    self.assertRaises(GroupForwardBundleError):
                validate(directory)

    def test_stale_legacy_names_reject_files_directories_and_broken_links(self):
        for name in ("manifest.json", "summary.json", "predictions.csv", "report.html"):
            for kind in ("file", "directory", "broken_link"):
                directory = self.install()
                target = directory / name
                if kind == "file":
                    target.write_text("stale")
                elif kind == "directory":
                    target.mkdir()
                else:
                    target.symlink_to(directory / "missing")
                with self.subTest(name=name, kind=kind):
                    with self.assertRaises(GroupForwardBundleError):
                        validate(directory)
                    with self.assertRaises(ValueError):
                        verification.verify_bundle(directory)

    def test_stale_legacy_name_appearing_during_semantics_rejects(self):
        directory = self.install()
        real_check = saved._check_document
        def stale_after_check(document):
            plan = real_check(document)
            (directory / "summary.json").symlink_to(directory / "missing")
            return plan
        with patch.object(saved, "_check_document", stale_after_check), \
                self.assertRaises(GroupForwardBundleError):
            validate(directory)


class FreshOnlyWriterTests(SavedFixtureTestCase):
    """Acceptance group 12: validate first, fresh leaf, no-clobber publication."""
    def test_entire_validation_and_strict_encoding_precede_first_write(self):
        target = self.fresh()
        order = []
        real_check, real_decode, real_mkdir = saved._check_document, saved._decode, Path.mkdir
        def check(document):
            self.assertFalse(target.exists())
            order.append("check")
            return real_check(document)
        def decode(raw):
            self.assertFalse(target.exists())
            order.append("decode")
            return real_decode(raw)
        def mkdir(path, *args, **kwargs):
            if path == target and not target.exists():
                self.assertEqual(order, ["check", "decode", "check"])
                order.append("mkdir")
            return real_mkdir(path, *args, **kwargs)
        with patch.object(saved, "_check_document", check), patch.object(saved, "_decode", decode), \
                patch.object(Path, "mkdir", mkdir):
            write(self.result, target)
        self.assertEqual(order[:4], ["check", "decode", "check", "mkdir"])

    def test_invalid_result_and_encoded_resource_failure_create_nothing(self):
        document = deepcopy(self.document)
        document["metrics"][0]["pooled"]["mae"] = 99.0
        for result, limit in ((exact_result(document), None), (self.result, 100)):
            target = self.fresh()
            with ExitStack() as stack:
                if limit is not None:
                    stack.enter_context(patch.object(saved, "MAX_PAYLOAD_BYTES", limit))
                stack.enter_context(patch.object(os, "mkdir", side_effect=AssertionError("wrote before validation")))
                with self.assertRaises(GroupForwardBundleError):
                    write(result, target)
            self.assertFalse(target.exists())

    def test_writer_refuses_every_observed_occupied_leaf(self):
        for kind in ("directory", "file", "symlink", "broken_link", "fifo"):
            target = self.fresh()
            if kind == "directory":
                target.mkdir()
            elif kind == "file":
                target.write_bytes(b"original")
            elif kind == "fifo":
                os.mkfifo(target)
            else:
                target.symlink_to(self.root if kind == "symlink" else self.root / "absent")
            with self.subTest(kind=kind), self.assertRaises(FileExistsError):
                write(self.result, target)
            if kind == "file":
                self.assertEqual(target.read_bytes(), b"original")

    def test_writer_rejects_bad_missing_symlink_and_traversal_parents(self):
        alias = self.root / "parent-link"
        alias.symlink_to(self.root, target_is_directory=True)
        regular = self.root / "regular-parent"
        regular.write_bytes(b"not a directory")
        for target in (alias / "child", regular / "child", self.root / "missing-parent" / "child",
                       self.root / ".." / self.root.name / "child"):
            with self.subTest(target=target), self.assertRaises(GroupForwardBundleError):
                write(self.result, target)
            self.assertFalse(target.exists())

    def test_second_invocation_cannot_overwrite_or_change_any_bytes(self):
        directory = self.fresh()
        write(self.result, directory)
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()}
        with self.assertRaises(FileExistsError):
            write(self.result, directory)
        self.assertEqual({path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in directory.iterdir()}, before)
        with self.assertRaises(TypeError):
            write(self.result, self.fresh(), overwrite=True)

    def test_mkdir_race_never_reuses_another_writers_directory(self):
        target = self.fresh()
        real_mkdir = Path.mkdir
        raced = False
        def racing_mkdir(path, *args, **kwargs):
            nonlocal raced
            if path == target and not raced:
                raced = True
                real_mkdir(path)
                (path / "other-writer-marker").write_bytes(b"keep")
                raise FileExistsError("injected race")
            return real_mkdir(path, *args, **kwargs)
        with patch.object(Path, "mkdir", racing_mkdir), \
                patch.object(saved, "write_payloads", side_effect=AssertionError("reused raced leaf")), \
                self.assertRaises(FileExistsError):
            write(self.result, target)
        self.assertEqual({path.name for path in target.iterdir()}, {"other-writer-marker"})

    def test_exact_utf8_bytes_and_checksum_under_crlf_default_simulation(self):
        target = self.fresh()
        real_open = Path.open
        def crlf_default(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
            if "w" in mode and "b" not in mode and newline is None:
                newline = "\r\n"
            return real_open(path, mode, buffering, encoding, errors, newline)
        with patch.object(Path, "open", crlf_default):
            write(exact_result(typed_document()), target)
        raw = (target / "temporal_result.json").read_bytes()
        self.assertEqual(raw, canonical(envelope(typed_document())))
        self.assertNotIn(b"\r\n", raw)
        expected = {"algorithm": "sha256", "files": {"temporal_result.json": hashlib.sha256(raw).hexdigest()}}
        self.assertEqual((target / "checksums.json").read_bytes(),
                         (json.dumps(expected, indent=2) + "\n").encode())
        self.assertEqual(validate(target).payload_bytes, len(raw))

    def test_staging_failure_leaves_fresh_empty_incomplete_directory_no_retry(self):
        target = self.fresh()
        real_write = Path.write_bytes
        def fail_checksum_staging(path, value):
            if path.name == "checksums.json":
                raise OSError("PRIVATE_IO_DETAILS")
            return real_write(path, value)
        with patch.object(Path, "write_bytes", fail_checksum_staging), self.assertRaises(OSError) as caught:
            write(self.result, target)
        self.assertNotIn("PRIVATE_IO_DETAILS", str(caught.exception))
        self.assertTrue(target.is_dir())
        self.assertEqual(list(target.iterdir()), [])
        with self.assertRaises(GroupForwardBundleError):
            validate(target)
        with self.assertRaises(FileExistsError):
            write(self.result, target)

    def test_partial_publication_failure_keeps_payload_but_never_claims_success(self):
        target = self.fresh()
        real_link = os.link
        links = []
        def fail_checksum(source, destination, *args, **kwargs):
            links.append(Path(destination).name)
            if Path(destination).name == "checksums.json":
                raise OSError("PRIVATE_LINK_ERROR")
            return real_link(source, destination, *args, **kwargs)
        with patch.object(os, "link", fail_checksum), self.assertRaises(OSError) as caught:
            write(self.result, target)
        self.assertEqual(links, ["temporal_result.json", "checksums.json"])
        self.assertNotIn("PRIVATE_LINK_ERROR", str(caught.exception))
        self.assertEqual({path.name for path in target.iterdir()}, {"temporal_result.json"})
        with self.assertRaises(GroupForwardBundleError):
            validate(target)
        with self.assertRaises(FileExistsError):
            write(self.result, target)

    def test_racing_payload_filename_is_never_replaced_or_cleaned_up(self):
        target = self.fresh()
        real_link = os.link
        raced = False
        def racing_link(source, destination, *args, **kwargs):
            nonlocal raced
            destination = Path(destination)
            if destination.name == "temporal_result.json" and not raced:
                raced = True
                destination.write_bytes(b"another writer's retained bytes")
            return real_link(source, destination, *args, **kwargs)
        with patch.object(os, "link", racing_link), self.assertRaises(FileExistsError):
            write(self.result, target)
        self.assertEqual((target / "temporal_result.json").read_bytes(), b"another writer's retained bytes")
        self.assertFalse((target / "checksums.json").exists())

    def test_generic_publication_is_used_once_and_only_without_overwrite(self):
        target = self.fresh()
        real_publish = saved.write_payloads
        calls = []
        def publish(payloads, directory, *, overwrite=False):
            calls.append((set(payloads), directory, overwrite))
            return real_publish(payloads, directory, overwrite=overwrite)
        with patch.object(saved, "write_payloads", publish):
            write(self.result, target)
        self.assertEqual(calls, [({"temporal_result.json"}, target, False)])


class SeparationSideEffectsAndCompatibilityTests(SavedFixtureTestCase):
    """Acceptance groups 13 through 15: no hidden execution or legacy dispatch."""
    def test_byte_verifier_accepts_opaque_temporal_bytes_without_semantic_dispatch(self):
        directory = self.install(raw=b"not JSON; deliberately opaque\xff\x00")
        with patch.object(saved, "_decode", side_effect=AssertionError("semantic dispatch")), \
                patch.object(saved, "_check_document", side_effect=AssertionError("semantic dispatch")):
            result = verification.verify_bundle(directory)
        self.assertEqual(result.bundle_type, "temporal_evaluation")
        self.assertEqual(result.files, ("temporal_result.json",))
        with self.assertRaises(GroupForwardBundleError):
            validate(directory)
        document = deepcopy(self.document)
        document["metrics"][0]["pooled"]["mae"] = 999.0
        directory = self.install(document)
        self.assertEqual(verification.verify_bundle(directory).bundle_type, "temporal_evaluation")
        with self.assertRaises(GroupForwardBundleError):
            validate(directory)

    def test_legacy_semantic_validators_reject_temporal_inventory(self):
        from groupbench.evaluation_schema import validate_evaluation_bundle
        from groupbench.prediction_schema import validate_prediction_bundle_schema
        directory = self.install()
        for validator in (validate_evaluation_bundle, validate_prediction_bundle_schema):
            with self.subTest(validator=validator.__name__), self.assertRaises(ValueError):
                validator(directory)

    def test_temporal_schema_disguised_in_legacy_inventory_is_not_legacy_result(self):
        from groupbench.evaluation_schema import validate_evaluation_bundle
        from groupbench.prediction_schema import validate_prediction_bundle_schema
        for inventory, validator in ((verification.PREDICTION_FILES, validate_prediction_bundle_schema),
                                     (verification.EVALUATION_FILES, validate_evaluation_bundle)):
            target = self.fresh()
            target.mkdir()
            payloads = {name: "{}\n" for name in inventory}
            payloads["manifest.json"] = canonical(self.document).decode()
            payloads["predictions.csv"] = "row_position,group,observed\n1,1,3\n"
            report.write_payloads(payloads, target)
            self.assertIn(verification.verify_bundle(target).bundle_type, ("prediction", "evaluation"))
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                validator(target)
            with self.assertRaises(GroupForwardBundleError):
                validate(target)

    def test_binary_budget_and_interval_document_contracts_reject_temporal_schemas(self):
        from groupbench.binary.saved_evaluation import (
            evaluate_saved_binary_probabilities, load_saved_binary_document)
        from groupbench.budget_comparison import compare_saved_budgets
        from groupbench.interval_evaluation import (
            evaluate_saved_intervals, diagnose_saved_interval_reference)
        for document in (deepcopy(self.document), envelope(deepcopy(self.document))):
            for validator in (evaluate_saved_binary_probabilities, compare_saved_budgets,
                              evaluate_saved_intervals, diagnose_saved_interval_reference):
                with self.subTest(schema=document["schema_id"], validator=validator.__name__), \
                        self.assertRaises(ValueError):
                    validator(document)
            with self.subTest(binary_loader_schema=document["schema_id"]), self.assertRaises(ValueError):
                load_saved_binary_document(canonical(document).decode("utf-8"))

    def test_delayed_scoring_and_paired_comparison_reject_before_reading_other_inputs(self):
        import groupbench.saved_scoring as scoring
        directory = self.install()
        unseen_query = self.root / "never-read-query.csv"
        unseen_outcomes = self.root / "never-read-outcomes.csv"
        contract = {"method_a": "invented-A", "method_b": "invented-B",
                    "target_column": "outcome", "target_definition": "invented response",
                    "target_unit": "invented unit", "cohort_definition": "invented cohort",
                    "row_mapping_definition": "invented positional mapping",
                    "group_definition": "invented group labels"}
        with patch.object(scoring, "_read_outcomes", side_effect=AssertionError("read outcomes")), \
                patch("groupbench.cli._read_csv_with_hash", side_effect=AssertionError("read query")):
            with self.assertRaises(scoring.SavedScoringError):
                scoring.score_saved_predictions(directory, unseen_outcomes)
            with self.assertRaises(scoring.SavedScoringError):
                scoring.compare_saved_predictions(directory, directory, unseen_query, unseen_outcomes, contract)
        self.assertFalse(unseen_query.exists())
        self.assertFalse(unseen_outcomes.exists())

    def test_all_forbidden_operations_are_trapped_during_reader_and_writer(self):
        directory = self.install()
        target = self.fresh()
        traps = [(groupbench, "preview_group_forward_evaluation"),
                 (groupbench, "evaluate_group_forward_baselines"),
                 (Pipeline, "__init__"), (ColumnTransformer, "__init__"),
                 (DummyRegressor, "__init__"), (DummyRegressor, "fit"), (DummyRegressor, "predict"),
                 (Ridge, "__init__"), (Ridge, "fit"), (Ridge, "predict"),
                 (SimpleImputer, "__init__"), (StandardScaler, "__init__"),
                 (pd, "read_csv"), (pickle, "load"), (pickle, "loads"), (joblib, "load"),
                 (urllib.request, "urlopen"), (socket, "create_connection"),
                 (http.client.HTTPConnection, "connect"), (http.client.HTTPSConnection, "connect"),
                 (report, "render_html"), (report, "write_report"),
                 (random, "random"), (random, "seed"), (random, "getrandbits")]
        traps.extend((np.random, name) for name in
                     ("default_rng", "seed", "rand", "randn", "random", "random_sample",
                      "choice", "permutation", "shuffle", "normal", "uniform", "randint"))
        with ExitStack() as stack:
            mocks = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
                     for owner, name in traps]
            self.assertEqual(validate(directory).validation_fits_performed, 0)
            write(self.result, target)
            for mock in mocks:
                mock.assert_not_called()
        self.assertEqual(validate(target).declared_pipeline_fits, 4)

    def test_public_exports_are_additive_and_current_owned_inventory_is_explicit(self):
        for name in ("GroupForwardBundleError", "GroupForwardBundleValidation",
                     "write_group_forward_evaluation_bundle", "validate_group_forward_evaluation_bundle"):
            self.assertIs(getattr(groupbench, name), getattr(saved, name))
        self.assertEqual(package_source.SOURCE_FILES, tuple(sorted(package_source.SOURCE_FILES)))
        self.assertEqual(len(package_source.SOURCE_FILES), 39)
        self.assertEqual(set(package_source.SOURCE_FILES), set(HISTORIC_37_FILES) | {"group_time_saved.py", "process_contract.py"})
        self.assertEqual(Path(saved.__file__).resolve().parent, Path(groupbench.__file__).resolve().parent)

    def test_historic_source_maps_are_retained_without_current_source_observation(self):
        # Full exact 28--37 legacy memberships and source-absent legacy records
        # are exercised in the historical source fixture module. These are
        # additional saved-temporal examples, not regenerated fitted results.
        for count in range(28, 38):
            document = deepcopy(self.document)
            document["environment"]["package_source"] = source_declaration(HISTORIC_37_FILES[:count])
            rebind(document)
            directory = self.install(document)
            before = (directory / "temporal_result.json").read_bytes()
            with self.subTest(count=count):
                self.assertEqual(validate(directory).n_rows, 9)
                self.assertEqual((directory / "temporal_result.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
