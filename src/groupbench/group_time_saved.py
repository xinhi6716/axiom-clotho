"""Bounded persistence and read-only declared temporal-result consistency.

No model is constructed, fitted or used; no installed or declared source is
observed. Internal consistency is not authentication or evidence of execution.
Existing package imports are intentionally not an import-isolation guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import errno
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from types import MappingProxyType

from . import verification as v
from .group_time_evaluation import (
    GroupForwardEvaluationResult, PROTOCOL, COVERAGE_MODE, MAX_ROWS,
    MAX_FEATURES, MAX_WINDOWS, MAX_WORK, MAX_PLAN_BYTES, MAX_MATRIX_CELLS,
    _LIMITATIONS, _coverage, _digest, _limits, _metrics, _recipes, _same_typed,
)
from .group_time_json import _document, _output_bound, _LIMITATIONS as _PLAN_LIMITATIONS
from .group_time_plan import (
    MIN_TICK, MAX_TICK, GroupSpan, TimeWindow, plan_group_forward_windows,
)
from .report import write_payloads

MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_DEPTH = 16
MAX_VALUES = 2_000_000
_BUNDLE_ID = "mlcalc.group_forward_baseline_bundle"
_RESULT_ID = "mlcalc.group_forward_baseline_evaluation"
_RESULT_KEYS = frozenset(field.name for field in fields(GroupForwardEvaluationResult))
_MEMBERSHIP_KEYS = frozenset({"rows", "groups", "n_rows", "n_groups"})
_PLAN_KEYS = frozenset({
    "schema_id", "schema_version", "protocol", "fits_performed", "coverage_mode",
    "time_unit", "clock_id", "gap", "row_groups", "group_spans", "folds",
    "total", "tested", "untested", "limitations",
})
_PREDICTION_KEYS = frozenset({
    "row_position", "group", "window_index", "observed", "prediction_dummy_mean",
    "prediction_ridge_alpha_1",
})
_ENVIRONMENT_KEYS = frozenset({
    "package_version", "package_source", "python", "numpy", "pandas", "sklearn",
    "scipy", "threadpoolctl",
})
_SOURCE_CONTRACT = {"schema_version": "1.0", "algorithm": "sha256",
                    "scope": "fixed_groupbench_python_sources_v1"}
_CHECKED = (
    "Retained payload SHA-256 and fixed checksum file inventory.",
    "Strict typed v1 bundle and result documents.",
    "Complete reconstructed declared temporal plan and partial test coverage.",
    "Fixed recipe, resource-limit and fit-count declarations.",
    "Tested row/group/window identities and finite observations/predictions.",
    "Retained-prediction metrics within rel_tol=1e-10 and abs_tol=1e-12.",
    "Plan/execution binding recomputation and saved source-map internal digest.",
)
_UNCHECKED = (
    "Authenticity, original row mapping, source truth and source timing.",
    "Original features, missingness, full targets and input digest reconstruction.",
    "Undeclared aliases, target-derived features and feature availability.",
    "Actual training, preprocessing isolation, solver state, executed fit counts and executed-code identity.",
    "Historical predeclaration/blinding, causality, group independence, model quality, deployment generalization and scientific validity.",
    "Filesystem-atomic snapshot and cryptographic authentication.",
)


class GroupForwardBundleError(ValueError):
    """Categorical invalid content or unsupported safe local bundle reading."""


@dataclass(frozen=True, slots=True)
class GroupForwardBundleValidation:
    bundle_schema_version: int
    result_schema_version: int
    protocol: str
    coverage_mode: str
    n_rows: int
    n_groups: int
    n_tested_rows: int
    n_tested_groups: int
    n_untested_rows: int
    n_untested_groups: int
    n_windows: int
    n_features: int
    n_recipes: int
    declared_pipeline_fits: int
    validation_fits_performed: int
    payload_bytes: int
    temporal_result_sha256: str
    checked_scope: tuple[str, ...]
    unchecked_scope: tuple[str, ...]


def _reject(message):
    raise GroupForwardBundleError(message)


def _keys(value, expected):
    if type(value) is not dict or value.keys() != expected:
        _reject("Saved records require exactly the supported fields.")


def _array(value, maximum, minimum=0):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        _reject("Saved arrays exceed their supported type or population bounds.")


def _integer_value(value, minimum=MIN_TICK, maximum=MAX_TICK):
    if type(value) is not int or not minimum <= value <= maximum:
        _reject("Saved integer fields require exact bounded built-in integers.")


def _identity(value):
    if type(value) is str and value.strip():
        return
    _integer_value(value)


def _name(value, maximum=None):
    if (type(value) is not str or not value or value.strip() != value
            or (maximum is not None and len(value) > maximum)):
        _reject("Saved names require bounded nonblank unpadded strings.")


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _reject("Saved digests require lowercase SHA-256 syntax.")


def _float_value(value):
    if type(value) is not float or not math.isfinite(value):
        _reject("Saved numeric observations, predictions and scores require finite binary64 values.")


def _equal(left, right, category):
    if not _same_typed(left, right):
        _reject(category)


def _integer(token):
    if len(token.lstrip("-")) > 19:
        _reject("JSON integer token exceeds signed-64-bit bounds.")
    value = int(token)
    _integer_value(value)
    return value


def _float(token):
    if len(token) > 64:
        _reject("JSON floating token exceeds its length bound.")
    value = float(token)
    if not math.isfinite(value):
        _reject("JSON numeric overflow is unsupported.")
    # JSON supplies valid finite-length decimal syntax here. Nonzero digits in
    # the significand distinguish real underflow from an exact signed zero,
    # without allocating arbitrary-precision exponent/coefficient arithmetic.
    if value == 0.0 and any(char in "123456789" for char in token.lower().split("e", 1)[0]):
        _reject("JSON numeric underflow is unsupported.")
    return value


def _constant(_token):
    _reject("JSON nonfinite constants are unsupported.")


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _reject("Duplicate JSON object keys are unsupported.")
        value[key] = item
    return value


def _preflight_json(text):
    """Count containers/scalars before decoding; syntax still belongs to JSON.

    Object keys are strings immediately followed by a colon outside strings.
    This deliberately does not try to accept or repair malformed JSON syntax.
    """
    depth = count = index = 0
    size = len(text)
    while index < size:
        char = text[index]
        if char in " \t\r\n,:":
            index += 1
            continue
        if char in "[{":
            depth += 1
            count += 1
            index += 1
            if depth > MAX_DEPTH:
                _reject("Saved JSON exceeds its container depth limit.")
        elif char in "]}":
            depth -= 1
            index += 1
        elif char == '"':
            index += 1
            while index < size:
                char = text[index]
                index += 1
                if char == "\\":
                    index += 1
                elif char == '"':
                    break
            following = index
            while following < size and text[following] in " \t\r\n":
                following += 1
            if following >= size or text[following] != ":":
                count += 1
        else:
            count += 1
            while index < size and text[index] not in ' \t\r\n,:[{}]"':
                index += 1
        if count > MAX_VALUES:
            _reject("Saved JSON exceeds its value-count limit.")


def _decode(raw):
    if type(raw) is not bytes or len(raw) > MAX_PAYLOAD_BYTES:
        _reject("Saved payload exceeds its byte limit or has an invalid byte type.")
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        _reject("Saved payload requires valid UTF-8 JSON.")
    if text.startswith("\ufeff"):
        _reject("Saved JSON must not contain a byte-order mark.")
    _preflight_json(text)
    try:
        envelope = json.loads(text, object_pairs_hook=_object, parse_int=_integer,
                              parse_float=_float, parse_constant=_constant)
    except (ValueError, OverflowError, RecursionError):
        _reject("Saved payload requires one complete strict bounded JSON document.")
    _keys(envelope, {"schema_id", "schema_version", "result"})
    _equal(envelope["schema_id"], _BUNDLE_ID, "Saved bundle schema identity is unsupported.")
    _equal(envelope["schema_version"], 1, "Saved bundle schema version is unsupported.")
    return envelope


def _project_result(result):
    """Detach only exact frozen protocol containers, before any conversion hook."""
    if type(result) is not GroupForwardEvaluationResult:
        _reject("Writer requires the exact completed temporal result class.")
    active, count = set(), 4  # envelope/result containers and two envelope scalars
    # Exact compact-ASCII JSON byte accounting bounds the detached projection
    # before json.dumps can allocate an expanded escaped string/document.
    encoded_size = len('{"schema_id":"' + _BUNDLE_ID
                       + '","schema_version":1,"result":}\n')

    def add_bytes(size):
        nonlocal encoded_size
        encoded_size += size
        if encoded_size > MAX_PAYLOAD_BYTES:
            _reject("Frozen result exceeds its encoded-byte limit.")

    def string_bytes(value):
        if len(value) > MAX_PAYLOAD_BYTES - encoded_size:
            _reject("Frozen result exceeds its encoded-byte limit.")
        size = 2
        for char in value:
            code = ord(char)
            size += (2 if char in '\"\\\b\f\n\r\t' else
                     6 if code < 32 or 126 < code <= 65535 else
                     12 if code > 65535 else 1)
            if size > MAX_PAYLOAD_BYTES - encoded_size:
                _reject("Frozen result exceeds its encoded-byte limit.")
        return size

    fixed_fields = fields(GroupForwardEvaluationResult)
    add_bytes(2 + len(fixed_fields) - 1)
    for field in fixed_fields:
        add_bytes(string_bytes(field.name) + 1)

    def project(value, depth):
        nonlocal count
        count += 1
        if count > MAX_VALUES:
            _reject("Frozen result exceeds its value-count limit.")
        kind = type(value)
        if kind in (MappingProxyType, tuple):
            if depth > MAX_DEPTH:
                _reject("Frozen result exceeds its container depth limit.")
            identity = id(value)
            if identity in active:
                _reject("Frozen result contains cyclic containers.")
            active.add(identity)
            if kind is tuple:
                if len(value) > MAX_VALUES - count:
                    _reject("Frozen result exceeds its value-count limit.")
                add_bytes(2 + max(0, len(value) - 1))
                output = [project(item, depth + 1) for item in value]
            else:
                # An exact mappingproxy can still wrap an arbitrary user
                # Mapping. Inspect its GC referent before invoking any mapping
                # operation, and fail closed on unsupported interpreter layouts.
                referents = gc.get_referents(value)
                if len(referents) != 1 or type(referents[0]) is not dict:
                    _reject("Frozen result requires proxies over exact built-in dictionaries.")
                backing = referents[0]
                if len(backing) > MAX_VALUES - count:
                    _reject("Frozen result exceeds its value-count limit.")
                add_bytes(2 + max(0, len(backing) - 1))
                output = {}
                for key, item in backing.items():
                    if type(key) is not str:
                        _reject("Frozen result object keys require exact strings.")
                    add_bytes(string_bytes(key) + 1)
                    output[key] = project(item, depth + 1)
            active.remove(identity)
            return output
        if kind is str:
            add_bytes(string_bytes(value))
        elif kind is int:
            _integer_value(value)
            add_bytes(len(str(value)))
        elif kind is float:
            _float_value(value)
            add_bytes(len(str(value)))
        elif kind not in (bool, type(None)):
            _reject("Frozen result contains an unsupported scalar or mutable container.")
        else:
            add_bytes(4 if value is None or value is True else 5)
        return value

    try:
        # Object attribute access is fixed to an exact slotted class. Never call
        # to_dict, arbitrary Mapping methods or user conversion/iteration hooks.
        return {field.name: project(object.__getattribute__(result, field.name), 3)
                for field in fixed_fields}
    except (AttributeError, TypeError, RecursionError, RuntimeError):
        _reject("Frozen result contains malformed fixed fields.")


def _membership_shape(value, n, g):
    _keys(value, _MEMBERSHIP_KEYS)
    _array(value["rows"], n)
    _array(value["groups"], g)
    _integer_value(value["n_rows"], 0, n)
    _integer_value(value["n_groups"], 0, g)
    for position in value["rows"]:
        _integer_value(position, 0, n - 1)
    for group in value["groups"]:
        _identity(group)


def _plan_shape(raw):
    _keys(raw, _PLAN_KEYS)
    for key, expected in {
        "schema_id": "mlcalc.group_forward_window_plan", "schema_version": 1,
        "protocol": "group-forward-window-plan-v1", "fits_performed": 0,
        "coverage_mode": COVERAGE_MODE, "limitations": list(_PLAN_LIMITATIONS),
    }.items():
        _equal(raw[key], expected, "Saved plan schema or fixed markers are unsupported.")
    _array(raw["row_groups"], MAX_ROWS, 1)
    n = len(raw["row_groups"])
    _array(raw["group_spans"], n, 1)
    _array(raw["folds"], MAX_WINDOWS, 1)
    g, w = len(raw["group_spans"]), len(raw["folds"])
    if w * (n + g) > MAX_WORK:
        _reject("Saved plan exceeds its window-by-population work bound.")
    for key in ("time_unit", "clock_id"):
        _name(raw[key])
    _integer_value(raw["gap"], 0)
    for group in raw["row_groups"]:
        _identity(group)
    for span in raw["group_spans"]:
        _keys(span, {"group", "start", "end", "label_available_at"})
        _identity(span["group"])
        for key in ("start", "end", "label_available_at"):
            _integer_value(span[key])
    for fold in raw["folds"]:
        _keys(fold, {"index", "window", "train", "test", "unused", "exclusions"})
        _integer_value(fold["index"], 0, w - 1)
        _keys(fold["window"], {"lower", "upper"})
        for tick in fold["window"].values():
            _integer_value(tick)
        for key in ("train", "test", "unused"):
            _membership_shape(fold[key], n, g)
        _array(fold["exclusions"], g)
        for exclusion in fold["exclusions"]:
            _keys(exclusion, {"group", "train", "test"})
            _identity(exclusion["group"])
            for key in ("train", "test"):
                if exclusion[key] is not None and type(exclusion[key]) is not str:
                    _reject("Saved plan exclusion types are unsupported.")
    for key in ("total", "tested", "untested"):
        _membership_shape(raw[key], n, g)
    return n, g, w


def _rebuild_plan(raw, shape=None):
    n, g, w = _plan_shape(raw) if shape is None else shape
    # Bound raw text before allocating complete reconstructed memberships.
    text_size = sum(len(group) for group in raw["row_groups"] if type(group) is str)
    text_size += sum(len(span["group"]) for span in raw["group_spans"] if type(span["group"]) is str)
    text_size += len(raw["time_unit"]) + len(raw["clock_id"])
    if text_size > MAX_PLAN_BYTES:
        _reject("Saved plan exceeds its conservative document-size bound.")
    metadata = {"row_groups": raw["row_groups"], "group_spans": raw["group_spans"],
                "windows": [fold["window"] for fold in raw["folds"]],
                "time_unit": raw["time_unit"], "clock_id": raw["clock_id"]}
    if _output_bound(metadata) > MAX_PLAN_BYTES:
        _reject("Saved plan exceeds its conservative document-size bound.")
    try:
        plan = plan_group_forward_windows(
            raw["row_groups"], group_spans=[GroupSpan(**span) for span in raw["group_spans"]],
            windows=[TimeWindow(**fold["window"]) for fold in raw["folds"]],
            time_unit=raw["time_unit"], clock_id=raw["clock_id"], gap=raw["gap"])
    except (ValueError, TypeError, OverflowError):
        _reject("Saved declarations do not form a supported complete forward plan.")
    _equal(raw, _document(plan), "Saved plan differs from its complete canonical reconstruction.")
    all_rows = set(range(n))
    all_groups = {(type(group), group) for group in plan.total.groups}
    tested_rows, tested_groups = set(), set()
    for fold in plan.folds:
        rows = [set(member.rows) for member in (fold.train, fold.test, fold.unused)]
        groups = [{(type(group), group) for group in member.groups}
                  for member in (fold.train, fold.test, fold.unused)]
        if (set.union(*rows) != all_rows or sum(map(len, rows)) != n
                or set.union(*groups) != all_groups or sum(map(len, groups)) != len(all_groups)
                or tested_rows & rows[1] or tested_groups & groups[1]):
            _reject("Saved plan partitions or unique test coverage are inconsistent.")
        tested_rows.update(rows[1])
        tested_groups.update(groups[1])
    if (tested_rows != set(plan.tested.rows)
            or tested_groups != {(type(group), group) for group in plan.tested.groups}
            or all_rows - tested_rows != set(plan.untested.rows)
            or all_groups - tested_groups != {(type(group), group) for group in plan.untested.groups}):
        _reject("Saved tested union or untested complement is inconsistent.")
    return plan


def _record_shapes(document):
    """Reject every malformed nested schema before planning or score arithmetic."""
    n, g, w = _plan_shape(document["plan"])
    coverage = document["coverage"]
    _keys(coverage, {"total", "tested", "untested", "untested_predictions_present"})
    for key in ("total", "tested", "untested"):
        _membership_shape(coverage[key], n, g)
    _equal(coverage["untested_predictions_present"], False,
           "Saved partial coverage must exclude untested predictions.")
    budget = document["fit_budget"]
    operations = {"inner_selection_fits", "full_data_refits", "selected_model_refits", "bootstraps", "retries"}
    totals = {"expected_total", "attempted_total", "completed_total"}
    _keys(budget, {"unit", "per_window", "resource_limits"} | operations | totals)
    _equal(budget["unit"], "pipeline.fit calls", "Saved fit-count unit is unsupported.")
    for key in operations | totals:
        _integer_value(budget[key], 0, 2 * MAX_WINDOWS)
    _array(budget["per_window"], w, w)
    for record in budget["per_window"]:
        _keys(record, {"window_index", "expected", "attempted", "completed"})
        _integer_value(record["window_index"], 0, w - 1)
        for key in ("expected", "attempted", "completed"):
            _integer_value(record[key], 0, 2)
    _keys(budget["resource_limits"], _limits(0).keys())
    for value in budget["resource_limits"].values():
        _integer_value(value, 0)
    _array(document["predictions"], n, 1)
    for record in document["predictions"]:
        _keys(record, _PREDICTION_KEYS)
        _integer_value(record["row_position"], 0, n - 1)
        _identity(record["group"])
        _integer_value(record["window_index"], 0, w - 1)
        for key in ("observed", "prediction_dummy_mean", "prediction_ridge_alpha_1"):
            _float_value(record[key])
    score_keys = {"n_rows", "mae", "rmse", "r2", "bias"}
    pooled_keys = score_keys | {"n_groups", "group_mae", "group_rmse", "worst_group_mae"}

    def score_shape(record, *, pooled, extra):
        _keys(record, (pooled_keys if pooled else score_keys) | extra)
        _integer_value(record["n_rows"], 1, n)
        if pooled:
            _integer_value(record["n_groups"], 1, g)
        for key in (pooled_keys if pooled else score_keys) - {"n_rows", "n_groups"}:
            if key != "r2" or record[key] is not None:
                _float_value(record[key])

    _array(document["metrics"], 2, 2)
    for metric, recipe in zip(document["metrics"], ("dummy_mean", "ridge_alpha_1")):
        _keys(metric, {"recipe", "pooled", "per_window", "per_group"})
        _equal(metric["recipe"], recipe, "Saved metric recipe order is unsupported.")
        score_shape(metric["pooled"], pooled=True, extra=set())
        _array(metric["per_window"], w, w)
        for record in metric["per_window"]:
            score_shape(record, pooled=True, extra={"window_index"})
            _integer_value(record["window_index"], 0, w - 1)
        _array(metric["per_group"], g, 1)
        for record in metric["per_group"]:
            score_shape(record, pooled=False, extra={"group"})
            _identity(record["group"])
    return n, g, w


def _roles(document):
    schema = document["feature_schema"]
    _array(schema, MAX_FEATURES, 1)
    names = []
    dtype_names = {kind + str(bits) for kind in ("int", "uint") for bits in (8, 16, 32, 64)}
    dtype_names.update({"float16", "float32", "float64"})
    for feature in schema:
        _keys(feature, {"name", "source_dtype", "normalized_dtype"})
        _name(feature["name"])
        names.append(feature["name"])
        dtype = feature["source_dtype"]
        if (type(dtype) is not str or len(dtype) > 16
                or (dtype not in dtype_names and re.fullmatch(r"[<>](?:[iu][248]|f[248])", dtype) is None)):
            _reject("Saved feature dtype is not a supported real NumPy declaration.")
        _equal(feature["normalized_dtype"], "float64", "Saved normalized dtype is unsupported.")
    metadata = document["metadata_columns"]
    _array(metadata, MAX_VALUES, 1)
    for name in metadata:
        _name(name)
    target, group = document["target_name"], document["group_name"]
    _name(target)
    _name(group)
    if (len(set(names)) != len(names) or len(set(metadata)) != len(metadata)
            or target == group or target in metadata or group not in metadata
            or target in names or group in names or set(names) & set(metadata)):
        _reject("Saved feature order, names or declared column roles conflict.")


def _environment(environment):
    _keys(environment, _ENVIRONMENT_KEYS)
    for key, value in environment.items():
        if key != "package_source":
            _name(value, 256)
    source = environment["package_source"]
    _keys(source, {"schema_version", "algorithm", "scope", "status", "sha256", "files"})
    for key, value in {**_SOURCE_CONTRACT, "status": "available"}.items():
        _equal(source[key], value, "Saved temporal source declaration is unsupported.")
    _sha(source["sha256"])
    records = source["files"]
    if type(records) is not dict or not 1 <= len(records) <= 256:
        _reject("Saved source file map exceeds its supported population bounds.")
    for name, digest in records.items():
        if (type(name) is not str or not 1 <= len(name) <= 256 or not name.endswith(".py")
                or "\\" in name or "\x00" in name or any(part in ("", ".", "..") for part in name.split("/"))):
            _reject("Saved source file names require inert relative POSIX Python paths.")
        _sha(digest)
    canonical = json.dumps({**_SOURCE_CONTRACT, "files": records}, sort_keys=True,
                           separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != source["sha256"]:
        _reject("Saved source-map aggregate digest is inconsistent.")


def _metric_equal(saved, expected):
    if type(expected) is float:
        _float_value(saved)
        if not math.isclose(saved, expected, rel_tol=1e-10, abs_tol=1e-12):
            _reject("Saved metric scores disagree with retained predictions.")
    elif type(expected) is dict:
        _keys(saved, expected.keys())
        for key in expected:
            _metric_equal(saved[key], expected[key])
    elif type(expected) is list:
        _array(saved, len(expected), len(expected))
        for left, right in zip(saved, expected):
            _metric_equal(left, right)
    else:
        _equal(saved, expected, "Saved metric identities, counts, order or nulls are inconsistent.")


def _check_document(document):
    """Validate declarations, never original inputs or an execution context."""
    _keys(document, _RESULT_KEYS)
    for key, expected in {"schema_id": _RESULT_ID, "schema_version": 1,
                          "protocol": PROTOCOL, "coverage_mode": COVERAGE_MODE,
                          "limitations": list(_LIMITATIONS)}.items():
        _equal(document[key], expected, "Saved result protocol or fixed markers are unsupported.")
    _keys(document["bindings"], {"encoding", "plan_sha256", "input_sha256", "execution_sha256"})
    bindings = document["bindings"]
    _equal(bindings["encoding"], "mlcalc.temporal-binding-v1", "Saved binding encoding is unsupported.")
    for key in ("plan_sha256", "input_sha256", "execution_sha256"):
        _sha(bindings[key])
    _roles(document)
    _environment(document["environment"])
    _equal(document["recipes"], _recipes(), "Saved recipe declarations differ from the fixed protocol.")
    shape = _record_shapes(document)
    plan = _rebuild_plan(document["plan"], shape)
    _equal(document["coverage"], _coverage(plan), "Saved partial coverage differs from the rebuilt plan.")
    cells = 2 * len(document["feature_schema"]) * sum(fold.train.n_rows + fold.test.n_rows for fold in plan.folds)
    if cells > MAX_MATRIX_CELLS:
        _reject("Saved schedule exceeds its matrix-cell visit bound.")
    budget = {"unit": "pipeline.fit calls", "expected_total": 2 * len(plan.folds),
              "attempted_total": 2 * len(plan.folds), "completed_total": 2 * len(plan.folds),
              "per_window": [{"window_index": fold.index, "expected": 2, "attempted": 2, "completed": 2}
                             for fold in plan.folds],
              "inner_selection_fits": 0, "full_data_refits": 0, "selected_model_refits": 0,
              "bootstraps": 0, "retries": 0, "resource_limits": _limits(cells)}
    _equal(document["fit_budget"], budget, "Saved fit-count or resource-limit declarations are inconsistent.")
    records = document["predictions"]
    _array(records, plan.tested.n_rows, plan.tested.n_rows)
    window_by_row = {position: fold.index for fold in plan.folds for position in fold.test.rows}
    dummy_by_window = {}
    for record, position in zip(records, plan.tested.rows):
        _keys(record, _PREDICTION_KEYS)
        _equal(record["row_position"], position, "Saved prediction positions differ from unique tested-row order.")
        _equal(record["group"], plan.row_groups[position], "Saved prediction typed group identity is inconsistent.")
        window = window_by_row[position]
        _equal(record["window_index"], window, "Saved prediction window identity is inconsistent.")
        for key in ("observed", "prediction_dummy_mean", "prediction_ridge_alpha_1"):
            _float_value(record[key])
        bits = struct.pack("<d", record["prediction_dummy_mean"])
        if dummy_by_window.setdefault(window, bits) != bits:
            _reject("Saved Dummy predictions are not constant within their window.")
    # Verify declaration bindings before metric arithmetic. Input binding remains
    # opaque: retained fields cannot reconstruct original features/full targets.
    if bindings["plan_sha256"] != _digest("plan", _document(plan)):
        _reject("Saved complete-plan binding is inconsistent.")
    execution = {"plan_sha256": bindings["plan_sha256"], "input_sha256": bindings["input_sha256"],
                 "protocol": PROTOCOL, "coverage_mode": COVERAGE_MODE,
                 "recipes": document["recipes"], "environment": document["environment"],
                 "resource_limits": _limits(cells)}
    if bindings["execution_sha256"] != _digest("execution", execution):
        _reject("Saved execution-declaration binding is inconsistent.")
    try:
        expected = _metrics(records, plan)
    except (ValueError, TypeError, OverflowError, FloatingPointError):
        _reject("Saved metric arithmetic is unsupported or nonfinite; rescale numeric units.")
    _metric_equal(document["metrics"], expected)
    return plan


def validate_group_forward_evaluation_bundle(run_dir) -> GroupForwardBundleValidation:
    """Check retained bytes and declarations read-only, with zero fits.

    Unknown files are unopened. Known legacy artifacts reject. Trusted idle
    POSIX storage is required; detected races fail, but no atomic snapshot or
    authenticated origin is promised. Declared source paths are never opened.
    """
    try:
        with v._open_run_directory(run_dir) as directory_fd:
            _, _, checksum_bytes, checksum_signature = v._read_file(
                directory_fd, "checksums.json", v.MAX_CHECKSUM_BYTES, collect=True)
            records = v._checksum_records(checksum_bytes)
            if set(records) != v.TEMPORAL_FILES:
                _reject("Saved temporal bundle requires its exact checksum inventory.")
            v._reject_stale_temporal(directory_fd)
            digest, size, raw, signature = v._read_file(
                directory_fd, "temporal_result.json", MAX_PAYLOAD_BYTES, collect=True)
            if digest != records["temporal_result.json"]:
                _reject("Saved temporal payload SHA-256 is inconsistent.")
            document = _decode(raw)["result"]
            plan = _check_document(document)
            v._reject_stale_temporal(directory_fd)
            for name, retained in (("checksums.json", checksum_signature), ("temporal_result.json", signature)):
                if v._signature(os.stat(name, dir_fd=directory_fd, follow_symlinks=False)) != retained:
                    _reject("Saved temporal bundle changed during validation.")
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        # Neither filesystem/parser messages nor declared labels/paths escape.
        raise GroupForwardBundleError("Saved temporal bundle is invalid, changed, or unsupported for safe reading.") from None
    return GroupForwardBundleValidation(
        1, 1, PROTOCOL, COVERAGE_MODE, plan.total.n_rows, plan.total.n_groups,
        plan.tested.n_rows, plan.tested.n_groups, plan.untested.n_rows, plan.untested.n_groups,
        len(plan.folds), len(document["feature_schema"]), 2, 2 * len(plan.folds), 0,
        size, digest, _CHECKED, _UNCHECKED)


def write_group_forward_evaluation_bundle(result, output_dir) -> Path:
    """Validate fully, then publish only to a fresh leaf in an existing parent.

    Publication is per-file no-clobber, not directory-atomic or race-free. No
    fsync guarantee, cleanup or retry: I/O failure can leave an incomplete new
    directory, which must not be reused. Parent storage must be trusted/idle.
    """
    try:
        document = _project_result(result)
        _check_document(document)
        envelope = {"schema_id": _BUNDLE_ID, "schema_version": 1, "result": document}
        text = json.dumps(envelope, sort_keys=True, ensure_ascii=True, allow_nan=False,
                          separators=(",", ":")) + "\n"
        raw = text.encode("utf-8")
        # Full encoded-byte and strict transport checks before the first write.
        _check_document(_decode(raw)["result"])
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise GroupForwardBundleError("Completed temporal result is invalid or exceeds persistence limits.") from None
    try:
        output = Path(output_dir)
        if (not output.name or output.name in (".", "..") or ".." in output.parts
                or output.anchor not in ("", "/") or len(os.fsencode(output)) > 4096
                or len(output.parts) > 256):
            _reject("Writer destination requires a bounded direct fresh-leaf path.")
        # Preflight only: the unchanged generic publisher remains path-based.
        with v._open_run_directory(output.parent) as parent_fd:
            try:
                os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError("Temporal writer destination already exists.")
    except FileExistsError:
        raise FileExistsError("Temporal writer destination already exists.") from None
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
            raise OSError("Temporal writer could not inspect its destination parent.") from None
        raise GroupForwardBundleError("Temporal writer requires a safe existing POSIX parent directory.") from None
    except (ValueError, TypeError):
        raise GroupForwardBundleError("Temporal writer requires a safe existing POSIX parent directory.") from None
    # Exclusivity at this leaf is mandatory; never reuse a racing directory.
    try:
        output.mkdir()
        write_payloads({"temporal_result.json": text}, output, overwrite=False)
    except FileExistsError:
        raise FileExistsError("Temporal writer destination or payload already exists.") from None
    except OSError:
        raise OSError("Temporal bundle publication failed; a new incomplete directory may remain.") from None
    return output / "temporal_result.json"
