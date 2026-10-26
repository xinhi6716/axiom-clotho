"""Opt-in bounded consistency checks, not scientific validation or model loading."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
import os
import re

from .verification import (
    MAX_CHECKSUM_BYTES, PREDICTION_FILES, _checksum_records,
    _open_run_directory, _read_file, _reject_stale_evaluation, _signature,
)

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PREDICTION_CSV_BYTES = 4 * 1024 * 1024
MAX_ROWS = 20000
MAX_FEATURES = 100
_FLAGS = ("missing_value", "numeric_outside_training_range", "unseen_category",
          "non_missing_value_in_all_missing_training_feature")
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


@dataclass(frozen=True)
class PredictionSchemaResult:
    contract_version: str
    schema_version: str
    n_predictions: int
    n_features: int
    input_support: str
    payload_bytes: int
    checked_scope: str = "checksums_prediction_rows_feature_names_support_count_consistency"
    unchecked_metadata: str = "other_manifest_fields_not_validated"


def _fail(message):
    raise ValueError("Prediction bundle schema: " + message)


def _count(value, maximum, *, positive=False):
    if type(value) is not int or not (int(positive) <= value <= maximum):
        _fail("count is not an integer in its documented population bound.")
    return value


def _object(value):
    if not isinstance(value, dict):
        _fail("required metadata object is missing or invalid.")
    return value


def _unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            _fail("duplicate JSON object keys are forbidden.")
        obj[key] = value
    return obj


def _number(token):
    if len(token) > 100:
        _fail("JSON numeric token exceeds 100 characters.")
    value = float(token)
    if not math.isfinite(value):
        _fail("JSON numeric values must be finite.")
    return value


def _integer(token):
    if len(token) > 100:
        _fail("JSON numeric token exceeds 100 characters.")
    return int(token)


def _constant(_token):
    _fail("JSON nonfinite constants are forbidden.")


def _manifest(raw):
    try:
        return _object(json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                                  parse_float=_number, parse_int=_integer,
                                  parse_constant=_constant))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Prediction bundle schema: invalid bounded UTF-8 JSON.") from exc


def _text(value):
    return isinstance(value, str) and bool(value) and "\0" not in value


def _union(union, components, population):
    """Necessary bounds only. Overlap cannot be reconstructed from aggregates."""
    if not max(components, default=0) <= union <= min(population, sum(components)):
        _fail("aggregate union count is inconsistent with component counts.")


def _support(support, names, n_train, n_pred):
    support = _object(support)
    if (support.get("schema_version") != "1.0"
            or support.get("reference") != "full_training_features_before_imputation"
            or support.get("action") != "diagnostic_only_no_input_changes"):
        _fail("unsupported input-support schema or reference/action.")
    if (_count(support.get("n_training_rows"), MAX_ROWS, positive=True) != n_train
            or _count(support.get("n_prediction_rows"), MAX_ROWS, positive=True) != n_pred):
        _fail("input-support row counts differ from manifest counts.")
    flags = _object(support.get("row_flag_counts"))
    if set(flags) != set(_FLAGS):
        _fail("input-support row flag keys must match version 1.0.")
    counts = {key: _count(flags[key], n_pred) for key in _FLAGS}
    any_flag = _count(support.get("rows_with_any_flag"), n_pred)
    _union(any_flag, list(counts.values()), n_pred)
    features = support.get("features")
    if not isinstance(features, list) or len(features) != len(names):
        _fail("input-support feature count differs from feature schema.")
    components = {key: [] for key in _FLAGS}
    per_feature_flagged = []
    for name, item in zip(names, features):
        item = _object(item)
        if item.get("name") != name or item.get("kind") not in ("numeric", "categorical"):
            _fail("input-support feature names/order or kind are inconsistent.")
        train_missing = _count(item.get("training_missing_count"), n_train)
        missing = _count(item.get("prediction_missing_count"), n_pred)
        unsupported = _count(item.get("prediction_non_missing_without_training_values_count"), n_pred)
        all_missing = train_missing == n_train
        expected = "unavailable_all_missing" if all_missing else "observed_non_missing_values"
        if item.get("training_support") != expected or unsupported != (n_pred - missing if all_missing else 0):
            _fail("feature missingness/support counts are inconsistent.")
        components["missing_value"].append(missing)
        components["non_missing_value_in_all_missing_training_feature"].append(unsupported)
        if item["kind"] == "numeric":
            if "unseen_category_count" in item:
                _fail("numeric support contains a categorical count.")
            # Require keys even when their values must be null.
            if not {"below_training_min_count", "above_training_max_count"} <= set(item):
                _fail("numeric support count keys are missing.")
            below, above = item["below_training_min_count"], item["above_training_max_count"]
            if all_missing:
                if below is not None or above is not None:
                    _fail("all-missing numeric support must use null range counts.")
                outside = 0
            else:
                below, above = _count(below, n_pred), _count(above, n_pred)
                outside = below + above
                if outside + missing > n_pred:
                    _fail("numeric range and missing counts exceed the population.")
            components["numeric_outside_training_range"].append(outside)
            # Missing, observed-range violations, and unsupported non-missing
            # values are disjoint within one numeric feature.
            per_feature_flagged.append(missing + outside + unsupported)
        else:
            if {"below_training_min_count", "above_training_max_count"} & set(item):
                _fail("categorical support contains numeric range counts.")
            unseen = _count(item.get("unseen_category_count"), n_pred)
            if unseen + missing > n_pred or (all_missing and unseen != n_pred - missing):
                _fail("categorical unseen/missing counts are inconsistent.")
            components["unseen_category"].append(unseen)
            # Unsupported all-missing-source values already count as unseen.
            per_feature_flagged.append(missing + unseen)
    for key in _FLAGS:
        _union(counts[key], components[key], n_pred)
    if any_flag < max(per_feature_flagged, default=0):
        _fail("aggregate union is below a feature's necessarily distinct flagged rows.")


def _metadata(manifest):
    version = manifest.get("schema_version")
    if (manifest.get("mode") != "fit_predict" or "schema_id" in manifest
            or version not in ("1.0", "1.1", "1.2")):
        _fail("only supported regression fit_predict schemas 1.0/1.1/1.2 are accepted.")
    config = _object(manifest.get("config"))
    if version == "1.2":
        if (manifest.get("protocol") != "group-fit-predict-distinct-v2"
                or config.get("categorical_encoding") != "distinct-v2"):
            _fail("schema 1.2 requires the distinct-v2 prediction protocol.")
    elif "protocol" in manifest or "categorical_encoding" in config:
        _fail("legacy schema must not declare an incompatible encoding protocol.")
    n_train = _count(manifest.get("n_training_rows"), MAX_ROWS, positive=True)
    _count(manifest.get("n_training_groups"), n_train, positive=True)
    n_pred = _count(manifest.get("n_predictions"), MAX_ROWS, positive=True)
    features = manifest.get("feature_schema")
    if not isinstance(features, list) or not 1 <= len(features) <= MAX_FEATURES:
        _fail("feature schema must contain 1 to 100 records.")
    names = []
    for record in features:
        record = _object(record)
        if not _text(record.get("name")) or not _text(record.get("dtype")):
            _fail("feature name/dtype must be nonempty strings without NUL.")
        names.append(record["name"])
    if len(set(names)) != len(names) or config.get("features") != names:
        _fail("feature names must be unique and match ordered config.features.")
    if version == "1.0":
        if "input_support" in manifest:
            _fail("schema 1.0 does not declare the input-support contract.")
        support_status = "unavailable"
    else:
        _support(manifest.get("input_support"), names, n_train, n_pred)
        support_status = "count_consistency_checked_not_recomputed"
    return version, n_pred, len(names), support_status


def _predictions(raw, n_predictions):
    try:
        text = raw.decode("utf-8")
        if "\0" in text:
            _fail("CSV NUL bytes are forbidden.")
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        if next(rows, None) != ["row_position", "prediction"]:
            _fail("CSV header must be exactly row_position,prediction.")
        count = 0
        for row in rows:
            if count >= n_predictions or len(row) != 2 or row[0] != str(count):
                _fail("CSV rows must match sequential zero-based positions and declared row count.")
            token = row[1]
            if len(token) > 64 or _DECIMAL.fullmatch(token) is None or not math.isfinite(float(token)):
                _fail("CSV predictions must be bounded finite decimal numbers.")
            count += 1
        if count != n_predictions:
            _fail("CSV row count differs from n_predictions.")
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("Prediction bundle schema: invalid bounded UTF-8 CSV.") from exc


def validate_prediction_bundle_schema(run_dir) -> PredictionSchemaResult:
    """Check exact hashed bytes and a versioned prediction-metadata subset.

    Limits: 16 KiB checksums, 1 MiB manifest, 4 MiB CSV, 20,000 rows,
    100 features. Extra manifest fields are NOT schema validated. Unlisted
    files are not read or checked. This cannot establish scientific accuracy,
    source authenticity, full schema correctness, or an atomic snapshot.
    """
    with _open_run_directory(run_dir) as fd:
        _, _, checksum_raw, signature = _read_file(fd, "checksums.json", MAX_CHECKSUM_BYTES, collect=True)
        records = _checksum_records(checksum_raw)
        if set(records) != PREDICTION_FILES:
            _fail("evaluation bundles are not supported by this prediction-only check.")
        _reject_stale_evaluation(fd)
        signatures = {"checksums.json": signature}
        payloads, total = {}, 0
        for name, limit in (("manifest.json", MAX_MANIFEST_BYTES), ("predictions.csv", MAX_PREDICTION_CSV_BYTES)):
            digest, size, raw, signature = _read_file(fd, name, limit, collect=True)
            if digest != records[name]:
                _fail("payload SHA-256 mismatch.")
            signatures[name], payloads[name] = signature, raw
            total += size
        version, n_pred, n_features, support = _metadata(_manifest(payloads["manifest.json"]))
        _predictions(payloads["predictions.csv"], n_pred)
        for name, signature in signatures.items():
            if _signature(os.stat(name, dir_fd=fd, follow_symlinks=False)) != signature:
                _fail("bundle changed during validation; retry with an idle directory.")
        _reject_stale_evaluation(fd)
    return PredictionSchemaResult("1.0", version, n_pred, n_features, support, total)
