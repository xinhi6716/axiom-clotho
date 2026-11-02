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


def _kernel_details(value, rows, seed, alpha=None):
    """Check bounded saved metadata consistency, not authenticity or execution."""
    value = _object(value)
    keys = {"implementation", "training_rows", "transformed_features", "rank_cap",
            "effective_n_components", "gamma_policy", "gamma", "random_state",
            "basis_sampling", "basis_indices_sha256", "alpha", "solver", "threads",
            "training_map_bytes", "map_byte_limit"}
    if set(value) != keys:
        _fail("extended kernel detail fields differ.")
    width = _count(value["transformed_features"], 6400, positive=True)
    k = min(64, rows)
    constants = {"implementation": "bounded-nystroem-rbf-ridge-v1", "training_rows": rows,
                 "rank_cap": 64, "effective_n_components": k,
                 "gamma_policy": "inverse_transformed_width", "random_state": seed,
                 "basis_sampling": "uniform_training_rows_without_replacement",
                 "solver": "svd", "threads": 1, "training_map_bytes": rows * k * 8,
                 "map_byte_limit": 16 * 1024 * 1024}
    for key, expected in constants.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            _fail("extended kernel details conflict with the saved schedule.")
    gamma = value["gamma"]
    if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma != 1.0 / width:
        _fail("extended kernel bandwidth differs from transformed width.")
    if type(value["alpha"]) not in (int, float) or value["alpha"] not in (0.1, 1.0, 10.0) or (alpha is not None and value["alpha"] != alpha):
        _fail("extended kernel regularization differs from the declared grid.")
    digest = value["basis_indices_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        _fail("extended kernel basis digest must be a SHA-256 token.")
    return value


def _median_details(value, rows, groups):
    value = _object(value)
    if set(value) != {"implementation", "training_rows", "training_groups",
                      "transformed_features", "constant", "weight_rule", "tie_break"}:
        _fail("extended median detail fields differ.")
    _count(value["transformed_features"], 6400, positive=True)
    expected = {"implementation": "group-balanced-lower-median-v1", "training_rows": rows,
                "training_groups": groups, "weight_rule": "equal_training_group_then_equal_row",
                "tie_break": "lower_observed_target"}
    for key, item in expected.items():
        if type(value[key]) is not type(item) or value[key] != item:
            _fail("extended median details conflict with the saved schedule.")
    if type(value["constant"]) not in (int, float) or not math.isfinite(value["constant"]):
        _fail("extended median constant must be finite.")


def _extension_families(value, config):
    baselines = config.get("baselines")
    known = ("dummy", "ridge", "extra_trees", "nystroem_ridge", "group_median")
    if (not isinstance(baselines, list) or not baselines
            or any(not isinstance(f, str) or f not in known for f in baselines)
            or len(set(baselines)) != len(baselines)):
        _fail("extended baseline coverage differs.")
    optional = [family for family in ("group_median", "nystroem_ridge") if family in baselines]
    if (not optional or value.get("capabilities") != [family + "-v1" for family in optional]
            or config.get("categorical_encoding") not in ("legacy-v1", "distinct-v2")):
        _fail("extended capabilities or encoding differ from requested families.")
    return optional


def _optional_selection_metadata(value, rows, groups, inner_splits, seed, optional):
    """Check saved extension details/counts, never authenticate or replay fits."""
    _count(seed, 2**32 - 1)
    audits = value.get("inner_folds")
    if not isinstance(audits, list) or len(audits) != inner_splits:
        _fail("extended inner fold coverage differs.")
    for fold, audit in enumerate(audits):
        audit = _object(audit)
        train = _count(audit.get("train_rows"), rows, positive=True)
        test = _count(audit.get("test_rows"), rows, positive=True)
        train_groups = _count(audit.get("train_group_count"), min(train, groups), positive=True)
        test_groups = _count(audit.get("test_group_count"), min(test, groups), positive=True)
        if train_groups + test_groups != groups:
            _fail("extended inner fold group counts differ.")
        if type(audit.get("fold")) is not int or audit["fold"] != fold or train + test != rows:
            _fail("extended inner fold row counts differ.")
    records = value.get("inner_selection")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        _fail("extended selection records are missing or invalid.")
    for family in optional:
        family_records = [record for record in records if record.get("family") == family]
        alphas = (0.1, 1.0, 10.0) if family == "nystroem_ridge" else (None,)
        if len(family_records) != len(alphas):
            _fail("extended candidate coverage differs.")
        for record, alpha in zip(family_records, alphas):
            expected = f"nystroem_ridge_alpha_{alpha:g}" if alpha is not None else "group_median_lower"
            if record.get("candidate") != expected:
                _fail("extended candidate order differs.")
            details = record.get("inner_fit_details")
            if not isinstance(details, list) or len(details) != inner_splits:
                _fail("extended per-fit coverage differs.")
            for fold, (item, audit) in enumerate(zip(details, audits)):
                item = _object(item)
                for key, expected in (("inner_fold", fold), ("train_rows", audit["train_rows"]), ("test_rows", audit["test_rows"])):
                    if type(item.get(key)) is not int or item[key] != expected:
                        _fail("extended per-fit row counts differ.")
                if family == "nystroem_ridge":
                    _kernel_details(item.get("details"), audit["train_rows"], seed, alpha)
                else:
                    _median_details(item.get("details"), audit["train_rows"], audit["train_group_count"])
    final = _object(value.get("family_fit_details"))
    if set(final) != set(optional):
        _fail("extended final-refit coverage differs.")
    for family in optional:
        if family == "nystroem_ridge":
            _kernel_details(final[family], rows, seed)
        else:
            _median_details(final[family], rows, groups)


def _metadata(manifest):
    version = manifest.get("schema_version")
    if (manifest.get("mode") != "fit_predict" or "schema_id" in manifest
            or version not in ("1.0", "1.1", "1.2", "1.3")):
        _fail("only supported regression fit_predict schemas 1.0/1.1/1.2/1.3 are accepted.")
    config = _object(manifest.get("config"))
    if version == "1.3":
        if manifest.get("protocol") != "group-fit-predict-extended-families-v1":
            _fail("schema 1.3 requires the extended-family prediction protocol.")
        optional = _extension_families(manifest, config)
        if manifest.get("selected_family") not in config["baselines"]:
            _fail("extended selected-family coverage differs.")
    elif version == "1.2":
        if (manifest.get("protocol") != "group-fit-predict-distinct-v2"
                or config.get("categorical_encoding") != "distinct-v2"):
            _fail("schema 1.2 requires the distinct-v2 prediction protocol.")
    elif "protocol" in manifest or "categorical_encoding" in config:
        _fail("legacy schema must not declare an incompatible encoding protocol.")
    if version != "1.3" and isinstance(config.get("baselines"), list) and any(
            family in config["baselines"] for family in ("nystroem_ridge", "group_median")):
        _fail("optional families require the extended prediction schema.")
    n_train = _count(manifest.get("n_training_rows"), MAX_ROWS, positive=True)
    _count(manifest.get("n_training_groups"), n_train, positive=True)
    n_pred = _count(manifest.get("n_predictions"), MAX_ROWS, positive=True)
    if version == "1.3":
        inner = _count(config.get("inner_splits"), manifest["n_training_groups"], positive=True)
        if inner < 2:
            _fail("extended kernel requires at least two inner folds.")
        _optional_selection_metadata(manifest, n_train, manifest["n_training_groups"], inner, config.get("random_state"), optional)
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


def _load_prediction_bundle(run_dir):
    """Check exact hashed bytes and a versioned prediction-metadata subset.

    Limits: 16 KiB checksums, 1 MiB manifest, 4 MiB CSV, 20,000 rows,
    100 features. Extra manifest fields are NOT schema validated. Unlisted
    files are not read or checked. This cannot establish scientific accuracy,
    source authenticity, full schema correctness, or an atomic snapshot.
    """
    with _open_run_directory(run_dir) as fd:
        checksum_digest, _, checksum_raw, signature = _read_file(fd, "checksums.json", MAX_CHECKSUM_BYTES, collect=True)
        records = _checksum_records(checksum_raw)
        if set(records) != PREDICTION_FILES:
            _fail("evaluation bundles are not supported by this prediction-only check.")
        _reject_stale_evaluation(fd)
        signatures = {"checksums.json": signature}
        payloads, total = {}, 0
        hashes = {"checksums.json": checksum_digest}
        for name, limit in (("manifest.json", MAX_MANIFEST_BYTES), ("predictions.csv", MAX_PREDICTION_CSV_BYTES)):
            digest, size, raw, signature = _read_file(fd, name, limit, collect=True)
            if digest != records[name]:
                _fail("payload SHA-256 mismatch.")
            signatures[name], payloads[name], hashes[name] = signature, raw, digest
            total += size
        manifest = _manifest(payloads["manifest.json"])
        version, n_pred, n_features, support = _metadata(manifest)
        _predictions(payloads["predictions.csv"], n_pred)
        for name, signature in signatures.items():
            if _signature(os.stat(name, dir_fd=fd, follow_symlinks=False)) != signature:
                _fail("bundle changed during validation; retry with an idle directory.")
        _reject_stale_evaluation(fd)
    result = PredictionSchemaResult("1.0", version, n_pred, n_features, support, total)
    return result, manifest, payloads["predictions.csv"], hashes


def validate_prediction_bundle_schema(run_dir) -> PredictionSchemaResult:
    """Validate exact saved bytes and bounded metadata; never prove execution.

    Additional extension details are checked for internal count/type consistency,
    not against authentic training data. See the internal loader for byte limits.
    """
    return _load_prediction_bundle(run_dir)[0]
