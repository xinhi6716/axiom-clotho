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


def _extension_families(value, config, *, allow_empty=False):
    baselines = config.get("baselines")
    known = ("dummy", "ridge", "extra_trees", "nystroem_ridge", "group_median")
    if (not isinstance(baselines, list) or not baselines
            or any(not isinstance(f, str) or f not in known for f in baselines)
            or len(set(baselines)) != len(baselines)):
        _fail("extended baseline coverage differs.")
    optional = [family for family in ("group_median", "nystroem_ridge") if family in baselines]
    if ((not optional and not allow_empty) or value.get("capabilities") != [family + "-v1" for family in optional]
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
        if family == "nystroem_ridge":
            alphas = (0.1, 1.0, 10.0)
        elif family == "group_median":
            alphas = (None,)
        else:
            _fail("unknown optional family in the v1 detail contract.")
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
                elif family == "group_median":
                    _median_details(item.get("details"), audit["train_rows"], audit["train_group_count"])
    final = _object(value.get("family_fit_details"))
    if set(final) != set(optional):
        _fail("extended final-refit coverage differs.")
    for family in optional:
        if family == "nystroem_ridge":
            _kernel_details(final[family], rows, seed)
        elif family == "group_median":
            _median_details(final[family], rows, groups)
        else:
            _fail("unknown optional family in the v1 detail contract.")


_DOMAIN_FIELDS = {"inner_strategy", "domain", "domain_values_sha256", "group_domains", "n_domains", "inner_splits_per_outer"}


# Frozen domain-inner-v1 format contract, not a lookup of installed model defaults.
# A different candidate grid requires a new protocol/schema contract.
_DOMAIN_GRID_V1 = {
    "dummy": (("dummy_mean", {"strategy": "mean"}),),
    "ridge": tuple((f"ridge_alpha_{a:g}", {"alpha": a}) for a in (0.1, 1.0, 10.0)),
    "extra_trees": tuple((f"extra_trees_depth_{d or 'none'}",
                          {"max_depth": d, "n_estimators": 64, "min_samples_leaf": 2}) for d in (6, None)),
    "group_median": (("group_median_lower", {}),),
    "nystroem_ridge": tuple((f"nystroem_ridge_alpha_{a:g}", {"alpha": a, "rank_cap": 64})
                           for a in (0.1, 1.0, 10.0)),
}
_DOMAIN_AUDIT_FIELDS = {"train_domains", "test_domains", "train_domain_count",
                        "test_domain_count", "domain_overlap_count"} | _DOMAIN_FIELDS


def _reject_domain_audits(value):
    """Reject explicit incompatible declarations, not unrelated legacy metadata."""
    if _DOMAIN_AUDIT_FIELDS & set(value):
        _fail("domain audit fields require a domain selection schema.")
    audits = value.get("inner_folds")
    if isinstance(audits, list):
        for audit in audits:
            if isinstance(audit, dict) and _DOMAIN_AUDIT_FIELDS & set(audit):
                _fail("domain audit fields require a domain selection schema.")


def _domain_candidates(config):
    return [{"name": name, "family": family, "params": params}
            for family in config["baselines"] for name, params in _DOMAIN_GRID_V1[family]]


def _same_declaration(actual, expected):
    # Reject bools masquerading as numeric parameters, but allow JSON 1 / 1.0.
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_same_declaration(actual[k], v) for k, v in expected.items()))
    if type(expected) in (int, float):
        return type(actual) in (int, float) and actual == expected
    return type(actual) is type(expected) and actual == expected


# Frozen group-only extended-families-v2 contracts. Keep these declarations
# separate from both installed model defaults and the domain-inner-v1 grid.
_EXTENDED_FAMILIES_V2 = ("dummy", "ridge", "extra_trees", "nystroem_ridge",
                         "group_median", "group_balanced_ridge")
_EXTENDED_CAPABILITIES_V2 = ("group_median", "nystroem_ridge", "group_balanced_ridge")
_EXTENDED_GRID_V2 = {
    "dummy": (("dummy_mean", {"strategy": "mean"}),),
    "ridge": tuple((f"ridge_alpha_{a:g}", {"alpha": a}) for a in (0.1, 1.0, 10.0)),
    "extra_trees": tuple((f"extra_trees_depth_{d or 'none'}",
                          {"max_depth": d, "n_estimators": 64, "min_samples_leaf": 2}) for d in (6, None)),
    "group_median": (("group_median_lower", {}),),
    "nystroem_ridge": tuple((f"nystroem_ridge_alpha_{a:g}", {"alpha": a, "rank_cap": 64})
                           for a in (0.1, 1.0, 10.0)),
    "group_balanced_ridge": tuple((f"group_balanced_ridge_alpha_{a:g}", {"alpha": a})
                                 for a in (0.1, 1.0, 10.0)),
}
_GROUP_FIT_AUDIT_V2 = frozenset({"fold", "train_rows", "test_rows", "train_group_count",
                               "test_group_count", "overlap_count", "train_groups", "test_groups"})
_SELECTION_RECORD_V2 = frozenset({"candidate", "family", "params", "selection_score",
                                "objective", "inner_fold_scores"})
_RIDGE_DETAILS_V1 = frozenset({"implementation", "training_rows", "training_groups",
    "transformed_features", "alpha", "solver", "tol", "max_iter", "fit_intercept", "copy_X",
    "positive", "threads", "fitting_loss", "weight_rule", "normalization",
    "preprocessing_weighting", "min_training_group_size", "max_training_group_size",
    "min_row_weight", "max_row_weight", "sum_row_weights"})
_WEIGHT_REL_TOL = 64 * 2.0 ** -52


def _reject_ridge_v2_markers(value):
    """Reject explicit new-family declarations when a bundle claims an old schema."""
    if not isinstance(value, dict):
        return
    objects = [value]
    outer = value.get("outer_folds")
    if isinstance(outer, list):
        objects.extend(item for item in outer if isinstance(item, dict))
    config = value.get("config")
    if isinstance(config, dict):
        baselines = config.get("baselines")
        if isinstance(baselines, list) and "group_balanced_ridge" in baselines:
            _fail("group_balanced_ridge requires the extended-families-v2 schema.")
        candidates = config.get("candidates")
        if isinstance(candidates, list):
            objects.extend(item for item in candidates if isinstance(item, dict))
    for obj in objects:
        caps = obj.get("capabilities")
        if isinstance(caps, list) and "group_balanced_ridge-v1" in caps:
            _fail("group_balanced_ridge capability requires the extended-families-v2 schema.")
        records = obj.get("inner_selection")
        inspected = [obj] + (records if isinstance(records, list) else [])
        final = obj.get("family_fit_details")
        winners = obj.get("family_candidates")
        if any(isinstance(d, dict) and "group_balanced_ridge" in d for d in (final, winners)):
            _fail("group_balanced_ridge refit metadata requires the extended-families-v2 schema.")
        if isinstance(final, dict):
            inspected.extend(final.values())
        for item in inspected:
            if not isinstance(item, dict):
                continue
            for key in ("name", "candidate", "family", "selected_candidate", "selected_family", "implementation"):
                token = item.get(key)
                if isinstance(token, str) and (token.startswith("group_balanced_ridge") or token == "group-balanced-ridge-v1"):
                    _fail("group_balanced_ridge metadata requires the extended-families-v2 schema.")
            details = item.get("inner_fit_details")
            if isinstance(details, list):
                for fit in details:
                    if (isinstance(fit, dict) and isinstance(fit.get("details"), dict)
                            and fit["details"].get("implementation") == "group-balanced-ridge-v1"):
                        _fail("group_balanced_ridge fit details require the extended-families-v2 schema.")


def _extension_families_v2(value, config):
    baselines = config.get("baselines")
    if (not isinstance(baselines, list) or not baselines
            or any(type(f) is not str or f not in _EXTENDED_FAMILIES_V2 for f in baselines)
            or len(set(baselines)) != len(baselines) or "group_balanced_ridge" not in baselines):
        _fail("extended-families-v2 requires explicit group_balanced_ridge and known unique families.")
    optional = [f for f in _EXTENDED_CAPABILITIES_V2 if f in baselines]
    if (value.get("capabilities") != [f + "-v1" for f in optional]
            or config.get("categorical_encoding") not in ("legacy-v1", "distinct-v2")):
        _fail("extended-families-v2 capabilities or encoding differ.")
    if type(config.get("threads")) is not int or config["threads"] != 1:
        _fail("extended-families-v2 requires the single-thread recipe.")
    candidates = [{"name": name, "family": family, "params": params}
                  for family in baselines for name, params in _EXTENDED_GRID_V2[family]]
    declared = config.get("candidates")
    if (not isinstance(declared, list) or len(declared) != len(candidates)
            or any(not _same_declaration(a, b) for a, b in zip(declared, candidates))):
        _fail("extended-families-v2 candidate declaration differs from the frozen grid.")
    return optional, candidates


def _balanced_ridge_details(value, rows, groups, alpha, group_sizes=None):
    """Validate finite summary arithmetic; metadata cannot authenticate fitting."""
    value = _object(value)
    if set(value) != _RIDGE_DETAILS_V1:
        _fail("group-balanced Ridge detail fields differ.")
    _count(value["transformed_features"], 6400, positive=True)
    expected = {"implementation": "group-balanced-ridge-v1", "training_rows": rows,
        "training_groups": groups, "solver": "lsqr", "max_iter": None,
        "fit_intercept": True, "copy_X": True, "positive": False, "threads": 1,
        "fitting_loss": "weighted_squared_error", "weight_rule": "equal_training_group_then_equal_row",
        "normalization": "sum_training_rows", "preprocessing_weighting": "unweighted_training_rows"}
    for key, item in expected.items():
        if type(value[key]) is not type(item) or value[key] != item:
            _fail("group-balanced Ridge details conflict with the declared fit recipe or counts.")
    for key, item in (("alpha", alpha), ("tol", 1e-4)):
        if type(value[key]) not in (int, float) or value[key] != item:
            _fail("group-balanced Ridge alpha or tolerance differs from the declared recipe.")
    smallest = _count(value["min_training_group_size"], rows, positive=True)
    largest = _count(value["max_training_group_size"], rows, positive=True)
    if (smallest > largest or not groups * smallest <= rows <= groups * largest
            or (groups == 1 and smallest != largest)
            or (groups > 1 and not smallest + largest + (groups - 2) * smallest <= rows
                <= smallest + largest + (groups - 2) * largest)):
        _fail("group-balanced Ridge group-size summaries cannot cover the declared rows.")
    if group_sizes is not None and (smallest != min(group_sizes) or largest != max(group_sizes)):
        _fail("group-balanced Ridge group-size summaries differ from retained rows.")
    for key, target in (("min_row_weight", rows / (groups * largest)),
                        ("max_row_weight", rows / (groups * smallest)), ("sum_row_weights", float(rows))):
        actual = value[key]
        if (type(actual) not in (int, float) or not math.isfinite(actual) or actual <= 0
                or not math.isclose(actual, target, rel_tol=_WEIGHT_REL_TOL, abs_tol=0.0)):
            _fail("group-balanced Ridge weight summaries conflict with the normalized weight rule.")
    return value


def _extended_selection_v2(value, rows, groups, inner, seed, config, *, group_sizes=None):
    """Strict v2 declarations and retained-count checks, without replaying fits."""
    _count(seed, 2**32 - 1)
    optional = [f for f in _EXTENDED_CAPABILITIES_V2 if f in config["baselines"]]
    candidates = [{"name": name, "family": family, "params": params}
                  for family in config["baselines"] for name, params in _EXTENDED_GRID_V2[family]]
    audits = value.get("inner_folds")
    if not isinstance(audits, list) or len(audits) != inner:
        _fail("extended-families-v2 inner fold coverage differs.")
    population, tested, test_rows = None, set(), 0
    for fold, audit in enumerate(audits):
        audit = _object(audit)
        if set(audit) != _GROUP_FIT_AUDIT_V2:
            _fail("extended-families-v2 group fit audit fields differ.")
        train = _count(audit["train_rows"], rows, positive=True)
        test = _count(audit["test_rows"], rows, positive=True)
        ng_train = _count(audit["train_group_count"], min(train, groups), positive=True)
        ng_test = _count(audit["test_group_count"], min(test, groups), positive=True)
        if (type(audit["fold"]) is not int or audit["fold"] != fold
                or type(audit["overlap_count"]) is not int or audit["overlap_count"] != 0
                or train + test != rows or ng_train + ng_test != groups):
            _fail("extended-families-v2 inner fold counts differ.")
        tr, te = audit["train_groups"], audit["test_groups"]
        if (not isinstance(tr, list) or not isinstance(te, list)
                or any(not _text(g) or not g.strip() for g in tr + te)
                or len(tr) != ng_train or len(te) != ng_test
                or len(set(tr)) != len(tr) or len(set(te)) != len(te)
                or tr != sorted(tr) or te != sorted(te) or set(tr) & set(te) or tested & set(te)):
            _fail("extended-families-v2 inner group partition differs.")
        partition = set(tr + te)
        if population is None:
            population = partition
        if partition != population or (group_sizes is not None and partition != set(group_sizes)):
            _fail("extended-families-v2 inner group coverage differs.")
        if group_sizes is not None and (train != sum(group_sizes[g] for g in tr) or test != sum(group_sizes[g] for g in te)):
            _fail("extended-families-v2 inner counts differ from retained rows.")
        tested.update(te)
        test_rows += test
    if tested != population or test_rows != rows:
        _fail("extended-families-v2 inner tests must cover the declared population once.")
    records = value.get("inner_selection")
    if not isinstance(records, list) or len(records) != len(candidates):
        _fail("extended-families-v2 selection candidate coverage differs.")
    objective = config.get("selection_objective")
    if objective not in ("mae", "group_mae"):
        _fail("extended-families-v2 selection objective differs.")
    widths, ridge_summaries = {}, {}
    for record, candidate in zip(records, candidates):
        record = _object(record)
        keys = _SELECTION_RECORD_V2 | ({"inner_fit_details"} if candidate["family"] in optional else set())
        if (set(record) != keys or record.get("candidate") != candidate["name"]
                or record.get("family") != candidate["family"] or record.get("objective") != objective
                or not _same_declaration(record.get("params"), candidate["params"])):
            _fail("extended-families-v2 candidate fields, identity, order, parameters or objective differ.")
        scores, score = record["inner_fold_scores"], record["selection_score"]
        if (not isinstance(scores, list) or len(scores) != inner
                or any(type(s) not in (int, float) or not math.isfinite(s) or s < 0 for s in scores)
                or type(score) not in (int, float) or not math.isfinite(score) or score < 0):
            _fail("extended-families-v2 MAE scores must be nonnegative finite fold-complete numbers.")
        count_key, total = (("test_group_count", groups) if objective == "group_mae" else ("test_rows", rows))
        scale = max(scores)
        # Normalize before multiplication so a finite saved score need not
        # overflow merely because a fold contains many rows or groups.
        pooled = 0.0 if scale == 0 else scale * math.fsum(
            (s / scale) * (a[count_key] / total) for s, a in zip(scores, audits))
        if not math.isfinite(pooled) or not math.isclose(score, pooled, rel_tol=1e-10, abs_tol=1e-12):
            _fail("extended-families-v2 selection score differs from its pooled fold scores.")
        if candidate["family"] in optional:
            details = record["inner_fit_details"]
            if not isinstance(details, list) or len(details) != inner:
                _fail("extended-families-v2 per-fit detail coverage differs.")
            for fold, (item, audit) in enumerate(zip(details, audits)):
                item = _object(item)
                if set(item) != {"inner_fold", "train_rows", "test_rows", "details"}:
                    _fail("extended-families-v2 per-fit detail wrapper fields differ.")
                for key, expected in (("inner_fold", fold), ("train_rows", audit["train_rows"]), ("test_rows", audit["test_rows"])):
                    if type(item[key]) is not int or item[key] != expected:
                        _fail("extended-families-v2 per-fit counts differ.")
                sizes = None if group_sizes is None else [group_sizes[g] for g in audit["train_groups"]]
                _extended_details_v2(candidate["family"], item["details"], audit["train_rows"],
                                     audit["train_group_count"], seed, candidate["params"], sizes)
                width = item["details"]["transformed_features"]
                if widths.setdefault(fold, width) != width:
                    _fail("extended-families-v2 preprocessing widths differ for the same inner training slice.")
                if candidate["family"] == "group_balanced_ridge":
                    summary = {k: v for k, v in item["details"].items() if k != "alpha"}
                    if not _same_declaration(ridge_summaries.setdefault(fold, summary), summary):
                        _fail("group-balanced Ridge summaries differ across alphas for the same training slice.")
    winner = min(records, key=lambda r: r["selection_score"])
    winners = {f: min((r for r in records if r["family"] == f), key=lambda r: r["selection_score"])
               for f in config["baselines"]}
    if (value.get("selected_candidate") != winner["candidate"] or value.get("selected_family") != winner["family"]
            or value.get("family_candidates") != {f: r["candidate"] for f, r in winners.items()}):
        _fail("extended-families-v2 winners differ from declared scores and candidate tie order.")
    final = _object(value.get("family_fit_details"))
    if set(final) != set(optional):
        _fail("extended-families-v2 final-refit detail coverage differs.")
    sizes = None if group_sizes is None else list(group_sizes.values())
    final_widths = set()
    for family in optional:
        _extended_details_v2(family, final[family], rows, groups, seed, winners[family]["params"], sizes)
        final_widths.add(final[family]["transformed_features"])
    if len(final_widths) != 1:
        _fail("extended-families-v2 preprocessing widths differ for the same final training slice.")
    full = final["group_balanced_ridge"]
    for summary in ridge_summaries.values():
        # GroupKFold retains whole groups. A training subset cannot contain
        # a group smaller or larger than the full training population's extrema.
        if (summary["min_training_group_size"] < full["min_training_group_size"]
                or summary["max_training_group_size"] > full["max_training_group_size"]):
            _fail("group-balanced Ridge inner group-size extrema exceed the final training population bounds.")
    # With at least two disjoint held-out folds, each full-training group is
    # present in an inner training slice. The envelope must therefore be exact.
    if (min(s["min_training_group_size"] for s in ridge_summaries.values()) != full["min_training_group_size"]
            or max(s["max_training_group_size"] for s in ridge_summaries.values()) != full["max_training_group_size"]):
        _fail("group-balanced Ridge inner extrema do not cover the final training population extrema.")


def _extended_details_v2(family, value, rows, groups, seed, params, sizes):
    if family == "group_balanced_ridge":
        _balanced_ridge_details(value, rows, groups, params["alpha"], sizes)
    elif family == "nystroem_ridge":
        _kernel_details(value, rows, seed, params["alpha"])
    elif family == "group_median":
        _median_details(value, rows, groups)
    else:
        _fail("unknown optional family in the extended-families-v2 detail contract.")


def _domain_records(value, config, inner):
    expected = _domain_candidates(config)
    records = value.get("inner_selection")
    if not isinstance(records, list) or len(records) != len(expected):
        _fail("domain candidate coverage differs from the frozen protocol grid.")
    objective = config.get("selection_objective")
    if objective not in ("mae", "group_mae"):
        _fail("domain selection objective differs.")
    for record, candidate in zip(records, expected):
        record = _object(record)
        if (record.get("candidate") != candidate["name"] or record.get("family") != candidate["family"]
                or not _same_declaration(record.get("params"), candidate["params"])
                or record.get("objective") != objective):
            _fail("domain candidate identity, parameters, order or objective differ.")
        scores = record.get("inner_fold_scores")
        score = record.get("selection_score")
        if (not isinstance(scores, list) or len(scores) != inner
                or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in scores)
                or type(score) not in (int, float) or not math.isfinite(score) or score < 0):
            _fail("domain MAE scores must be nonnegative finite numbers covering every fold.")
    winner = min(records, key=lambda r: r["selection_score"])
    if (value.get("selected_candidate") != winner["candidate"]
            or value.get("selected_family") != winner["family"]):
        _fail("domain selected candidate/family differs from declared scores and tie order.")
    if "family_candidates" in value:
        winners = {family: min((r for r in records if r["family"] == family),
                              key=lambda r: r["selection_score"])["candidate"]
                   for family in config["baselines"]}
        if value["family_candidates"] != winners:
            _fail("domain family winners differ from declared scores and tie order.")


def _domain_config(config, expected_groups):
    if config.get("inner_strategy") != "leave_one_domain_out":
        _fail("domain schema requires explicit leave_one_domain_out strategy.")
    domain = config.get("domain")
    if not isinstance(config.get("features"), list) or not _text(domain) or domain in (config.get("target"), config.get("group")) or domain in config["features"]:
        _fail("domain must be separate metadata, not a predictor or target/group.")
    mapping = _object(config.get("group_domains"))
    if len(mapping) != expected_groups or not mapping:
        _fail("domain mapping group coverage differs.")
    def valid(v):
        return isinstance(v, str) and bool(v) and len(v) <= 128 and v == v.strip() and not any(ord(c)<32 or 127<=ord(c)<=159 for c in v)
    if any(not isinstance(g, str) or not valid(d) for g, d in mapping.items()):
        _fail("invalid domain mapping identity.")
    count = _count(config.get("n_domains"), expected_groups, positive=True)
    if count < 2 or count != len(set(mapping.values())):
        _fail("domain count differs from mapping.")
    fingerprint = config.get("domain_values_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        _fail("domain fingerprint must be a SHA256 declaration.")
    return mapping


def _domain_selection(value, rows, group_ids, mapping, inner, config):
    groups = set(group_ids)
    domains = set(mapping[g] for g in groups)
    if type(inner) is not int or inner < 2 or inner != len(domains):
        _fail("domain-dependent inner fold count differs.")
    audits = value.get("inner_folds")
    if not isinstance(audits, list) or len(audits) != inner:
        _fail("domain inner fold coverage differs.")
    seen, tests = set(), 0
    for fold, audit in enumerate(audits):
        audit = _object(audit)
        train, test = audit.get("train_groups"), audit.get("test_groups")
        if (not isinstance(train, list) or not isinstance(test, list) or
                any(not isinstance(g, str) for g in train + test) or
                len(train) != len(set(train)) or len(test) != len(set(test)) or
                not train or not test or set(train) & set(test) or set(train + test) != groups):
            _fail("inner domain group partition differs.")
        td, vd = sorted(set(mapping[g] for g in train)), sorted(set(mapping[g] for g in test))
        if len(vd) != 1 or set(td) & set(vd) or vd[0] in seen:
            _fail("inner domains overlap or validation domain repeats.")
        seen.add(vd[0])
        for key, expected in dict(fold=fold, train_domains=td, test_domains=vd,
                train_domain_count=len(td), test_domain_count=1, domain_overlap_count=0,
                train_group_count=len(train), test_group_count=len(test), overlap_count=0).items():
            actual = audit.get(key)
            if type(actual) is not type(expected) or actual != expected:
                _fail("inner domain audit fields differ.")
        tr = _count(audit.get("train_rows"), rows, positive=True)
        te = _count(audit.get("test_rows"), rows, positive=True)
        if tr + te != rows or tr < len(train) or te < len(test):
            _fail("inner domain row partition differs.")
        tests += te
    if seen != domains or tests != rows:
        _fail("domain validation must cover all training rows exactly once by declared counts.")
    _domain_records(value, config, inner)


def _metadata(manifest):
    version = manifest.get("schema_version")
    if (manifest.get("mode") != "fit_predict" or "schema_id" in manifest
            or version not in ("1.0", "1.1", "1.2", "1.3", "1.4", "1.5")):
        _fail("only supported regression fit_predict schemas 1.0/1.1/1.2/1.3/1.4/1.5 are accepted.")
    config = _object(manifest.get("config"))
    if version != "1.5":
        _reject_ridge_v2_markers(manifest)
    if version != "1.4":
        if _DOMAIN_FIELDS & set(config):
            _fail("domain selection requires prediction schema 1.4.")
        _reject_domain_audits(manifest)
    if version == "1.5":
        if manifest.get("protocol") != "group-fit-predict-extended-families-v2":
            _fail("schema 1.5 requires the extended-families-v2 prediction protocol.")
        optional, _ = _extension_families_v2(manifest, config)
    elif version == "1.4":
        if manifest.get("protocol") != "domain-inner-fit-predict-v1":
            _fail("domain prediction protocol differs.")
        optional = _extension_families(manifest, config, allow_empty=True)
        if manifest.get("selected_family") not in config["baselines"]:
            _fail("domain selected family differs.")
    elif version == "1.3":
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
    if version not in ("1.3", "1.4", "1.5") and isinstance(config.get("baselines"), list) and any(
            family in config["baselines"] for family in ("nystroem_ridge", "group_median", "group_balanced_ridge")):
        _fail("optional families require the extended prediction schema.")
    n_train = _count(manifest.get("n_training_rows"), MAX_ROWS, positive=True)
    _count(manifest.get("n_training_groups"), n_train, positive=True)
    n_pred = _count(manifest.get("n_predictions"), MAX_ROWS, positive=True)
    if version in ("1.3", "1.4", "1.5"):
        inner = _count(config.get("inner_splits"), manifest["n_training_groups"], positive=True)
        if inner < 2:
            _fail("extended kernel requires at least two inner folds.")
        if version == "1.5":
            _extended_selection_v2(manifest, n_train, manifest["n_training_groups"], inner,
                                   config.get("random_state"), config)
        elif optional:
            _optional_selection_metadata(manifest, n_train, manifest["n_training_groups"], inner, config.get("random_state"), optional)
        if version == "1.4":
            mapping = _domain_config(config, manifest["n_training_groups"])
            _domain_selection(manifest, n_train, list(mapping), mapping, inner, config)
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
    if version == "1.5":
        result = PredictionSchemaResult("1.0", version, n_pred, n_features, support, total,
            checked_scope="checksums_prediction_rows_features_support_counts_and_extended_family_declarations",
            unchecked_metadata="training_rows_actual_inner_predictions_and_score_computation_weight_execution_fit_authenticity_and_other_manifest_fields_not_validated")
    else:
        result = PredictionSchemaResult("1.0", version, n_pred, n_features, support, total)
    return result, manifest, payloads["predictions.csv"], hashes


def validate_prediction_bundle_schema(run_dir) -> PredictionSchemaResult:
    """Validate exact saved bytes and bounded metadata; never prove execution.

    Additional extension details are checked for internal count/type consistency,
    not against authentic training data. See the internal loader for byte limits.
    """
    return _load_prediction_bundle(run_dir)[0]
