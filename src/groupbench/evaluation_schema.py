"""Opt-in saved regression metric consistency; never proof of execution."""
from dataclasses import dataclass
import csv
import io
import math
import os

import numpy as np

from .data import validate_groups
from .metrics import regression_metrics, _checked_arithmetic
from .prediction_schema import (_manifest, _DECIMAL, _optional_selection_metadata, _extension_families,
    _DOMAIN_FIELDS, _domain_config, _domain_selection, _reject_domain_audits, _domain_candidates,
    _same_declaration, _extension_families_v2, _extended_selection_v2, _reject_ridge_v2_markers,
    _EXTENDED_FAMILIES_V2, _GROUP_FIT_AUDIT_V2)
from .verification import (EVALUATION_FILES, MAX_CHECKSUM_BYTES, MAX_PAYLOAD_BYTES,
                           _checksum_records, _open_run_directory, _read_file, _signature)

REL_TOL = 1e-10
ABS_TOL = 1e-12
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CSV_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class EvaluationSchemaResult:
    contract_version: str
    schema_version: str
    n_rows: int
    n_groups: int
    n_models: int
    payload_bytes: int
    checked_scope: str = "saved_regression_rows_folds_models_and_metrics"
    unchecked_metadata: str = "inner_selection_features_source_authenticity_and_other_metadata"


def _fail(message):
    raise ValueError("Evaluation bundle consistency: " + message)


def _count(value, maximum, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        _fail("invalid bounded integer count.")
    return value


def _compare(actual, expected):
    """Exact identities/counts/nulls; tolerant finite float64 metric comparison."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            _fail("metric fields or identities differ.")
        for key in expected:
            _compare(actual[key], expected[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            _fail("metric population differs.")
        for a, e in zip(actual, expected):
            _compare(a, e)
    elif isinstance(expected, float):
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            _fail("saved metric differs from retained predictions.")
    elif type(actual) is not type(expected) or actual != expected:
        _fail("count, identity, or null metric differs.")


def _aggregate_saved(y, pred, per_group):
    """Same arithmetic/order as grouped_metrics; only partition discovery differs."""
    overall = regression_metrics(y, pred)
    with _checked_arithmetic():
        overall.update({
            "n_groups": len(per_group),
            "group_mae": float(np.mean([row["mae"] for row in per_group])),
            "group_rmse": float(np.mean([row["rmse"] for row in per_group])),
            "worst_group_mae": float(max(row["mae"] for row in per_group)),
        })
    return overall


def _partitioned_metrics(observed, predicted, group_rows, fold_rows, fold_groups):
    """Stable positional slices preserve reference float64 reduction order."""
    y, pred = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    per_group = [{"group": group, **regression_metrics(y[indexes], pred[indexes])}
                 for group, indexes in group_rows.items()]
    by_group = {row["group"]: row for row in per_group}
    overall = _aggregate_saved(y, pred, per_group)
    per_fold = [{"fold": fold, **_aggregate_saved(y[indexes], pred[indexes],
                 [by_group[group] for group in fold_groups[fold]])}
                for fold, indexes in enumerate(fold_rows)]
    return dict(overall=overall, per_group=per_group, per_fold=per_fold)


def _check(manifest, summary, raw):
    version = manifest.get("schema_version")
    protocols = {"1.0": "nested-group-kfold-v1", "1.1": "nested-group-kfold-distinct-v2",
                 "1.2": "nested-group-kfold-extended-families-v1", "1.3": "nested-group-outer-domain-inner-v1",
                 "1.4": "nested-group-kfold-extended-families-v2"}
    if not isinstance(version, str) or version not in protocols or manifest.get("protocol") != protocols[version] or "schema_id" in manifest or "schema_id" in summary:
        _fail("only supported regression evaluation schemas/protocols are accepted.")
    config = manifest.get("config")
    if not isinstance(config, dict):
        _fail("missing configuration.")
    if version != "1.4":
        _reject_ridge_v2_markers(manifest)
        _reject_ridge_v2_markers(summary)
    if (version == "1.1" and config.get("categorical_encoding") != "distinct-v2") or (version == "1.0" and "categorical_encoding" in config):
        _fail("encoding and protocol differ.")
    if summary.get("schema_version") != version or (version == "1.1" and summary.get("protocol") != protocols[version]) or (version == "1.0" and "protocol" in summary):
        _fail("summary schema/protocol differs.")
    if version != "1.3":
        if _DOMAIN_FIELDS & set(config):
            _fail("domain selection requires evaluation schema 1.3.")
        _reject_domain_audits(manifest)
    if version in ("1.2", "1.3", "1.4"):
        if version == "1.4":
            optional, _ = _extension_families_v2(manifest, config)
        else:
            optional = _extension_families(manifest, config, allow_empty=version == "1.3")
        if summary.get("capabilities") != manifest["capabilities"] or summary.get("protocol") != protocols[version]:
            _fail("extended capability or protocol differs in summary.")
    baselines = config.get("baselines")
    known = (_EXTENDED_FAMILIES_V2 if version == "1.4" else
             ("dummy", "ridge", "extra_trees", "nystroem_ridge", "group_median") if version in ("1.2", "1.3") else
             ("dummy", "ridge", "extra_trees"))
    if not isinstance(baselines, list) or not baselines or any(not isinstance(x, str) or x not in known for x in baselines) or len(set(baselines)) != len(baselines):
        _fail("unsupported or duplicate model family.")
    names = baselines + ["selected"]
    if not isinstance(summary.get("models"), dict) or set(summary["models"]) != set(names):
        _fail("model coverage differs.")
    if config.get("selection_objective") not in ("group_mae", "mae") or summary.get("selection_objective") != config["selection_objective"] or summary.get("primary_entry") != "selected":
        _fail("selection metadata differs.")
    n = _count(manifest.get("n_rows"), 20000, 1)
    ng = _count(manifest.get("n_groups"), n, 2)
    nf = _count(config.get("outer_splits"), ng, 2)
    try:
        text = raw.decode("utf-8")
        if "\0" in text:
            _fail("NUL is forbidden in CSV.")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = ["row_position", "group", "outer_fold", "observed"] + ["prediction_" + x for x in names]
        if next(reader, None) != header:
            _fail("prediction columns differ from configured models.")
        groups, folds, observed = [], [], []
        predictions = {name: [] for name in names}
        fold_tokens = {str(f) for f in range(nf)}
        for row in reader:
            if len(groups) >= n or len(row) != len(header) or row[0] != str(len(groups)) or not row[1] or row[2] not in fold_tokens:
                _fail("invalid row identity, group, fold, or width.")
            values = []
            for token in row[3:]:
                if len(token) > 64 or not _DECIMAL.fullmatch(token) or not math.isfinite(float(token)):
                    _fail("predictions and observations must be finite decimal numbers.")
                values.append(float(token))
            groups.append(row[1]); folds.append(int(row[2])); observed.append(values[0])
            for name, value in zip(names, values[1:]):
                predictions[name].append(value)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("Evaluation bundle consistency: invalid UTF-8 CSV.") from exc
    if len(groups) != n or len(set(groups)) != ng or set(folds) != set(range(nf)):
        _fail("row/group/fold coverage differs.")
    # Validate literal text identities before reusing group scores across folds.
    validate_groups(groups, n)
    membership, group_rows = {}, {}
    fold_rows = [[] for _ in range(nf)]
    fold_groups = [[] for _ in range(nf)]
    for position, (group, fold) in enumerate(zip(groups, folds)):
        if group in membership and membership[group] != fold:
            _fail("one group spans multiple held-out folds.")
        if group not in membership:
            group_rows[group] = []
            fold_groups[fold].append(group)
        membership[group] = fold
        group_rows[group].append(position)
        fold_rows[fold].append(position)
    audits = manifest.get("outer_folds")
    if not isinstance(audits, list) or len(audits) != nf:
        _fail("outer fold audit coverage differs.")
    sorted_groups = sorted(membership)
    if version == "1.3":
        candidates = _domain_candidates(config)
        declared = config.get("candidates")
        if (not isinstance(declared, list) or len(declared) != len(candidates)
                or any(not _same_declaration(a, b) for a, b in zip(declared, candidates))):
            _fail("domain candidate declaration differs from the frozen protocol grid.")
        mapping = _domain_config(config, ng)
        if set(mapping) != set(sorted_groups) or config.get("inner_splits") is not None:
            _fail("domain group coverage or inner_splits declaration differs.")
        inner_counts = config.get("inner_splits_per_outer")
        if not isinstance(inner_counts, list) or len(inner_counts) != nf:
            _fail("domain inner fold schedule coverage differs.")
    for fold, audit in enumerate(audits):
        test_groups = sorted(fold_groups[fold])
        # These explicit lists are part of the saved format, not row rescans.
        train_groups = [group for group in sorted_groups if membership[group] != fold]
        test_rows = len(fold_rows[fold])
        expected = dict(fold=fold, train_rows=n-test_rows, test_rows=test_rows,
                        train_group_count=len(train_groups), test_group_count=len(test_groups),
                        overlap_count=0, train_groups=train_groups, test_groups=test_groups)
        if not isinstance(audit, dict):
            _fail("invalid fold audit.")
        if version == "1.4":
            required = _GROUP_FIT_AUDIT_V2 | {"inner_seed", "selected_candidate", "selected_family",
                "family_candidates", "inner_selection", "inner_folds", "family_fit_details"}
            if set(audit) != required:
                _fail("extended-families-v2 outer fold audit fields differ.")
        _compare({key: audit.get(key) for key in expected}, expected)
        if version != "1.3":
            _reject_domain_audits(audit)
        if version in ("1.2", "1.3", "1.4"):
            inner = _count(inner_counts[fold] if version == "1.3" else config.get("inner_splits"), len(train_groups), 2)
            if version == "1.4":
                seed = _count(config.get("random_state"), 2**32 - 1)
                expected_seed = seed + fold + 1
                if type(audit.get("inner_seed")) is not int or audit["inner_seed"] != expected_seed:
                    _fail("extended-families-v2 inner seed differs from the scheduled fold.")
                _extended_selection_v2(audit, n-test_rows, len(train_groups), inner, expected_seed,
                                       config, group_sizes={g: len(group_rows[g]) for g in train_groups})
                # The selected column is exactly the selected family refit's
                # predictions, not another fit selected on the outer fold.
                if any(predictions["selected"][i] != predictions[audit["selected_family"]][i]
                       for i in fold_rows[fold]):
                    _fail("selected predictions differ from their declared family refit.")
            elif optional:
                _optional_selection_metadata(audit, n-test_rows, len(train_groups), inner, audit.get("inner_seed"), optional)
            if version == "1.3":
                if not isinstance(audit.get("family_candidates"), dict):
                    _fail("domain family winner declaration is missing.")
                _domain_selection(audit, n-test_rows, train_groups, mapping, inner, config)
    for name in names:
        _compare(summary["models"][name], _partitioned_metrics(
            observed, predictions[name], group_rows, fold_rows, fold_groups))
    return version, n, ng, len(names)


def validate_evaluation_bundle(run_dir) -> EvaluationSchemaResult:
    """Check saved regression metrics only; no models, sources, HTML rendering or writes."""
    with _open_run_directory(run_dir) as fd:
        _, _, raw, signature = _read_file(fd, "checksums.json", MAX_CHECKSUM_BYTES, collect=True)
        records = _checksum_records(raw)
        if set(records) != EVALUATION_FILES:
            _fail("requires an evaluation bundle.")
        signatures = {"checksums.json": signature}
        payloads, total = {}, 0
        for name, limit in (("manifest.json", MAX_JSON_BYTES), ("summary.json", MAX_JSON_BYTES), ("predictions.csv", MAX_CSV_BYTES), ("report.html", MAX_PAYLOAD_BYTES)):
            digest, size, raw, signature = _read_file(fd, name, limit, collect=name != "report.html")
            if digest != records[name]:
                _fail("payload SHA-256 mismatch.")
            signatures[name], payloads[name] = signature, raw
            total += size
        version, n, ng, models = _check(_manifest(payloads["manifest.json"]), _manifest(payloads["summary.json"]), payloads["predictions.csv"])
        for name, signature in signatures.items():
            if _signature(os.stat(name, dir_fd=fd, follow_symlinks=False)) != signature:
                _fail("bundle changed during validation; retry an idle directory.")
    if version == "1.4":
        return EvaluationSchemaResult("1.0", version, n, ng, models, total,
            checked_scope="saved_regression_rows_folds_models_metrics_and_extended_family_declarations",
            unchecked_metadata="features_actual_inner_predictions_and_score_computation_weight_execution_fit_source_authenticity_and_other_metadata")
    return EvaluationSchemaResult("1.0", version, n, ng, models, total)
