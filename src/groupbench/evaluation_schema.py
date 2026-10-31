"""Opt-in saved regression metric consistency; never proof of execution."""
from dataclasses import dataclass
import csv
import io
import math
import os

from .metrics import grouped_metrics
from .prediction_schema import _manifest, _DECIMAL
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


def _check(manifest, summary, raw):
    version = manifest.get("schema_version")
    protocols = {"1.0": "nested-group-kfold-v1", "1.1": "nested-group-kfold-distinct-v2"}
    if not isinstance(version, str) or version not in protocols or manifest.get("protocol") != protocols[version] or "schema_id" in manifest or "schema_id" in summary:
        _fail("only supported regression evaluation schemas/protocols are accepted.")
    config = manifest.get("config")
    if not isinstance(config, dict):
        _fail("missing configuration.")
    if (version == "1.1" and config.get("categorical_encoding") != "distinct-v2") or (version == "1.0" and "categorical_encoding" in config):
        _fail("encoding and protocol differ.")
    if summary.get("schema_version") != version or (version == "1.1" and summary.get("protocol") != protocols[version]) or (version == "1.0" and "protocol" in summary):
        _fail("summary schema/protocol differs.")
    baselines = config.get("baselines")
    if not isinstance(baselines, list) or not baselines or any(not isinstance(x, str) or x not in ("dummy", "ridge", "extra_trees") for x in baselines) or len(set(baselines)) != len(baselines):
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
    membership = {}
    for group, fold in zip(groups, folds):
        if group in membership and membership[group] != fold:
            _fail("one group spans multiple held-out folds.")
        membership[group] = fold
    audits = manifest.get("outer_folds")
    if not isinstance(audits, list) or len(audits) != nf:
        _fail("outer fold audit coverage differs.")
    for fold, audit in enumerate(audits):
        test_groups = sorted(g for g, f in membership.items() if f == fold)
        train_groups = sorted(set(groups) - set(test_groups))
        test_rows = folds.count(fold)
        expected = dict(fold=fold, train_rows=n-test_rows, test_rows=test_rows,
                        train_group_count=len(train_groups), test_group_count=len(test_groups),
                        overlap_count=0, train_groups=train_groups, test_groups=test_groups)
        if not isinstance(audit, dict):
            _fail("invalid fold audit.")
        _compare({key: audit.get(key) for key in expected}, expected)
    for name in names:
        overall, per_group = grouped_metrics(observed, predictions[name], groups)
        per_fold = []
        for fold in range(nf):
            indexes = [i for i, f in enumerate(folds) if f == fold]
            scores, _ = grouped_metrics([observed[i] for i in indexes], [predictions[name][i] for i in indexes], [groups[i] for i in indexes])
            per_fold.append({"fold": fold, **scores})
        _compare(summary["models"][name], dict(overall=overall, per_group=per_group, per_fold=per_fold))
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
    return EvaluationSchemaResult("1.0", version, n, ng, models, total)
