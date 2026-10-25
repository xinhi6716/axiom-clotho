#!/usr/bin/env python3
"""Fixed environment-heldout feature-set prediction and separate exact-cohort scoring.

Training-label shuffling is a descriptive stress diagnostic, not a causal
negative control, randomization test or proof that any feature is causal.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from groupbench.model import Candidate, make_pipeline

MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 10000
MAX_REPORT_BYTES = 8 * 1024 * 1024
META = ("device_id", "dependency_block", "environment")
ARMS = ("dummy", "a", "b", "shuffle_a", "shuffle_b")
SPEC_KEYS = {"schema_version", "heldout_environment", "features_a", "features_b", "ridge_alpha", "shuffle_seed", "shuffle_scope", "target_unit"}
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]{1,4})?\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
LIMITS = [
    "A declared environment is not a randomized intervention. These predictions and contrasts do not identify a causal effect or causal feature.",
    "Device uniqueness and disjoint declared dependency blocks do not verify independence, correct identities or absence of omitted dependencies.",
    "Within-training-environment label shuffling preserves label multisets but not all block dependence; this is not an exchangeability-preserving randomization test.",
    "One seeded shuffle may leave identities or label values unchanged. It is never redrawn to force degradation and gives no p-value or null distribution.",
    "Configuration and source hashes identify bytes, not authenticity or proof that choices were frozen before outcome exposure.",
    "Feature timing, units, source comparability, causal structure and the selected held-out environment remain caller declarations.",
    "Only supplied devices define the cohort. No missing device, hidden preprocessing leakage, confounder or post-outcome predictor is discovered.",
    "No feature selection, tuning, automatic winner, confidence interval, causal effect, model persistence or network operation is performed.",
]


class InputError(ValueError):
    """Categorical diagnostics exclude raw values, IDs and paths."""


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise InputError("invalid command-line arguments; use --help")


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def read_regular(path):
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise InputError("input must be a regular file")
            raw = stream.read(MAX_BYTES + 1)
    except (OSError, ValueError):
        raise InputError("cannot read bounded regular input") from None
    if len(raw) > MAX_BYTES:
        raise InputError("input byte limit exceeded")
    return raw


def strict_json(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputError("duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise InputError("nonfinite JSON number")
    if not isinstance(blob, bytes) or len(blob) > MAX_BYTES:
        raise InputError("JSON byte or type limit exceeded")
    try:
        return json.loads(blob.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, ValueError, RecursionError):
        raise InputError("invalid strict UTF-8 JSON") from None


def identity(value):
    if (not isinstance(value, str) or not value or len(value) > 128 or value != value.strip()
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 or ord(c) == 0xfeff for c in value)):
        raise InputError("identity strings must be bounded, nonempty, unpadded and control-free")
    return value


def spec_from_bytes(blob):
    spec = strict_json(blob)
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS or spec["schema_version"] != "environment-feature-probe-v1":
        raise InputError("unsupported exact comparison specification")
    identity(spec["heldout_environment"])
    identity(spec["target_unit"])
    for name in ("features_a", "features_b"):
        features = spec[name]
        if not isinstance(features, list) or not 1 <= len(features) <= 32:
            raise InputError("feature lists must have one through 32 explicit names")
        for feature in features:
            identity(feature)
            if feature in (*META, "target"):
                raise InputError("identifiers, environment, dependency block and target cannot be predictors")
        if len(set(features)) != len(features):
            raise InputError("feature names must be unique within each list")
    if set(spec["features_a"]) == set(spec["features_b"]):
        raise InputError("two distinct prespecified feature sets are required")
    if len(set(spec["features_a"] + spec["features_b"])) > 32:
        raise InputError("selected feature union exceeds 32 columns")
    alpha, seed = spec["ridge_alpha"], spec["shuffle_seed"]
    if type(alpha) not in (int, float) or not 0 < alpha <= 1e12 or not math.isfinite(alpha):
        raise InputError("ridge alpha must be explicitly positive finite and bounded")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise InputError("shuffle seed must be an explicit unsigned 32-bit integer")
    if spec["shuffle_scope"] != "within_training_environment":
        raise InputError("unsupported explicit shuffle scope")
    return spec


def finite_token(token, missing=False):
    if missing and token == "":
        return float("nan")
    if not isinstance(token, str) or len(token) > 64 or not NUMBER.fullmatch(token):
        raise InputError("numeric fields must be finite unpadded decimal tokens or explicitly allowed empty features")
    value = float(token)
    if not math.isfinite(value) or (value == 0 and any(c in "123456789" for c in token.split("e")[0].split("E")[0])):
        raise InputError("numeric field outside finite representable range")
    if value != 0 and not 1e-100 <= abs(value) <= 1e100:
        raise InputError("nonzero numeric magnitude must be between 1e-100 and 1e100; rescale source units")
    return value


def csv_rows(blob, expected):
    if not isinstance(blob, bytes) or len(blob) > MAX_BYTES:
        raise InputError("CSV byte or type limit exceeded")
    try:
        text = blob.decode("utf-8-sig")
        if "\x00" in text:
            raise InputError("NUL is not allowed in CSV")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader, None)
        if header is None or len(header) != len(expected) or set(header) != set(expected):
            raise InputError("CSV must contain exactly the declared columns")
        result = []
        for ordinal, values in enumerate(reader, 1):
            if len(values) != len(header) or ordinal > MAX_ROWS:
                raise InputError("CSV row width or row limit violation")
            result.append(dict(zip(header, values)))
        if not result:
            raise InputError("CSV must contain data rows")
        return result
    except (UnicodeError, csv.Error):
        raise InputError("invalid UTF-8 CSV") from None


def prepare(train_blob, future_blob, spec):
    features = list(dict.fromkeys(spec["features_a"] + spec["features_b"]))
    train = csv_rows(train_blob, (*META, "target", *features))
    future = csv_rows(future_blob, (*META, *features))
    for records in (train, future):
        ids = set()
        for row in records:
            for name in META:
                identity(row[name])
            if row["device_id"] in ids:
                raise InputError("one unique row per physical device is required in each input")
            ids.add(row["device_id"])
            for feature in features:
                row[feature] = finite_token(row[feature], missing=True)
    if set(r["device_id"] for r in train) & set(r["device_id"] for r in future):
        raise InputError("training and future device IDs overlap")
    if set(r["dependency_block"] for r in train) & set(r["dependency_block"] for r in future):
        raise InputError("declared dependency block crosses the held-out boundary")
    if any(r["environment"] == spec["heldout_environment"] for r in train):
        raise InputError("held-out environment appears in training")
    if any(r["environment"] != spec["heldout_environment"] for r in future):
        raise InputError("future input must contain exactly the declared held-out environment")
    for row in train:
        row["target"] = finite_token(row["target"])
    environments = sorted({r["environment"] for r in train})
    if len(environments) < 2:
        raise InputError("at least two training environments are required")
    for env in environments:
        target = [r["target"] for r in train if r["environment"] == env]
        if len(target) < 2 or len(set(target)) < 2:
            raise InputError("each training environment needs at least two devices and two distinct target values")
    if any(all(math.isnan(r[f]) for r in train) for f in features):
        raise InputError("selected feature has no observed training values")
    return train, future, features


def shuffle_labels(train, seed):
    rng = np.random.default_rng(seed)
    donor = np.arange(len(train))
    per_environment = []
    for env in sorted({row["environment"] for row in train}):
        indices = sorted((i for i, row in enumerate(train) if row["environment"] == env), key=lambda i: train[i]["device_id"])
        shuffled = rng.permutation(indices)
        for recipient, source in zip(indices, shuffled):
            donor[recipient] = source
        changed_ids = sum(int(i != j) for i, j in zip(indices, shuffled))
        changed_values = sum(train[i]["target"] != train[j]["target"] for i, j in zip(indices, shuffled))
        per_environment.append({"environment": env, "devices": len(indices), "changed_device_assignments": changed_ids,
                                "changed_target_values": changed_values, "identity_mapping": changed_ids == 0,
                                "all_target_values_unchanged": changed_values == 0})
    return np.array([train[i]["target"] for i in donor], dtype=float), {
        "scope": "within_training_environment", "seed": seed,
        "environment_results": per_environment,
        "mapping": [{"recipient_device_id": row["device_id"], "donor_device_id": train[int(donor[i])]["device_id"]}
                    for i, row in enumerate(train)],
        "same_mapping_used_for_both_feature_sets": True, "redrawn": False}


def state_fingerprint(model):
    """Hash numeric fitted state, never serialize an executable model."""
    pre = model.named_steps["preprocess"].named_transformers_["numeric"]
    estimator = model.named_steps["model"]
    state = {"imputation": pre.named_steps["impute"].statistics_.tolist(),
             "scale_mean": pre.named_steps["scale"].mean_.tolist(),
             "scale_scale": pre.named_steps["scale"].scale_.tolist()}
    for attribute in ("coef_", "intercept_", "constant_"):
        if hasattr(estimator, attribute):
            state[attribute] = np.asarray(getattr(estimator, attribute)).tolist()
    return digest(canonical(state))


def seal(body):
    return {"artifact_schema": "environment-feature-probe-artifact-v1", "body_sha256": digest(canonical(body)), "body": body}


def predict(train_blob, future_blob, spec_blob, *, contract_declared=False):
    if contract_declared is not True:
        raise InputError("literal acceptance of the held-out prediction contract is required")
    spec = spec_from_bytes(spec_blob)
    train, future, features = prepare(train_blob, future_blob, spec)
    X = pd.DataFrame([{f: r[f] for f in features} for r in train], dtype=float)
    new = pd.DataFrame([{f: r[f] for f in features} for r in future], dtype=float)
    y = np.array([r["target"] for r in train], dtype=float)
    shuffled, shuffle = shuffle_labels(train, spec["shuffle_seed"])
    statuses, outputs = {}, {}
    for arm in ARMS:
        cols = features if arm == "dummy" else spec["features_a" if arm.endswith("a") or arm == "a" else "features_b"]
        candidate = Candidate("fixed_mean", "dummy", {"strategy": "mean"}) if arm == "dummy" else Candidate("fixed_ridge", "ridge", {"alpha": spec["ridge_alpha"]})
        stage = "fit"
        try:
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter("always")
                model = make_pipeline(X[cols], candidate, spec["shuffle_seed"])
                if arm != "dummy":
                    model.named_steps["model"].set_params(solver="svd")
                model.fit(X[cols], shuffled if arm.startswith("shuffle_") else y)
                fingerprint = state_fingerprint(model)
                stage = "predict"
                result = np.asarray(model.predict(new[cols]), dtype=float)
                if result.shape != (len(future),) or not np.isfinite(result).all() or np.any(np.abs(result) > 1e150):
                    raise ValueError("nonfinite or malformed prediction")
            statuses[arm] = {"status": "ok", "fitted_state_sha256": fingerprint,
                             "warning_categories": sorted({type(w.message).__name__ for w in caught})}
            outputs[arm] = result.tolist()
        except (ValueError, OverflowError, FloatingPointError, RuntimeError, np.linalg.LinAlgError):
            statuses[arm] = {"status": stage + "_failed", "fitted_state_sha256": None, "warning_categories": []}
            outputs[arm] = [None] * len(future)
    failed = [arm for arm in ARMS if statuses[arm]["status"] != "ok"]
    body = {"kind": "prediction", "protocol": "fixed-environment-feature-probe-v1",
            "status": "INCOMPLETE_PREDICTIONS" if failed else "COMPLETE_PREDICTIONS", "exit_code": int(bool(failed)),
            "spec": spec, "inputs_sha256": {"training_csv": digest(train_blob), "future_csv": digest(future_blob), "spec_json": digest(spec_blob)},
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
            "fit_contract": {"planned_fits": 5, "attempted_fits": 5, "completed_arms": 5 - len(failed),
                             "ridge_solver": "svd", "learned_preprocessing": "training-only median imputation and standardization",
                             "selection_or_tuning": False, "target_free_future_input": True},
            "training_membership": [{k: row[k] for k in META} for row in train], "shuffle": shuffle,
            "coverage": {"training_devices": len(train), "future_devices": len(future),
                         "training_dependency_blocks": len({r["dependency_block"] for r in train}),
                         "future_dependency_blocks": len({r["dependency_block"] for r in future}), "cross_boundary_block_overlap": 0,
                         "failed_arms": failed, "cohort_complete_for_scoring": not failed},
            "arms": statuses,
            "predictions": [{**{k: row[k] for k in META}, "predictions": {arm: outputs[arm][i] for arm in ARMS}} for i, row in enumerate(future)],
            "limitations": LIMITS.copy()}
    result = seal(body)
    report_bytes(result)
    return result


def validate_prediction_metadata(body, spec, rows):
    """Check internal shape/cohort consistency, never authenticate provenance."""
    hashes = body["inputs_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != {"training_csv", "future_csv", "spec_json"} or any(not isinstance(v, str) or not HASH.fullmatch(v) for v in hashes.values()):
        raise InputError("source fingerprint schema mismatch")
    runtime = body["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {"python", "numpy", "pandas", "scikit_learn"}:
        raise InputError("runtime schema mismatch")
    for value in runtime.values():
        identity(value)
    fit = body["fit_contract"]
    expected_fit = {"planned_fits": 5, "attempted_fits": 5, "completed_arms": 5,
                    "ridge_solver": "svd",
                    "learned_preprocessing": "training-only median imputation and standardization",
                    "selection_or_tuning": False, "target_free_future_input": True}
    if not isinstance(fit, dict) or fit != expected_fit or any(type(fit[k]) is not int for k in ("planned_fits", "attempted_fits", "completed_arms")) or fit["selection_or_tuning"] is not False or fit["target_free_future_input"] is not True:
        raise InputError("complete fixed-fit contract mismatch")
    training = body["training_membership"]
    if not isinstance(training, list) or not 4 <= len(training) <= MAX_ROWS:
        raise InputError("training membership shape mismatch")
    by_id = {}
    for row in training:
        if not isinstance(row, dict) or set(row) != set(META):
            raise InputError("training membership row schema mismatch")
        for name in META:
            identity(row[name])
        if row["device_id"] in by_id or row["environment"] == spec["heldout_environment"]:
            raise InputError("training membership identity mismatch")
        by_id[row["device_id"]] = row
    future_ids = {r["device_id"] for r in rows}
    train_blocks = {r["dependency_block"] for r in training}
    future_blocks = {r["dependency_block"] for r in rows}
    environments = {r["environment"] for r in training}
    if len(environments) < 2 or any(sum(r["environment"] == env for r in training) < 2 for env in environments) or future_ids & set(by_id) or train_blocks & future_blocks:
        raise InputError("declared training/future boundary is inconsistent")
    expected_coverage = {"training_devices": len(training), "future_devices": len(rows),
                         "training_dependency_blocks": len(train_blocks), "future_dependency_blocks": len(future_blocks),
                         "cross_boundary_block_overlap": 0, "failed_arms": [], "cohort_complete_for_scoring": True}
    if body["coverage"] != expected_coverage or any(type(body["coverage"][k]) is not int for k in ("training_devices", "future_devices", "training_dependency_blocks", "future_dependency_blocks", "cross_boundary_block_overlap")) or body["coverage"]["cohort_complete_for_scoring"] is not True:
        raise InputError("complete prediction coverage mismatch")
    shuffle = body["shuffle"]
    keys = {"scope", "seed", "environment_results", "mapping", "same_mapping_used_for_both_feature_sets", "redrawn"}
    if not isinstance(shuffle, dict) or set(shuffle) != keys or shuffle["scope"] != "within_training_environment" or type(shuffle["seed"]) is not int or shuffle["seed"] != spec["shuffle_seed"] or shuffle["same_mapping_used_for_both_feature_sets"] is not True or shuffle["redrawn"] is not False:
        raise InputError("shuffle contract mismatch")
    mapping = shuffle["mapping"]
    if not isinstance(mapping, list) or len(mapping) != len(training):
        raise InputError("shuffle mapping coverage mismatch")
    recipients, donors, pairs = set(), set(), {}
    for row in mapping:
        if not isinstance(row, dict) or set(row) != {"recipient_device_id", "donor_device_id"}:
            raise InputError("shuffle mapping row mismatch")
        recipient, donor = row["recipient_device_id"], row["donor_device_id"]
        identity(recipient); identity(donor)
        if recipient not in by_id or donor not in by_id or recipient in recipients or donor in donors or by_id[recipient]["environment"] != by_id[donor]["environment"]:
            raise InputError("shuffle is not a complete within-environment bijection")
        recipients.add(recipient); donors.add(donor); pairs[recipient] = donor
    _, expected_shuffle = shuffle_labels([dict(row, target=0.0) for row in training], spec["shuffle_seed"])
    expected_pairs = {row["recipient_device_id"]: row["donor_device_id"] for row in expected_shuffle["mapping"]}
    if pairs != expected_pairs:
        raise InputError("shuffle mapping does not match the declared seed and training membership")
    summaries = shuffle["environment_results"]
    if not isinstance(summaries, list) or len(summaries) != len(environments):
        raise InputError("shuffle environment coverage mismatch")
    seen = set()
    for summary in summaries:
        keys = {"environment", "devices", "changed_device_assignments", "changed_target_values", "identity_mapping", "all_target_values_unchanged"}
        if not isinstance(summary, dict) or set(summary) != keys:
            raise InputError("shuffle environment summary mismatch")
        env = summary["environment"]
        identity(env)
        if env not in environments or env in seen:
            raise InputError("shuffle environment identity mismatch")
        seen.add(env)
        ids = [i for i in by_id if by_id[i]["environment"] == env]
        changed = sum(i != pairs[i] for i in ids)
        for key in ("devices", "changed_device_assignments", "changed_target_values"):
            if type(summary[key]) is not int:
                raise InputError("shuffle summary count must be an integer")
        if summary["devices"] != len(ids) or summary["changed_device_assignments"] != changed or not 0 <= summary["changed_target_values"] <= changed or summary["identity_mapping"] is not (changed == 0) or summary["all_target_values_unchanged"] is not (summary["changed_target_values"] == 0):
            raise InputError("shuffle summary counts are inconsistent")
    if body["limitations"] != LIMITS:
        raise InputError("prediction limitations contract mismatch")


def prediction_body(blob):
    artifact = strict_json(blob)
    if not isinstance(artifact, dict) or set(artifact) != {"artifact_schema", "body_sha256", "body"} or artifact["artifact_schema"] != "environment-feature-probe-artifact-v1":
        raise InputError("unsupported exact prediction artifact schema")
    body = artifact["body"]
    if not isinstance(body, dict) or not isinstance(artifact["body_sha256"], str) or not HASH.fullmatch(artifact["body_sha256"]):
        raise InputError("malformed prediction artifact")
    try:
        if digest(canonical(body)) != artifact["body_sha256"]:
            raise InputError("prediction artifact integrity mismatch")
    except (ValueError, OverflowError, RecursionError):
        raise InputError("prediction artifact integrity mismatch") from None
    expected = {"kind", "protocol", "status", "exit_code", "spec", "inputs_sha256", "runtime", "fit_contract", "training_membership", "shuffle", "coverage", "arms", "predictions", "limitations"}
    if set(body) != expected or body["kind"] != "prediction" or body["protocol"] != "fixed-environment-feature-probe-v1":
        raise InputError("unsupported exact prediction body schema")
    spec = spec_from_bytes(canonical(body["spec"]))
    if body["status"] != "COMPLETE_PREDICTIONS" or type(body["exit_code"]) is not int or body["exit_code"] != 0:
        raise InputError("all five complete arms are required before scoring")
    arms, rows = body["arms"], body["predictions"]
    if not isinstance(arms, dict) or set(arms) != set(ARMS) or not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise InputError("prediction arm or cohort schema mismatch")
    for arm in ARMS:
        status = arms[arm]
        if not isinstance(status, dict) or set(status) != {"status", "fitted_state_sha256", "warning_categories"} or status["status"] != "ok" or not isinstance(status["fitted_state_sha256"], str) or not HASH.fullmatch(status["fitted_state_sha256"]):
            raise InputError("all five finite fitted predictions are required")
        if not isinstance(status["warning_categories"], list) or any(not isinstance(v, str) or not v.isidentifier() or len(v) > 128 for v in status["warning_categories"]):
            raise InputError("warning category schema mismatch")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {*META, "predictions"}:
            raise InputError("prediction row schema mismatch")
        for name in META:
            identity(row[name])
        if row["environment"] != spec["heldout_environment"] or row["device_id"] in ids:
            raise InputError("prediction cohort identity mismatch")
        ids.add(row["device_id"])
        values = row["predictions"]
        if not isinstance(values, dict) or set(values) != set(ARMS):
            raise InputError("prediction row arm coverage mismatch")
        if any(type(v) not in (int, float) or abs(v) > 1e150 or not math.isfinite(v) for v in values.values()):
            raise InputError("all five bounded finite predictions are required")
    coverage = body["coverage"]
    if not isinstance(coverage, dict) or coverage.get("future_devices") != len(rows) or coverage.get("cohort_complete_for_scoring") is not True or coverage.get("failed_arms") != []:
        raise InputError("prediction coverage mismatch")
    validate_prediction_metadata(body, spec, rows)
    return body


def stable_rmse(errors):
    maximum = max(errors)
    if maximum == 0:
        return 0.0
    return maximum * math.sqrt(math.fsum((error / maximum) ** 2 for error in errors) / len(errors))


def score(prediction_blob, outcome_blob):
    body = prediction_body(prediction_blob)
    outcomes = csv_rows(outcome_blob, ("device_id", "target"))
    by_id = {}
    for row in outcomes:
        identity(row["device_id"])
        if row["device_id"] in by_id:
            raise InputError("duplicate outcome device ID")
        by_id[row["device_id"]] = finite_token(row["target"])
    if set(by_id) != {r["device_id"] for r in body["predictions"]}:
        raise InputError("outcomes must match the exact complete prediction device cohort")
    records, losses = [], {arm: [] for arm in ARMS}
    for row in body["predictions"]:
        target = by_id[row["device_id"]]
        absolute = {arm: abs(row["predictions"][arm] - target) for arm in ARMS}
        if not all(math.isfinite(v) for v in absolute.values()):
            raise InputError("prediction residual exceeds supported range")
        for arm, error in absolute.items():
            losses[arm].append(error)
        records.append({**{k: row[k] for k in META}, "target": target, "predictions": row["predictions"],
                        "absolute_errors": absolute, "absolute_error_b_minus_a": absolute["b"] - absolute["a"]})
    metrics = {arm: {"mae": math.fsum(losses[arm]) / len(records), "rmse": stable_rmse(losses[arm])} for arm in ARMS}
    if any(any(error > 0 for error in losses[arm]) and (metrics[arm]["mae"] == 0 or metrics[arm]["rmse"] == 0) for arm in ARMS):
        raise InputError("aggregate error below supported representable range")
    contrast_sum = math.fsum(r["absolute_error_b_minus_a"] for r in records)
    contrast = contrast_sum / len(records)
    if contrast_sum != 0 and contrast == 0:
        raise InputError("aggregate contrast below supported representable range")
    report = {"kind": "score", "protocol": "fixed-environment-feature-probe-score-v1", "status": "COMPLETE_DESCRIPTIVE_SCORES", "exit_code": 0,
              "prediction_artifact_sha256": digest(prediction_blob), "outcomes_csv_sha256": digest(outcome_blob),
              "target_unit": body["spec"]["target_unit"], "heldout_environment": body["spec"]["heldout_environment"],
              "coverage": {"predicted_devices": len(records), "scored_devices": len(records), "omitted_devices": 0, "failed_arms": []},
              "weighting": "equal supplied physical-device rows; not an independence claim", "metrics": metrics,
              "paired_mean_absolute_error_b_minus_a": contrast,
              "negative_contrast_means": "lower B absolute error on this exact supplied held-out cohort only",
              "rows": records, "limitations": LIMITS.copy()}
    result = seal(report)
    report_bytes(result)
    return result


def report_bytes(report):
    payload = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    if len(payload) > MAX_REPORT_BYTES:
        raise InputError("complete output exceeds report byte limit")
    return payload


def write_report(report, output):
    payload = report_bytes(report)
    try:
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
    except OSError:
        raise InputError("cannot create requested exclusive output") from None


def main(argv=None):
    parser = Parser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prediction = sub.add_parser("predict")
    prediction.add_argument("training_csv")
    prediction.add_argument("future_csv")
    prediction.add_argument("--spec", required=True)
    prediction.add_argument("--accept-heldout-contract", action="store_true", required=True)
    prediction.add_argument("--output", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("predictions_json")
    scoring.add_argument("outcomes_csv")
    scoring.add_argument("--output", required=True)
    try:
        args = parser.parse_args(argv)
        if os.path.lexists(args.output):
            raise InputError("output destination already exists")
        if args.command == "predict":
            result = predict(read_regular(args.training_csv), read_regular(args.future_csv), read_regular(args.spec), contract_declared=args.accept_heldout_contract)
        else:
            result = score(read_regular(args.predictions_json), read_regular(args.outcomes_csv))
        write_report(result, args.output)
        return result["body"]["exit_code"]
    except InputError as error:
        print("Error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
