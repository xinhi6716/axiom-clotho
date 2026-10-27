"""One declared FET acquisition per composite device, joined to external endpoints.

Measurement algorithms and ML estimators are delegated unchanged. All preparation
and saved-score validation is standard-library-only; no future outcomes enter
preparation, planning or prediction. Descriptive log-slope extraction is not an ML
model fit and is not a certification of subthreshold behavior.
"""
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import re
import tempfile

from .. import contracts as c
from . import FET_ADAPTER, FET_FEATURES, FET_FILES, FET_INVENTORIES, load_bundle, write_bundle, finish_bundle

PHYSICAL = ("lot", "wafer", "device")
ACQUISITION = (*PHYSICAL, "condition", "replicate", "branch")
METADATA = (*ACQUISITION, "split_group", "temperature_K", "drain_source_voltage_V", "body_source_voltage_V",
            "signed_current_criterion_A", "window_min_V", "window_max_V", "max_bracket_span_V",
            "branch_sample_count", "log_selected_sample_count", "criterion_method", "criterion_bracket_span_V")
FEATURE_COLUMNS = (*METADATA, *FET_FEATURES)
MAP_COLUMNS = ("row_id", "group", "partition", *ACQUISITION)
COHORT_COLUMNS = ("row_id", *PHYSICAL, "partition", "group")
PREDICTION_COLUMNS = ("row_id", *PHYSICAL, "group", "prediction")
MEASUREMENT_KEYS = {"voltage_unit", "current_unit", "criterion_current", "criterion_current_unit",
                    "current_polarity", "max_bracket_span", "window_min", "window_max"}
CONFIG_KEYS = {"schema_version", "adapter", "endpoint_name", "target_unit", "group_by", "features", "measurement",
               "outer_splits", "inner_splits", "baselines", "seed", "objective", "max_fits", "declarations"}
LIMITATIONS = [
    "The endpoint, units, physical identities, pre-outcome feature timing and measurement assumptions are caller declarations, not verified facts.",
    "Exactly one declared acquisition branch is accepted per lot/wafer/device. No repeat, condition or branch is averaged or selected by this adapter.",
    "Criterion VGS is defined by the signed current criterion; windowed log slope is descriptive, not certified threshold physics or subthreshold swing.",
    "Grouping checks exact composite physical-key prefixes only; aliases and unrecorded dependence may remain.",
    "Preparation performs zero ML fits but includes descriptive log-slope least squares. Nested comparison evaluates the fixed model-selection procedure.",
    "Prediction-cohort outcomes enter only separate saved scoring, which performs zero ML fits. Future predictions are not uncertainty intervals.",
    "Checksums and cross-file consistency do not authenticate evidence, historical blinding, causal validity or physical independence.",
    "Bundles retain raw transfer measurements, physical identities, targets and predictions. Keep real data private unless sharing is separately authorized.",
]


def _key(row, fields=PHYSICAL):
    values = tuple(c.identifier(row[name]) for name in fields)
    if any(len(value) > 128 for value in values):
        raise c.WorkflowError("FET identities exceed the documented bounded length")
    return values


def _keyed(rows, fields=PHYSICAL):
    result = {}
    for row in rows:
        key = _key(row, fields)
        if key in result:
            raise c.WorkflowError("duplicate composite physical identity; no rows were joined or dropped")
        result[key] = row
    return result


def _decimal(value):
    if not isinstance(value, str) or len(value) > 64 or not c.NUMBER.fullmatch(value):
        raise c.WorkflowError("measurement parameters require explicit finite decimal strings without units")
    try:
        result = Decimal(value)
        if len(result.as_tuple().digits) > 32 or abs(result.as_tuple().exponent) > 200:
            raise c.WorkflowError("measurement parameter precision exceeds the supported range")
        return result
    except InvalidOperation:
        raise c.WorkflowError("measurement parameter is invalid") from None


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != CONFIG_KEYS
            or config["schema_version"] != "measurement-regression-v2" or config["adapter"] != FET_ADAPTER):
        raise c.WorkflowError("configuration must match the explicit FET measurement-regression-v2 schema")
    c.identifier(config["endpoint_name"])
    c.identifier(config["target_unit"])
    if config["group_by"] not in ("lot", "wafer", "device"):
        raise c.WorkflowError("choose an explicit composite lot, wafer or device grouping boundary")
    features = config["features"]
    if (not isinstance(features, list) or not features
            or any(not isinstance(f, str) or f not in FET_FEATURES for f in features)
            or len(features) != len(set(features))):
        raise c.WorkflowError("choose a nonempty unique subset of the two FET measurement features")
    models = config["baselines"]
    if (not isinstance(models, list) or "dummy" not in models or len(models) < 2
            or any(not isinstance(m, str) or m not in ("dummy", "ridge", "extra_trees") for m in models)
            or len(models) != len(set(models))):
        raise c.WorkflowError("comparison requires dummy and at least one supported unique model family")
    for name, low, high in (("outer_splits", 2, 100), ("inner_splits", 2, 100), ("seed", 0, 2 ** 32 - 101), ("max_fits", 1, 100000)):
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise c.WorkflowError("split count, seed or fit budget is outside the supported integer range")
    declarations = config["declarations"]
    if (config["objective"] != "group_mae" or not isinstance(declarations, dict)
            or set(declarations) != c.DECLARATIONS or any(v is not True for v in declarations.values())):
        raise c.WorkflowError("FET regression requires group_mae and all explicit scientific declarations")
    params = config["measurement"]
    if not isinstance(params, dict) or set(params) != MEASUREMENT_KEYS:
        raise c.WorkflowError("all eight FET measurement parameters must be explicitly declared")
    if (params["voltage_unit"] not in ("V", "mV", "uV")
            or params["current_unit"] not in ("A", "mA", "uA", "nA", "pA")
            or params["criterion_current_unit"] not in ("A", "mA", "uA", "nA", "pA")
            or params["current_polarity"] not in ("positive", "negative")):
        raise c.WorkflowError("unsupported declared FET voltage/current unit or polarity")
    criterion, span, lower, upper = (_decimal(params[n]) for n in ("criterion_current", "max_bracket_span", "window_min", "window_max"))
    if criterion * (1 if params["current_polarity"] == "positive" else -1) <= 0 or span <= 0 or lower >= upper:
        raise c.WorkflowError("FET criterion sign, positive bracket span or ordered voltage window is invalid")
    return config


def _feature_rows(features, config):
    """Validate retained report identity and byte lineage, without rerunning extraction."""
    if set(features) != set(FET_FILES) or any(not isinstance(v, bytes) for v in features.values()):
        raise c.WorkflowError("FET extraction must produce the complete fixed eight-file artifact set")
    try:
        schema = c.parse_json(features["feature_schema.json"])
        status = c.parse_json(features["workflow_status.json"])
        provenance = c.parse_json(features["provenance.json"])
        if (set(schema) != {"schema_version", "row_key", "model_feature_allowlist", "metadata_columns", "target_columns", "feature_units", "warnings"}
                or type(schema["schema_version"]) is not int or schema["schema_version"] != 1
                or schema["row_key"] != list(ACQUISITION) or schema["model_feature_allowlist"] != list(FET_FEATURES)
                or schema["metadata_columns"] != list(METADATA) or schema["target_columns"] != []
                or schema["feature_units"] != {FET_FEATURES[0]: "V", FET_FEATURES[1]: "decades/V"}
                or not isinstance(schema["warnings"], list) or any(not isinstance(w, str) for w in schema["warnings"])):
            raise c.WorkflowError("unsupported FET feature schema, row unit or allowlist")
        if (status["status"] != "complete" or status["feature_exported"] is not True
                or type(status["schema_version"]) is not int or status["schema_version"] != 1):
            raise c.WorkflowError("FET extraction is audit-only; no partial cohort may enter ML")
        rows = c.parse_csv(features["features.csv"], FEATURE_COLUMNS)
        indexed = _keyed(rows)  # Enforces ONE acquisition per composite physical device.
        for row in rows:
            _key(row, ACQUISITION)
            if row["split_group"] != row["lot"]:
                raise c.WorkflowError("FET upstream split metadata is inconsistent")
            for name in FET_FEATURES:
                if not c.NUMBER.fullmatch(row[name]) or len(row[name]) > 64 or not math.isfinite(float(row[name])):
                    raise c.WorkflowError("FET feature export must retain finite numeric values")
            for name in config["features"]:
                c.number(row[name])  # Explicit model-domain narrowing; never clip/rescale.
        predictors = c.parse_csv(features["predictors.csv"], FET_FEATURES)
        if predictors != [{f: r[f] for f in FET_FEATURES} for r in rows]:
            raise c.WorkflowError("FET predictor row order or values differ from the feature export")
        count = len(rows)
        if (status["branch_count"] != count or status["feature_row_count"] != count
                or len(status["branch_statuses"]) != count
                or { _key(r, ACQUISITION) for r in status["branch_statuses"] } != {_key(r, ACQUISITION) for r in rows}
                or any(r["vth_status"] != "ok" or r["log_slope_status"] != "ok" for r in status["branch_statuses"])):
            raise c.WorkflowError("FET complete-cohort status does not match every declared acquisition")
        raw_sha = c.sha(features["transfer_input.csv"])
        if (provenance["input_sha256"] != raw_sha or provenance["retained_input"] != "transfer_input.csv"
                or provenance["physical_device_key"] != list(PHYSICAL)
                or provenance["counts"]["feature_rows"] != count or provenance["counts"]["branch_count"] != count
                or provenance["counts"]["distinct_physical_device_ids"] != count
                or set(provenance["output_sha256"]) != set(FET_FILES) - {"provenance.json"}
                or any(provenance["output_sha256"][n] != c.sha(features[n]) for n in set(FET_FILES) - {"provenance.json"})):
            raise c.WorkflowError("FET feature provenance does not match the retained complete artifacts")
        reports = [c.parse_json(features[n]) for n in ("vth_report.json", "log_slope_report.json")]
        params = config["measurement"]
        vth, logarithmic = reports
        if (vth["tool"] != "vth-constant-current" or logarithmic["tool"] != "transfer-log-slope"
                or vth["input_units"] != {"gate_drain_body_voltage": params["voltage_unit"], "drain_current": params["current_unit"], "temperature": "K", "criterion_current": params["criterion_current_unit"]}
                or logarithmic["input_units"] != {"gate_drain_body_voltage": params["voltage_unit"], "drain_current": params["current_unit"], "temperature": "K"}
                or vth["parameters"]["criterion_current_input"] != params["criterion_current"]
                or vth["parameters"]["max_bracket_span_input"] != params["max_bracket_span"]
                or vth["parameters"]["branch_filter"] is not None
                or logarithmic["parameters"]["window_min_input"] != params["window_min"]
                or logarithmic["parameters"]["window_max_input"] != params["window_max"]):
            raise c.WorkflowError("saved FET reports do not match the explicit measurement configuration")
        for report in reports:
            if (type(report["schema_version"]) is not int or report["schema_version"] != 1
                    or report["provenance"]["input_sha256"] != raw_sha
                    or report["parameters"]["current_polarity"] != params["current_polarity"]
                    or report["counts"]["rejected_groups"] != 0 or report["counts"]["extracted_groups"] != count
                    or len(report["results"]) != count
                    or any(r["status"] != "ok" for r in report["results"])):
                raise c.WorkflowError("FET reports do not describe the same complete retained input")
        threshold = _keyed(vth["results"], ACQUISITION)
        slopes = _keyed(logarithmic["results"], ACQUISITION)
        expected = {_key(r, ACQUISITION) for r in rows}
        if set(threshold) != expected or set(slopes) != expected:
            raise c.WorkflowError("FET report acquisition identities differ from feature rows")
        for row in rows:
            key = _key(row, ACQUISITION)
            if (float(row[FET_FEATURES[0]]) != threshold[key]["extraction"]["gate_source_voltage_at_criterion_V"]
                    or float(row[FET_FEATURES[1]]) != slopes[key]["fit"]["slope_decades_per_V"]):
                raise c.WorkflowError("FET feature values differ from retained extractor reports")
        return indexed
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, c.WorkflowError):
            raise
        raise c.WorkflowError("saved FET extraction artifacts are malformed") from None


def _prefix(key, group_by):
    return key[:{"lot": 1, "wafer": 2, "device": 3}[group_by]]


def _identity_map(feature_rows, partitions, config):
    if set(partitions) != set(feature_rows):
        raise c.WorkflowError("partition identities must match every extracted physical device exactly")
    if any(row["partition"] not in ("train", "predict") for row in partitions.values()):
        raise c.WorkflowError("partition must be explicitly train or predict")
    keys = {p: sorted(k for k, r in partitions.items() if r["partition"] == p) for p in ("train", "predict")}
    groups = {p: sorted({_prefix(k, config["group_by"]) for k in keys[p]}) for p in keys}
    # Validate ORIGINAL PHYSICAL prefixes before creating separate token namespaces.
    if not keys["train"] or not keys["predict"] or set(groups["train"]) & set(groups["predict"]):
        raise c.WorkflowError("training and prediction cohorts require disjoint original composite physical groups")
    if len(groups["train"]) < max(config["outer_splits"], config["inner_splits"] + 1):
        raise c.WorkflowError("too few training groups for the requested nested comparison")
    result = []
    for partition, namespace in (("train", "train"), ("predict", "predict")):
        group_tokens = {key: f"{namespace}_group_{i:05d}" for i, key in enumerate(groups[partition], 1)}
        for i, key in enumerate(keys[partition], 1):
            result.append({"row_id": f"{namespace}_row_{i:05d}", "group": group_tokens[_prefix(key, config["group_by"])],
                           "partition": partition, **{n: feature_rows[key][n] for n in ACQUISITION}})
    return result


def _validate_map(rows, config):
    by_physical = _keyed(rows)
    if len({r["row_id"] for r in rows}) != len(rows):
        raise c.WorkflowError("duplicate normalized row identity")
    for row in rows:
        _key(row, ACQUISITION)
        c.identifier(row["row_id"])
        c.identifier(row["group"])
    expected = _identity_map(by_physical, by_physical, config)
    if rows != expected:
        raise c.WorkflowError("identity map is remapped, orphaned or not in its canonical per-partition order")
    return rows


def prepare(input_dir, partitions_csv, training_targets_csv, config_json, output_dir):
    c.protect_output(output_dir, input_dir)
    config_raw = c.read_regular(config_json)
    config = validate_config(c.parse_json(config_raw))
    raw = c.read_regular(Path(input_dir) / "transfer.csv")
    partition_raw, target_raw = map(c.read_regular, (partitions_csv, training_targets_csv))
    partitions = _keyed(c.parse_csv(partition_raw, (*PHYSICAL, "partition")))
    targets = _keyed(c.parse_csv(target_raw, (*PHYSICAL, "target")))
    from .. import regression as flow
    extractor = flow._load_component("_clotho_fet_extractor", "semiconductor/examples/fet-measurement-to-features/build_features.py")
    with tempfile.TemporaryDirectory(prefix="clotho-fet-") as directory:
        snapshot = Path(directory) / "transfer.csv"
        c.write_new(snapshot, raw)
        try:
            features = extractor.build(snapshot, **config["measurement"], assumptions_accepted=True)
        except extractor.InputError:
            raise c.WorkflowError("FET extraction rejected the declared measurements; inspect the input contract") from None
    # The upstream returns an audit-only dictionary rather than raising for branch failures.
    if c.parse_json(features["workflow_status.json"]).get("status") != "complete":
        raise c.WorkflowError("FET extraction is audit-only; no partial cohort may enter ML")
    originals = _feature_rows(features, config)
    mapping = _identity_map(originals, partitions, config)
    train_keys = {_key(r) for r in mapping if r["partition"] == "train"}
    if set(targets) != train_keys:
        raise c.WorkflowError("training endpoints must match only training physical devices exactly; prediction labels are forbidden")
    training, future = [], []
    for row in mapping:
        key = _key(row)
        model = {"row_id": row["row_id"], "group": row["group"], **{f: originals[key][f] for f in config["features"]}}
        if row["partition"] == "train":
            model["target"] = targets[key]["target"]
            c.number(model["target"])
            training.append(model)
        else:
            future.append(model)
    provenance = {"schema_version": "measurement-join-v2", "adapter": FET_ADAPTER,
                  "input_sha256": {"transfer.csv": c.sha(raw), "partitions.csv": c.sha(partition_raw),
                                   "training_targets.csv": c.sha(target_raw), "config.json": c.sha(config_raw)},
                  "training_rows": len(training), "prediction_rows": len(future),
                  "feature_row_unit": "one declared acquisition per composite physical device",
                  "physical_key": list(PHYSICAL), "acquisition_key": list(ACQUISITION),
                  "endpoint_join": "exact lot/wafer/device; training partition only; no derived or filled labels",
                  "caller_prediction_outcomes_opened": False, "ml_fits_performed": 0,
                  "limitations": LIMITATIONS}
    payloads = {"features/" + n: data for n, data in features.items()}
    payloads.update({"training.csv": c.csv_bytes(("row_id", "group", "target", *config["features"]), training),
                     "future.csv": c.csv_bytes(("row_id", "group", *config["features"]), future),
                     "cohort.csv": c.csv_bytes(COHORT_COLUMNS, [{n: r[n] for n in COHORT_COLUMNS} for r in mapping]),
                     "identity_map.csv": c.csv_bytes(MAP_COLUMNS, mapping), "config.json": c.json_bytes(config),
                     "provenance.json": c.json_bytes(provenance)})
    write_bundle(output_dir, "prepared", payloads)
    return provenance


def prepared(path):
    payloads, receipt_sha = load_bundle(path, "prepared")
    config = validate_config(c.parse_json(payloads["config.json"]))
    originals = _feature_rows({n: payloads["features/" + n] for n in FET_FILES}, config)
    mapping = _validate_map(c.parse_csv(payloads["identity_map.csv"], MAP_COLUMNS), config)
    if {_key(r) for r in mapping} != set(originals) or any(_key(r, ACQUISITION) != _key(originals[_key(r)], ACQUISITION) for r in mapping):
        raise c.WorkflowError("saved physical/acquisition identities differ from the FET feature export")
    cohort = c.parse_csv(payloads["cohort.csv"], COHORT_COLUMNS)
    if cohort != [{n: r[n] for n in COHORT_COLUMNS} for r in mapping]:
        raise c.WorkflowError("saved cohort differs from the exact identity map")
    training = c.parse_csv(payloads["training.csv"], ("row_id", "group", "target", *config["features"]))
    future = c.parse_csv(payloads["future.csv"], ("row_id", "group", *config["features"]))
    for model_rows, partition in ((training, "train"), (future, "predict")):
        identities = [r for r in mapping if r["partition"] == partition]
        if len(model_rows) != len(identities):
            raise c.WorkflowError("saved model row count differs from its declared partition")
        for model, identity in zip(model_rows, identities):
            if (model["row_id"] != identity["row_id"] or model["group"] != identity["group"]
                    or any(model[f] != originals[_key(identity)][f] for f in config["features"])):
                raise c.WorkflowError("saved model features or row mapping differ from the measurement export")
            for feature in config["features"]:
                c.number(model[feature])
            if partition == "train":
                c.number(model["target"])
    provenance = c.parse_json(payloads["provenance.json"])
    if (not isinstance(provenance, dict) or provenance.get("schema_version") != "measurement-join-v2"
            or provenance.get("adapter") != FET_ADAPTER or provenance.get("caller_prediction_outcomes_opened") is not False
            or provenance.get("physical_key") != list(PHYSICAL) or provenance.get("acquisition_key") != list(ACQUISITION)
            or provenance.get("training_rows") != len(training) or provenance.get("prediction_rows") != len(future)
            or not isinstance(provenance.get("input_sha256"), dict)
            or provenance["input_sha256"].get("transfer.csv") != c.sha(payloads["features/transfer_input.csv"])):
        raise c.WorkflowError("saved FET join provenance is inconsistent")
    return config, training, future, payloads, receipt_sha


def run(prepared_dir, output_dir):
    c.protect_output(output_dir, prepared_dir)
    from .. import regression as flow
    config, train, future, payloads, receipt_sha, frame, new, options, planned = flow._plans(prepared_dir)
    from groupbench import benchmark, fit_predict
    try:
        comparison = benchmark(frame, outer_splits=config["outer_splits"], **options)
        prediction = fit_predict(frame, new, prediction_source_sha256=c.sha(payloads["future.csv"]), **options)
    except (ValueError, FloatingPointError, OverflowError):
        raise c.WorkflowError("model comparison or prediction failed; no complete run was produced") from None
    if len(prediction.predictions) != len(future):
        raise c.WorkflowError("prediction row count changed")
    mapping = c.parse_csv(payloads["identity_map.csv"], MAP_COLUMNS)
    expected_future = [r for r in mapping if r["partition"] == "predict"]
    predictions = [{"row_id": identity["row_id"], **{n: identity[n] for n in PHYSICAL},
                    "group": identity["group"], "prediction": repr(float(value))}
                   for identity, value in zip(expected_future, prediction.predictions["prediction"])]
    for row in predictions:
        c.number(row["prediction"])
    metadata = {"schema_version": "measurement-regression-run-v2", "adapter": FET_ADAPTER, "config": config,
                "prepared_receipt_sha256": receipt_sha, "prepared_files_sha256": {n: c.sha(b) for n, b in sorted(payloads.items())},
                "training_rows": len(train), "future_cohort": [{n: r[n] for n in PREDICTION_COLUMNS if n != "prediction"} for r in predictions],
                "training_groups": sorted({r["group"] for r in train}), "plan": planned,
                "caller_prediction_outcomes_opened": False, "limitations": LIMITATIONS}
    output = c.begin_bundle(output_dir)
    comparison.write(output / "comparison")
    prediction.write(output / "prediction")
    c.write_new(output / "prediction_rows.csv", c.csv_bytes(PREDICTION_COLUMNS, predictions))
    c.write_new(output / "identity_map.csv", payloads["identity_map.csv"])
    c.write_new(output / "run.json", c.json_bytes(metadata))
    c.write_new(output / "report.html", flow._report("FET grouped comparison and target-free predictions", config,
        [f"Compared {len(train)} training physical devices; predicted {len(future)} separate physical devices.",
         "Only the two allowlisted measurement descriptors were eligible predictors. Prediction-cohort outcomes were not accepted or opened."],
        [("comparison/report.html", "Nested grouped comparison"), ("prediction_rows.csv", "Composite-device predictions"),
         ("identity_map.csv", "Exact physical/acquisition identity map"), ("run.json", "Configuration and plan"), ("COMPLETE.json", "Byte checksums")],
        limitations=LIMITATIONS))
    finish_bundle(output, "run")
    return metadata


def score(run_dir, outcomes_csv, output_dir):
    c.protect_output(output_dir, run_dir)
    payloads, receipt_sha = load_bundle(run_dir, "run")
    try:
        metadata = c.parse_json(payloads["run.json"])
        required = {"schema_version", "adapter", "config", "prepared_receipt_sha256", "prepared_files_sha256", "training_rows",
                    "future_cohort", "training_groups", "plan", "caller_prediction_outcomes_opened", "limitations"}
        if (not isinstance(metadata, dict) or set(metadata) != required
                or metadata["schema_version"] != "measurement-regression-run-v2" or metadata["adapter"] != FET_ADAPTER
                or metadata["caller_prediction_outcomes_opened"] is not False):
            raise c.WorkflowError("saved run metadata must match the exact FET v2 schema")
        config = validate_config(metadata["config"])
        hashes = metadata["prepared_files_sha256"]
        if (not isinstance(hashes, dict) or set(hashes) != FET_INVENTORIES["prepared"]
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
                       for h in [metadata["prepared_receipt_sha256"], *hashes.values()])
                or hashes["identity_map.csv"] != c.sha(payloads["identity_map.csv"])):
            raise c.WorkflowError("saved preparation provenance or retained identity map is inconsistent")
        mapping = _validate_map(c.parse_csv(payloads["identity_map.csv"], MAP_COLUMNS), config)
        train_map = [r for r in mapping if r["partition"] == "train"]
        future_map = [r for r in mapping if r["partition"] == "predict"]
        if (type(metadata["training_rows"]) is not int or metadata["training_rows"] != len(train_map)
                or metadata["training_groups"] != sorted({r["group"] for r in train_map})):
            raise c.WorkflowError("saved training identities or group count are inconsistent")
        rows = c.parse_csv(payloads["prediction_rows.csv"], PREDICTION_COLUMNS)
        identities = [{n: r[n] for n in PREDICTION_COLUMNS if n != "prediction"} for r in future_map]
        if ([{n: r[n] for n in PREDICTION_COLUMNS if n != "prediction"} for r in rows] != identities
                or metadata["future_cohort"] != identities):
            raise c.WorkflowError("saved prediction identities differ from the physical identity map")
        engine = c.parse_csv(payloads["prediction/predictions.csv"], ("row_position", "prediction"))
        if len(engine) != len(rows) or any(r["row_position"] != str(i) or c.number(r["prediction"]) != c.number(rows[i]["prediction"]) for i, r in enumerate(engine)):
            raise c.WorkflowError("keyed predictions differ from the unchanged calculator prediction bundle")
    except (KeyError, TypeError):
        raise c.WorkflowError("saved FET run metadata is malformed") from None
    outcome_raw = c.read_regular(outcomes_csv)
    outcomes = _keyed(c.parse_csv(outcome_raw, (*PHYSICAL, "target")))
    if set(outcomes) != {_key(r) for r in future_map}:
        raise c.WorkflowError("outcomes must match the exact predicted composite-device cohort")
    joined = [{"id": r["row_id"], "group": r["group"], "y_true": c.number(outcomes[_key(r)]["target"]),
               "y_pred": c.number(r["prediction"])} for r in rows]
    from .. import regression as flow
    calculator = flow._load_component("_workflow_saved_scorecard", "scorecard/tools/prediction-scorecard/prediction_scorecard.py")
    try:
        scorecard = calculator.score(joined, grouped=True)
    except calculator.InputError:
        raise c.WorkflowError("saved-score arithmetic exceeded the supported numeric range") from None
    result = {"schema_version": "measurement-saved-score-v2", "adapter": FET_ADAPTER, "config": config,
              "run_receipt_sha256": receipt_sha, "outcomes_sha256": c.sha(outcome_raw), "scorecard": scorecard,
              "fits_performed": 0, "cohort_rows": len(joined), "limitations": LIMITATIONS}
    scored = [{**{n: r[n] for n in ("row_id", *PHYSICAL, "group")}, "y_true": j["y_true"], "y_pred": j["y_pred"]} for r, j in zip(rows, joined)]
    payloads = {"scores.json": c.json_bytes(result), "scored_rows.csv": c.csv_bytes(("row_id", *PHYSICAL, "group", "y_true", "y_pred"), scored),
                "report.html": flow._report("FET saved prediction evaluation", config,
                    [f"Scored all {len(joined)} exact predicted composite physical devices; zero ML fits.",
                     "Scoring checks retained map/prediction consistency, not omitted raw measurements or feature extraction."],
                    [("scores.json", "Descriptive saved metrics"), ("scored_rows.csv", "Exact joined device outcomes"), ("COMPLETE.json", "Byte checksums")], limitations=LIMITATIONS)}
    write_bundle(output_dir, "score", payloads)
    return result
