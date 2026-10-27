"""Explicit measurement/endpoint joins, grouped ML, and separate saved scoring.

Contracts, preparation and scoring are stdlib-only. The existing groupbench ML
implementation is imported only by plan/run; the existing scorecard computes
saved metrics. No target inference, automatic grouping or hidden fitting.
"""
import html
import importlib.util
import json
from pathlib import Path
import re
import tempfile

from .components import RESOURCE_ROOT, verify_components
from .contracts import (FEATURES, FEATURE_FILES, INVENTORIES, WorkflowError, begin_bundle,
                        check_fresh_output, csv_bytes, finish_bundle, identifier,
                        json_bytes, keyed, load_bundle, number, parse_csv,
                        parse_json, protect_output, read_regular, sha, validate_config,
                        write_bundle, write_new)

RAW_FILES = ("manifest.csv", "conditions.csv", "iv_sweeps.csv")
FEATURE_COLUMNS = ("batch", "wafer", "device", "condition", "split_group", "read_voltage_V", "cycle_count", "ratio_below_one_cycle_count", *FEATURES)
COHORT_COLUMNS = ("device", "partition", "group", "batch", "wafer")
LIMITATIONS = [
    "Caller-declared endpoint meaning, units, feature timing and physical identities are not scientifically verified.",
    "Group disjointness checks exact supplied identities only; aliases and unrecorded dependence may remain.",
    "Measurement cycles are aggregated within physical device; v1 does not treat cycles or conditions as independent rows.",
    "Nested training-group comparison evaluates the selection procedure. Do not use its outer leaderboard as another tuning layer.",
    "Future predictions are not evaluation or uncertainty intervals; later exact-cohort scoring is descriptive only.",
    "Checksums detect byte changes, not authenticity, historical blinding, independence, causal validity or model usefulness.",
    "Local bundles include physical IDs, targets and predictions. Review before sharing.",
]


def _load_component(name, relative):
    verify_components()
    spec = importlib.util.spec_from_file_location(name, RESOURCE_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_groups(cohort, config):
    for row in cohort:
        for key in ("device", "group", "batch", "wafer"):
            identifier(row[key])
        if row["partition"] not in ("train", "predict") or row["group"] != row[config["group_by"]]:
            raise WorkflowError("partition or declared grouping boundary is inconsistent")
    wafers = {}
    for row in cohort:
        if wafers.setdefault(row["wafer"], row["batch"]) != row["batch"]:
            raise WorkflowError("one globally declared wafer identity maps to multiple batches")
    train = {r["group"] for r in cohort if r["partition"] == "train"}
    future = {r["group"] for r in cohort if r["partition"] == "predict"}
    if not train or not future or train & future:
        raise WorkflowError("training and prediction cohorts must both exist with disjoint declared groups")
    if len(train) < max(config["outer_splits"], config["inner_splits"] + 1):
        raise WorkflowError("too few training groups for the requested nested comparison; do not split repeated devices")


def prepare(input_dir, partitions_csv, training_targets_csv, config_json, output_dir):
    """Use exact device joins; no future outcome path is accepted or opened."""
    protect_output(output_dir, input_dir)
    raw = {name: read_regular(Path(input_dir) / name) for name in RAW_FILES}
    partition_raw, target_raw, config_raw = map(read_regular, (partitions_csv, training_targets_csv, config_json))
    config = validate_config(parse_json(config_raw))
    partitions = keyed(parse_csv(partition_raw, ("device", "partition")))
    targets = keyed(parse_csv(target_raw, ("device", "target")))
    extractor = _load_component("_workflow_measurement_extractor", "semiconductor/examples/measurement-to-features/build_features.py")
    # Snapshot once: both existing extraction and recorded hashes use these bytes.
    with tempfile.TemporaryDirectory(prefix="clotho-measurements-") as directory:
        for name, data in raw.items():
            write_new(Path(directory) / name, data)
        try:
            features = extractor.build(directory, max_bracket_span_V=config["max_bracket_span_V"], assumptions_accepted=True)
        except extractor.InputError:
            raise WorkflowError("measurement extraction rejected the declared acquisitions; inspect the original input contract") from None
    feature_rows = keyed(parse_csv(features["features.csv"].encode(), FEATURE_COLUMNS))
    schema = parse_json(features["feature_schema.json"].encode())
    if schema["row_key"] != ["device"] or schema["target_columns"] != [] or schema["model_feature_allowlist"] != list(FEATURES):
        raise WorkflowError("unsupported measurement feature row unit or allowlist")
    if set(partitions) != set(feature_rows):
        raise WorkflowError("partition IDs must match every extracted device exactly; no rows were dropped or guessed")
    cohort = []
    for device, row in sorted(feature_rows.items()):
        cohort.append({"device": device, "partition": partitions[device]["partition"],
                       "group": row[config["group_by"]], "batch": row["batch"], "wafer": row["wafer"]})
    _check_groups(cohort, config)
    train_ids = {r["device"] for r in cohort if r["partition"] == "train"}
    if set(targets) != train_ids:
        raise WorkflowError("training endpoints must match training devices exactly; prediction-device labels are forbidden here")
    training, future = [], []
    for row in cohort:
        record = {"device": row["device"], "group": row["group"],
                  **{name: feature_rows[row["device"]][name] for name in config["features"]}}
        for name in config["features"]:
            number(record[name])
        if row["partition"] == "train":
            record["target"] = targets[row["device"]]["target"]
            number(record["target"])
            training.append(record)
        else:
            future.append(record)
    provenance = {"schema_version": "measurement-join-v1", "input_sha256": {
        **{name: sha(data) for name, data in raw.items()}, "partitions.csv": sha(partition_raw),
        "training_targets.csv": sha(target_raw), "config.json": sha(config_raw)},
        "training_rows": len(training), "prediction_rows": len(future), "feature_row_unit": "device",
        "endpoint_join": "exact device IDs; training partition only; no labels derived or filled",
        "caller_prediction_outcomes_opened": False, "limitations": LIMITATIONS}
    payloads = {"features/" + name: text.encode() for name, text in features.items()}
    payloads.update({"training.csv": csv_bytes(("device", "group", "target", *config["features"]), training),
                     "future.csv": csv_bytes(("device", "group", *config["features"]), future),
                     "cohort.csv": csv_bytes(COHORT_COLUMNS, cohort), "config.json": json_bytes(config),
                     "provenance.json": json_bytes(provenance)})
    write_bundle(output_dir, "prepared", payloads)
    return provenance


def _prepared(path):
    payloads, receipt_sha = load_bundle(path, "prepared")
    config = validate_config(parse_json(payloads["config.json"]))
    cohort = parse_csv(payloads["cohort.csv"], COHORT_COLUMNS)
    by_id = keyed(cohort)
    _check_groups(cohort, config)
    training = parse_csv(payloads["training.csv"], ("device", "group", "target", *config["features"]))
    future = parse_csv(payloads["future.csv"], ("device", "group", *config["features"]))
    originals = keyed(parse_csv(payloads["features/features.csv"], FEATURE_COLUMNS))
    if set(originals) != set(by_id):
        raise WorkflowError("saved feature and cohort device sets differ")
    for rows, partition in ((training, "train"), (future, "predict")):
        if set(keyed(rows)) != {r["device"] for r in cohort if r["partition"] == partition}:
            raise WorkflowError("saved model input IDs do not match their declared partition")
        for row in rows:
            meta, original = by_id[row["device"]], originals[row["device"]]
            if (row["group"] != meta["group"] or meta["batch"] != original["batch"] or meta["wafer"] != original["wafer"]):
                raise WorkflowError("saved physical grouping metadata is inconsistent")
            for name in config["features"]:
                if row[name] != original[name]:
                    raise WorkflowError("saved model features differ from the measurement export")
                number(row[name])
            if partition == "train":
                number(row["target"])
    return config, training, future, payloads, receipt_sha


def _plans(prepared_dir):
    config, train, future, payloads, receipt_sha = _prepared(prepared_dir)
    # This is the sole transition to the existing ML stack.
    import pandas as pd
    from groupbench import plan_benchmark, plan_prediction
    frame = pd.DataFrame([{**r, "target": number(r["target"]), **{f: number(r[f]) for f in config["features"]}} for r in train])
    new = pd.DataFrame([{f: number(r[f]) for f in config["features"]} for r in future])
    options = dict(target="target", group="group", features=config["features"], inner_splits=config["inner_splits"],
                   random_state=config["seed"], baselines=config["baselines"], objective=config["objective"],
                   source_sha256=sha(payloads["training.csv"]))
    try:
        comparison = plan_benchmark(frame, outer_splits=config["outer_splits"], **options)
        prediction = plan_prediction(frame, new, prediction_source_sha256=sha(payloads["future.csv"]), **options)
    except ValueError:
        raise WorkflowError("the exact grouped plan is infeasible; inspect group counts, split choices and numeric units") from None
    total = comparison["fit_budget"]["total_fits"] + prediction["fit_budget"]["total_fits"]
    if total > config["max_fits"]:
        raise WorkflowError("planned pipeline fits exceed the explicitly declared max_fits; no fitting was performed")
    plan = {"schema_version": "measurement-execution-plan-v1", "fits_performed": 0,
            "total_planned_fits": total, "max_fits": config["max_fits"], "comparison": comparison,
            "prediction": prediction, "prepared_receipt_sha256": receipt_sha}
    return config, train, future, payloads, receipt_sha, frame, new, options, plan


def plan(prepared_dir):
    return _plans(prepared_dir)[-1]


def _report(title, config, paragraphs, links):
    esc = html.escape
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; style-src 'unsafe-inline'\">"
            "<title>Axiom Clotho — " + esc(title) + "</title><style>"
            "body{max-width:1000px;margin:0 auto;padding:2.5rem 1.5rem;background:#F5F1E7;color:#29271F;font:17px/1.7 system-ui,sans-serif}"
            "h1,h2{font-family:Georgia,'DejaVu Serif',serif;font-weight:400;line-height:1.18;overflow-wrap:anywhere}h1{font-size:clamp(2rem,5vw,3rem);letter-spacing:-.03em;margin:.8rem 0 1.2rem}h2{font-size:1.6rem}a{color:#856021;text-underline-offset:4px}code{overflow-wrap:anywhere}"
            "header{border-top:1px solid #C8BDA7;border-bottom:3px double #856021;padding:1.5rem 0 1.2rem;margin-bottom:2rem}header small{letter-spacing:.08em}header p{overflow-wrap:anywhere}section{background:#FBF8F0;border:1px solid #C8BDA7;border-radius:0;padding:1.5rem;margin:2rem 0}small{color:#625D4E}li{margin:.65rem 0}a:focus-visible{outline:3px solid #856021;outline-offset:4px}"
            "@media(max-width:600px){body{padding:1.25rem 1rem}section{padding:1.25rem 1rem}ul{padding-left:1.25rem}}"
            "@media(prefers-color-scheme:dark){body{background:#0B1519;color:#F1E8D5}a{color:#D4B77B}small{color:#B9B7AA}header{border-color:#D4B77B}section{background:#122025;border-color:#435052}a:focus-visible{outline-color:#D4B77B}}"
            "@media print{body,section{background:white;color:#29271F}a,small{color:#29271F}section{break-inside:avoid}}"
            "</style><header><small>Axiom Clotho · Moirai · local regression beta</small><h1>" + esc(title) + "</h1>"
            "<p>Endpoint: <strong>" + esc(config["endpoint_name"]) + "</strong> · units: " + esc(config["target_unit"]) + " · grouping: " + esc(config["group_by"]) + "</p></header><main>"
            + "".join("<p>" + esc(p) + "</p>" for p in paragraphs)
            + "<section><h2>Inspect the saved evidence</h2><ul>"
            + "".join("<li><a href='" + esc(url, quote=True) + "'>" + esc(label) + "</a></li>" for url, label in links)
            + "</ul></section><section><h2>What this does not establish</h2><ul>"
            + "".join("<li>" + esc(line) + "</li>" for line in LIMITATIONS)
            + "</ul></section></main></html>\n").encode()


def run(prepared_dir, output_dir):
    protect_output(output_dir, prepared_dir)
    config, train, future, payloads, receipt_sha, frame, new, options, planned = _plans(prepared_dir)
    from groupbench import benchmark, fit_predict
    try:
        comparison = benchmark(frame, outer_splits=config["outer_splits"], **options)
        prediction = fit_predict(frame, new, prediction_source_sha256=sha(payloads["future.csv"]), **options)
    except (ValueError, FloatingPointError, OverflowError):
        raise WorkflowError("model comparison or prediction failed; no complete run was produced; inspect numeric units and the plan") from None
    if len(prediction.predictions) != len(future):
        raise WorkflowError("prediction row count changed")
    predictions = [{"device": row["device"], "group": row["group"], "prediction": repr(float(value))}
                   for row, value in zip(future, prediction.predictions["prediction"])]
    if len(predictions) != len(future):
        raise WorkflowError("prediction row count changed")
    for row in predictions:
        number(row["prediction"])
    metadata = {"schema_version": "measurement-regression-run-v1", "config": config,
                "prepared_receipt_sha256": receipt_sha, "prepared_files_sha256": {n: sha(b) for n, b in sorted(payloads.items())},
                "training_rows": len(train), "future_cohort": [{"device": r["device"], "group": r["group"]} for r in future],
                "training_groups": sorted({r["group"] for r in train}), "plan": planned,
                "caller_prediction_outcomes_opened": False, "limitations": LIMITATIONS}
    output = begin_bundle(output_dir)
    comparison.write(output / "comparison")
    prediction.write(output / "prediction")
    write_new(output / "prediction_rows.csv", csv_bytes(("device", "group", "prediction"), predictions))
    write_new(output / "run.json", json_bytes(metadata))
    write_new(output / "report.html", _report("Comparison and target-free predictions", config,
        [f"Completed nested comparison on {len(train)} training devices and refit prediction for {len(future)} separately held-out devices.",
         "No caller-supplied prediction-cohort outcomes were accepted or opened. Supply later endpoints only to the separate score command.",
         "Read the training-group comparison before interpreting prediction values. They are not an evaluation of the future cohort."],
        [("comparison/report.html", "Nested grouped model comparison"), ("prediction_rows.csv", "Device-keyed target-free predictions"),
         ("prediction/manifest.json", "Refit selection and input-support record"), ("run.json", "Configuration and exact no-fit plan"), ("COMPLETE.json", "Completion and byte checksums")]))
    finish_bundle(output, "run")
    return metadata


def score(run_dir, outcomes_csv, output_dir):
    """Use already saved predictions; do not load ML, train, select, or refit."""
    protect_output(output_dir, run_dir)
    payloads, receipt_sha = load_bundle(run_dir, "run")
    metadata = parse_json(payloads["run.json"])
    try:
        required = {"schema_version", "config", "prepared_receipt_sha256", "prepared_files_sha256", "training_rows", "future_cohort", "training_groups", "plan", "caller_prediction_outcomes_opened", "limitations"}
        if not isinstance(metadata, dict) or set(metadata) != required:
            raise WorkflowError("saved run metadata does not match the fixed v1 schema")
        config = validate_config(metadata["config"])
        if metadata["schema_version"] != "measurement-regression-run-v1" or metadata["caller_prediction_outcomes_opened"] is not False:
            raise WorkflowError("saved run metadata is inconsistent")
        groups = metadata["training_groups"]
        if (not isinstance(groups, list) or not groups or any(not isinstance(g, str) for g in groups)
                or len(groups) != len(set(groups)) or type(metadata["training_rows"]) is not int
                or not len(groups) <= metadata["training_rows"] <= 10000):
            raise WorkflowError("saved training group metadata is malformed")
        for group in groups:
            identifier(group)
        hashes = metadata["prepared_files_sha256"]
        if (not isinstance(hashes, dict) or set(hashes) != INVENTORIES["prepared"]
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
                       for h in [metadata["prepared_receipt_sha256"], *hashes.values()])):
            raise WorkflowError("saved preparation provenance is malformed")
        rows = parse_csv(payloads["prediction_rows.csv"], ("device", "group", "prediction"))
        predictions = keyed(rows)
        expected = [{"device": r["device"], "group": identifier(r["group"])} for r in rows]
        if metadata["future_cohort"] != expected or {r["group"] for r in rows} & set(groups):
            raise WorkflowError("saved prediction identities or group boundary are inconsistent")
        engine = parse_csv(payloads["prediction/predictions.csv"], ("row_position", "prediction"))
        if len(engine) != len(rows) or any(r["row_position"] != str(i) or number(r["prediction"]) != number(rows[i]["prediction"]) for i, r in enumerate(engine)):
            raise WorkflowError("keyed predictions differ from the unchanged calculator prediction bundle")
    except (KeyError, TypeError):
        raise WorkflowError("saved run metadata is malformed") from None
    outcome_raw = read_regular(outcomes_csv)
    outcomes = keyed(parse_csv(outcome_raw, ("device", "target")))
    if set(outcomes) != set(predictions):
        raise WorkflowError("outcomes must match the exact predicted device cohort; partial, extra or duplicate joins are forbidden")
    joined = [{"id": r["device"], "group": r["group"], "y_true": number(outcomes[r["device"]]["target"]),
               "y_pred": number(r["prediction"])} for r in rows]
    calculator = _load_component("_workflow_saved_scorecard", "scorecard/tools/prediction-scorecard/prediction_scorecard.py")
    try:
        scorecard = calculator.score(joined, grouped=True)
    except calculator.InputError:
        raise WorkflowError("saved-score arithmetic exceeded the supported numeric range; rescale explicitly") from None
    result = {"schema_version": "measurement-saved-score-v1", "config": config,
              "run_receipt_sha256": receipt_sha, "outcomes_sha256": sha(outcome_raw),
              "scorecard": scorecard, "fits_performed": 0, "cohort_rows": len(joined), "limitations": LIMITATIONS}
    payloads = {"scores.json": json_bytes(result), "scored_rows.csv": csv_bytes(("id", "group", "y_true", "y_pred"), joined),
                "report.html": _report("Saved prediction evaluation", config,
                    [f"Scored all {len(joined)} exact predicted devices. No model was fitted or selected.",
                     f"Row MAE: {scorecard['overall']['mae']:.8g}; row RMSE: {scorecard['overall']['rmse']:.8g}.",
                     f"Equal-group MAE: {scorecard['equal_group_summary']['mae_mean']:.8g}. These are descriptive errors on this cohort, not population guarantees."],
                    [("scores.json", "All metrics, provenance and undefined-R² status"), ("scored_rows.csv", "Every exact joined observation and prediction"), ("COMPLETE.json", "Completion and byte checksums")])}
    write_bundle(output_dir, "score", payloads)
    return result
