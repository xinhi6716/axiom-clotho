#!/usr/bin/env python3
"""Run an entirely invented semiconductor measurement-to-heldout-prediction example.

Trusted local source checkouts only. No network, downloads, real data, hidden
training, endpoint construction from resistance, or causal inference.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ENVS = ("invented_process_A", "invented_process_B", "invented_process_C")
FEATURES = ("hrs_resistance_median_ohm", "lrs_resistance_median_ohm", "resistance_ratio_median")
COHORT_COLUMNS = ("fabrication_id", "lot_id", "batch_id", "wafer_id", "physical_device_id", "environment", "partition", "dependency_block")
SEMICONDUCTOR_COMPONENTS = ("examples/measurement-to-features/generate_synthetic.py", "examples/measurement-to-features/build_features.py",
                            "tools/iv-readpoint/iv_readpoint.py", "tools/rram-window/rram_window.py",
                            "tools/physical-holdout-audit/physical_holdout_audit.py")
CALCULATOR_COMPONENTS = ("examples/environment-heldout-comparison/probe.py", "src/groupbench/model.py", "src/groupbench/categorical.py", "src/groupbench/data.py")


class WalkthroughError(ValueError):
    """Categorical failure with no raw input data or local paths."""


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise WalkthroughError("invalid walkthrough arguments; use --help")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write_new(path, blob):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(blob)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n").encode()


def csv_bytes(columns, rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def read_csv(path, columns):
    try:
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:
            raise WalkthroughError("invented fixture byte limit exceeded")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        if reader.fieldnames != list(columns):
            raise WalkthroughError("unexpected exact fixture columns")
        rows = list(reader)
        if not rows or any(set(r) != set(columns) or any(v is None for v in r.values()) for r in rows):
            raise WalkthroughError("invalid invented fixture row shape")
        return rows
    except (OSError, UnicodeError, csv.Error):
        raise WalkthroughError("cannot read the expected local fixture") from None


def component_hashes(semiconductor_source, calculator_source):
    result = {"walkthrough/run.py": sha(Path(__file__).read_bytes())}
    for label, root, names in (("semiconductor", semiconductor_source, SEMICONDUCTOR_COMPONENTS), ("calculator", calculator_source, CALCULATOR_COMPONENTS)):
        for name in names:
            path = root / name
            if not path.is_file():
                raise WalkthroughError("required source companion is missing; use the documented matching source checkouts")
            result[label + "/" + name] = sha(path.read_bytes())
    return result


def run_stage(name, script, arguments, log, calculator_source):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(calculator_source / "src"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        completed = subprocess.run([sys.executable, str(script), *map(str, arguments)],
                                   capture_output=True, timeout=120, env=env)
    except (OSError, subprocess.TimeoutExpired):
        log.append({"stage": name, "status": "execution_failed", "exit_code": None})
        raise WalkthroughError("a required local command could not complete") from None
    log.append({"stage": name, "status": "complete" if completed.returncode == 0 else "command_rejected", "exit_code": completed.returncode})
    if completed.returncode != 0:
        # Upstream scripts are trusted, but their raw stderr may contain paths.
        raise WalkthroughError("a required local command rejected the declared fixture or runtime")


def expand_fixture(base, output):
    manifest_columns = ("batch", "wafer", "device", "cycle", "replicate", "branch", "state", "condition")
    iv_columns = ("device", "cycle", "branch", "voltage", "current")
    condition_columns = ("condition", "read_voltage_V", "temperature_K", "read_delay_ms", "protocol")
    manifest = read_csv(base / "manifest.csv", manifest_columns)
    sweeps = read_csv(base / "iv_sweeps.csv", iv_columns)
    conditions = read_csv(base / "conditions.csv", condition_columns)
    templates = {}
    for row in manifest:
        templates.setdefault(row["device"], (row["batch"], row["wafer"]))
        if templates[row["device"]] != (row["batch"], row["wafer"]):
            raise WalkthroughError("base synthetic device changes fabrication identity")
    if len(templates) != 6 or len({pair[0] for pair in templates.values()}) != 2:
        raise WalkthroughError("unsupported synthetic generator contract")
    if len(manifest) != 36 or len(sweeps) != 108 or len(conditions) != 1:
        raise WalkthroughError("unsupported fixed synthetic fixture size")
    for batch in {pair[0] for pair in templates.values()}:
        members = [pair for pair in templates.values() if pair[0] == batch]
        if len(members) != 3 or len(set(members)) != 1:
            raise WalkthroughError("the fixed example requires three devices on one wafer per batch")
    for device in templates:
        device_rows = [r for r in manifest if r["device"] == device]
        cycles = {r["cycle"] for r in device_rows}
        if (len(device_rows) != 6 or len(cycles) != 3
                or any(sorted(r["state"] for r in device_rows if r["cycle"] == cycle) != ["HRS", "LRS"] for cycle in cycles)
                or sum(r["device"] == device for r in sweeps) != 18):
            raise WalkthroughError("the fixed example requires three paired HRS/LRS cycles per device")
    expanded, expanded_iv, cohort = [], [], []
    for env_index, environment in enumerate(ENVS):
        lot = "invented_fab/lot_" + "ABC"[env_index]
        mapping = {}
        batches = {batch: i + 1 for i, batch in enumerate(sorted({pair[0] for pair in templates.values()}))}
        for template, (batch_template, _) in sorted(templates.items()):
            batch = lot + "/batch_" + str(batches[batch_template])
            wafer = batch + "/wafer_1"
            number = sorted(t for t, p in templates.items() if p[0] == batch_template).index(template) + 1
            device = wafer + "/device_" + str(number)
            mapping[template] = {"batch": batch, "wafer": wafer, "device": device}
            cohort.append({"fabrication_id": "invented_fab", "lot_id": lot, "batch_id": batch, "wafer_id": wafer,
                           "physical_device_id": device, "environment": environment,
                           "partition": "test" if env_index == 2 else "train", "dependency_block": wafer})
        for row in manifest:
            expanded.append({**row, **mapping[row["device"]]})
        for row in sweeps:
            if row["device"] not in mapping:
                raise WalkthroughError("unmapped synthetic sweep device")
            expanded_iv.append({**row, "device": mapping[row["device"]]["device"]})
    output.mkdir(mode=0o700)
    write_new(output / "manifest.csv", csv_bytes(manifest_columns, expanded))
    write_new(output / "iv_sweeps.csv", csv_bytes(iv_columns, expanded_iv))
    write_new(output / "conditions.csv", csv_bytes(condition_columns, conditions))
    return cohort


def prepare_prediction_tables(features_dir, cohort, fixture_dir, output):
    expected = ("batch", "wafer", "device", "condition", "split_group", "read_voltage_V", "cycle_count",
                "ratio_below_one_cycle_count", *FEATURES)
    features = read_csv(features_dir / "features.csv", expected)
    schema = json.loads((features_dir / "feature_schema.json").read_bytes())
    if schema["model_feature_allowlist"] != list(FEATURES) or schema["target_columns"] != [] or schema["row_key"] != ["device"]:
        raise WalkthroughError("unsupported semiconductor predictor allowlist or row unit")
    by_id = {r["device"]: r for r in features}
    if len(by_id) != len(features) or set(by_id) != {r["physical_device_id"] for r in cohort}:
        raise WalkthroughError("feature export does not match the complete declared device cohort")
    endpoints = read_csv(fixture_dir / "training_endpoints.csv", ("device_id", "target"))
    labels = {r["device_id"]: r["target"] for r in endpoints}
    train_ids = {r["physical_device_id"] for r in cohort if r["partition"] == "train"}
    if len(labels) != len(endpoints) or set(labels) != train_ids:
        raise WalkthroughError("separate training endpoints must match exactly the declared training devices")
    training, future = [], []
    for row in cohort:
        device = row["physical_device_id"]
        observed = by_id[device]
        if (observed["batch"], observed["wafer"], observed["split_group"], observed["cycle_count"]) != (row["batch_id"], row["wafer_id"], row["batch_id"], "3"):
            raise WalkthroughError("exported fabrication provenance or repeated-cycle count changed")
        converted = {"device_id": device, "dependency_block": row["dependency_block"], "environment": row["environment"],
                     **{feature: observed[feature] for feature in FEATURES}}
        if row["partition"] == "train":
            training.append({**converted, "target": labels[device]})
        else:
            future.append(converted)
    write_new(output / "training.csv", csv_bytes(("device_id", "dependency_block", "environment", "target", *FEATURES), training))
    write_new(output / "future.csv", csv_bytes(("device_id", "dependency_block", "environment", *FEATURES), future))
    spec = {"schema_version": "environment-feature-probe-v1", "heldout_environment": ENVS[-1],
            "features_a": list(FEATURES), "features_b": [FEATURES[-1]], "ridge_alpha": 1.0,
            "shuffle_seed": 7, "shuffle_scope": "within_training_environment", "target_unit": "arbitrary_test_units"}
    write_new(output / "spec.json", json_bytes(spec))


def interpretation(prediction, score, audit):
    lines = ["INVENTED SEMICONDUCTOR WALKTHROUGH: SOFTWARE EXAMPLE ONLY", "",
             "Completed measurement-to-features, actual lot holdout audit, target-free prediction, and separate exact-device scoring.",
             "18 invented device IDs; 3 lots/environments; 6 batches/wafers; 54 paired HRS/LRS cycles; 324 raw I-V rows.",
             "12 device rows from 2 lots trained the models. 6 device rows from one held-out lot were scored.",
             "Repeated cycles are aggregated within device. They are not 54 independent devices, and no count here establishes independence.",
             "The three environments repeat the same six invented numeric device templates under new labels; this is not evidence of independent sampling or distributional transfer.",
             "The source generator uses I=V*G arithmetic. The separate endpoint CSVs contain literal arbitrary test values, not targets calculated from HRS, LRS or their ratio.",
             "Environment labels coincide with lots. This is not a randomized intervention or an identified environment effect.", "",
             "SOFTWARE CHECKS", "Lot audit: " + audit["status"],
             "Prediction status: " + prediction["status"],
             "Pipeline fit attempts: " + str(prediction["fit_contract"]["attempted_fits"]),
             "Scored exact device cohort: " + str(score["coverage"]["scored_devices"]), "",
             "DESCRIPTIVE ERRORS IN ARBITRARY TEST UNITS"]
    for arm in ("dummy", "a", "b", "shuffle_a", "shuffle_b"):
        metric = score["metrics"][arm]
        lines.append(f"{arm}: MAE {metric['mae']:.8g}; RMSE {metric['rmse']:.8g}")
    for arm in ("a", "b", "shuffle_a", "shuffle_b"):
        delta = score["metrics"][arm]["mae"] - score["metrics"]["dummy"]["mae"]
        relation = "higher than" if delta > 0 else "lower than" if delta < 0 else "equal to"
        lines.append(f"{arm} MAE is {relation} the mean dummy on these invented rows; this is not a significance result.")
    lines += ["Paired mean absolute-error contrast B minus A: " + format(score["paired_mean_absolute_error_b_minus_a"], ".8g"),
              "A uses HRS median, LRS median and their paired ratio median. B uses only the paired ratio median. These are related measurement summaries, not independent mechanisms.",
              "Negative B-A means lower B error only on these exact invented rows. Neither sign selects a causal feature set or an overall winner.", "",
              "DO NOT OVERINTERPRET", "Held-out prediction comparison is not causal-effect estimation.",
              "The shared within-training-environment label shuffle is a diagnostic, not a formal permutation test or causal negative control. No p-values or confidence intervals are produced.",
              "A no-op shuffle is retained, not redrawn until degradation. See predictions.json for mappings and unchanged-label counts.",
              "This public invented example cannot establish historical preregistration, blinding, correct physical identity, real target timing or experimental generalization.",
              "For real data: justify the endpoint and pre-outcome inputs, dependence boundary, source rights, sample sufficiency and analysis freeze separately. Never create a target from these same resistance features.",
              "Native400 is a separate workflow: these opaque repeated HRS/LRS cycle labels do not represent 400-write chronological paths. Do not feed this feature table into that path comparator.", "",
              "FILES", "cohort.csv: explicit fabrication, lot, batch, wafer, physical device and actual train/test assignments.",
              "raw/: original synthetic I-V inputs after declared label expansion; features/: unchanged exporter reports and lineage.",
              "holdout.json: actual 12/6 lot-boundary audit. training.csv / future.csv: exact model inputs, future has no target.",
              "predictions.json: all five fits, source hashes, dependency boundaries, shuffle and predicted values.",
              "scores.json: all six later supplied outcomes and five-arm paired numerical errors.",
              "COMPLETE.json: last-written receipt of this invocation; hashes are integrity checks, not authentication."]
    return ("\n".join(lines) + "\n").encode()


def run(semiconductor_source, calculator_source, output_dir, *, invented_example_declared=False, fixture_dir=None):
    if invented_example_declared is not True:
        raise WalkthroughError("literal acceptance that all inputs are invented is required")
    semiconductor_source, calculator_source, output = map(Path, (semiconductor_source, calculator_source, output_dir))
    fixture_dir = HERE / "fixtures" if fixture_dir is None else Path(fixture_dir)
    if os.path.lexists(output):
        raise WalkthroughError("output directory must not exist")
    before = component_hashes(semiconductor_source, calculator_source)
    # The held-out endpoint file is not opened, copied or hashed here.
    output.mkdir(mode=0o700, parents=False)
    log = []
    try:
        run_stage("original_synthetic_generator", semiconductor_source / SEMICONDUCTOR_COMPONENTS[0],
                  ["--output-dir", output / "base_fixture"], log, calculator_source)
        cohort = expand_fixture(output / "base_fixture", output / "raw")
        write_new(output / "cohort.csv", csv_bytes(COHORT_COLUMNS, cohort))
        assignments = [{"observation_id": r["physical_device_id"], "partition": r["partition"],
                        "physical_device_id": r["physical_device_id"], "wafer_id": r["wafer_id"], "lot_id": r["lot_id"]} for r in cohort]
        write_new(output / "assignments.csv", csv_bytes(("observation_id", "partition", "physical_device_id", "wafer_id", "lot_id"), assignments))
        run_stage("measurement_to_features", semiconductor_source / SEMICONDUCTOR_COMPONENTS[1],
                  ["--input-dir", output / "raw", "--output-dir", output / "features", "--max-bracket-span-V", "0.11", "--accept-measurement-assumptions"], log, calculator_source)
        run_stage("actual_lot_holdout_audit", semiconductor_source / SEMICONDUCTOR_COMPONENTS[-1],
                  [output / "assignments.csv", "--holdout-level", "lot", "--declare-global-identities", "--output", output / "holdout.json"], log, calculator_source)
        audit = json.loads((output / "holdout.json").read_bytes())
        if audit["status"] != "NO_SELECTED_HOLDOUT_OVERLAP" or audit["counts"]["partition_rows"] != {"train": 12, "test": 6}:
            raise WalkthroughError("actual declared lot holdout did not satisfy the fixed example")
        prepare_prediction_tables(output / "features", cohort, fixture_dir, output)
        run_stage("target_free_prediction", calculator_source / CALCULATOR_COMPONENTS[0],
                  ["predict", output / "training.csv", output / "future.csv", "--spec", output / "spec.json", "--accept-heldout-contract", "--output", output / "predictions.json"], log, calculator_source)
        # First read of the held-out endpoint values by this workflow occurs in score.
        run_stage("separate_exact_cohort_scoring", calculator_source / CALCULATOR_COMPONENTS[0],
                  ["score", output / "predictions.json", fixture_dir / "heldout_outcomes.csv", "--output", output / "scores.json"], log, calculator_source)
        prediction = json.loads((output / "predictions.json").read_bytes())["body"]
        scores = json.loads((output / "scores.json").read_bytes())["body"]
        if prediction["status"] != "COMPLETE_PREDICTIONS" or scores["coverage"]["scored_devices"] != 6:
            raise WalkthroughError("incomplete five-arm prediction or exact-cohort scoring")
        outcome_bytes = (fixture_dir / "heldout_outcomes.csv").read_bytes()
        if sha(outcome_bytes) != scores["outcomes_csv_sha256"]:
            raise WalkthroughError("held-out endpoint bytes changed after scoring")
        write_new(output / "heldout_outcomes.csv", outcome_bytes)
        if component_hashes(semiconductor_source, calculator_source) != before:
            raise WalkthroughError("trusted component source changed during the walkthrough")
        write_new(output / "INTERPRETATION.txt", interpretation(prediction, scores, audit))
        files = {str(p.relative_to(output)): sha(p.read_bytes()) for p in sorted(output.rglob("*")) if p.is_file()}
        receipt = {"schema_version": "semiconductor-heldout-walkthrough-v1", "status": "COMPLETE_INVENTED_SOFTWARE_WALKTHROUGH",
                   "stages": log, "component_source_sha256": before, "output_files_sha256": files,
                   "inputs_are_invented": True, "numeric_templates_repeated_across_environments": True,
                   "heldout_endpoint_first_reader": "separate_exact_cohort_scoring_after_successful_prediction",
                   "scientific_validation": False, "causal_effect_estimation": False, "network_or_downloads": False}
        write_new(output / "COMPLETE.json", json_bytes(receipt))
        return receipt
    except (WalkthroughError, OSError, UnicodeError, ValueError, KeyError, TypeError):
        if not (output / "FAILED.json").exists():
            write_new(output / "FAILED.json", json_bytes({"status": "INCOMPLETE_DO_NOT_USE_AS_COMPLETE_RUN", "stages": log}))
        raise WalkthroughError("walkthrough incomplete; inspect the categorical stage receipt and use a fresh output directory") from None


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument("--semiconductor-source", required=True)
    parser.add_argument("--calculator-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-invented-example", action="store_true", required=True)
    try:
        args = parser.parse_args(argv)
        run(args.semiconductor_source, args.calculator_source, args.output_dir, invented_example_declared=args.run_invented_example)
    except (WalkthroughError, OSError, ValueError):
        print("Error: unable to complete the declared invented walkthrough; check matching local sources, dependencies and fresh output destination", file=sys.stderr)
        return 2
    print("Completed invented semiconductor walkthrough. Read INTERPRETATION.txt in the chosen output directory; no causal or device-performance claim is implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
