#!/usr/bin/env python3
"""Simulation-only causal-design rehearsal, never a calibrated device model.

Fixed invented equations separate pre-setting context, adjustable setting,
post-setting intermediary and readout. No real CSV input or empirical causal
certification is offered. Population truth is stipulated by each simulator.
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
import sys

import numpy as np
from threadpoolctl import threadpool_limits

SCENARIOS = ("measured_context", "omitted_device_confounder", "deterministic_assignment")
OBS_COLUMNS = ("device_id", "batch_id", "batch_context", "process_setting", "intermediary", "readout")
PAIR_COLUMNS = ("device_id", "batch_id", "intermediary_do_0", "intermediary_do_1", "readout_do_0", "readout_do_1", "total_contrast", "controlled_direct_contrast_at_intermediary_0")
PLANS = (("naive_association", "association", ()),
         ("context_adjusted", "total_effect", ("batch_context",)),
         ("intermediary_adjusted", "controlled_direct_effect", ("batch_context", "intermediary")))
RANK_RTOL = 1e-12
MAX_FILE_BYTES = 8 * 1024 * 1024
LIMITATIONS = [
    "Every value, identity, coefficient and equation is invented. This is not measured semiconductor data, a calibrated device model, RRAM/PINN validation or evidence about a physical mechanism.",
    "The setting is fictional and device-level. A shared batch process cannot generally be changed independently per device; the example does not model such an intervention.",
    "Intervention truth follows only from the explicitly stipulated equations and unchanged exogenous draws. It is not independently discovered from observations.",
    "Known simulation structure supplies consistency, no interference, timing, independent errors and model form. Real data require separate defensible assumptions and source knowledge.",
    "Batch context is shared. Device count is not an unconditional independent sample size. No p-values, confidence intervals or precision guarantees are produced.",
    "Numerical rank at a stated tolerance is not proof of positivity, absence of confounding, correct causal roles or general identification.",
    "No predictive-error selection, feature discovery, hyperparameter search, seed redraw, model download, executable saved model or network operation occurs.",
]


class InputError(ValueError):
    """Categorical diagnostics omit caller paths and values."""


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise InputError("invalid rehearsal arguments; use --help")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def json_bytes(value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True) + "\n").encode()
    if len(data) > MAX_FILE_BYTES:
        raise InputError("complete output file exceeds byte bound")
    return data


def csv_bytes(columns, records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    data = stream.getvalue().encode()
    if len(data) > MAX_FILE_BYTES:
        raise InputError("complete output file exceeds byte bound")
    return data


def validate_counts(batches, devices_per_batch, seed):
    if type(batches) is not int or not 2 <= batches <= 128:
        raise InputError("batch count must be an integer from 2 through 128")
    if type(devices_per_batch) is not int or not 2 <= devices_per_batch <= 64:
        raise InputError("devices per batch must be an integer from 2 through 64")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise InputError("seed must be an unsigned 32-bit integer")


def draw_exogenous(batches, devices_per_batch, seed):
    validate_counts(batches, devices_per_batch, seed)
    rng = np.random.default_rng(seed)
    size = batches * devices_per_batch
    return {"C": np.repeat(rng.standard_normal(batches), devices_per_batch),
            **{name: rng.standard_normal(size) for name in ("UA", "UM", "UY", "U")}}


def structural_values(exogenous, scenario, *, do_setting=None, do_intermediary=None):
    """Intervention surgery: replace only the intervened structural equation."""
    if scenario not in SCENARIOS or set(exogenous) != {"C", "UA", "UM", "UY", "U"}:
        raise InputError("unsupported fixed simulation structure")
    arrays = {k: np.asarray(v, dtype=float) for k, v in exogenous.items()}
    shape = arrays["C"].shape
    if len(shape) != 1 or not 4 <= shape[0] <= 8192 or any(v.shape != shape or not np.isfinite(v).all() for v in arrays.values()):
        raise InputError("exogenous test draws must be bounded, finite equal-length vectors")
    for value in (do_setting, do_intermediary):
        if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or abs(value) > 100):
            raise InputError("intervention test value must be a bounded finite number")
    context = arrays["C"]
    hidden = arrays["U"] if scenario == "omitted_device_confounder" else np.zeros(shape)
    setting = context if scenario == "deterministic_assignment" else context + arrays["UA"] + hidden
    if do_setting is not None:
        setting = np.full(shape, do_setting, dtype=float)
    intermediary = 2 * setting + context + arrays["UM"]
    if do_intermediary is not None:
        intermediary = np.full(shape, do_intermediary, dtype=float)
    readout = setting + 3 * intermediary + 2 * context + 2 * hidden + arrays["UY"]
    if any(not np.isfinite(value).all() for value in (setting, intermediary, readout)):
        raise InputError("structural arithmetic exceeded finite range")
    return {"batch_context": context.copy(), "process_setting": setting.copy(),
            "intermediary": intermediary.copy(), "readout": readout.copy()}


def observation_rows(values, batches, devices_per_batch):
    records = []
    for index in range(batches * devices_per_batch):
        batch = "invented_batch_" + format(index // devices_per_batch + 1, "03d")
        records.append({"device_id": batch + "/device_" + format(index % devices_per_batch + 1, "03d"),
                        "batch_id": batch, **{k: float(values[k][index]) for k in OBS_COLUMNS[2:]}})
    return records


def estimate_observational(records, *, estimand, adjustment):
    """OLS sees ONLY observed rows and a declared role/estimand request.

    This validates the fixed example's role contract, not causal identification.
    No simulator parameters, latent draws or intervention outcomes are accepted.
    """
    allowed = {"association": (), "total_effect": ("batch_context",),
               "controlled_direct_effect": ("batch_context", "intermediary")}
    if (not isinstance(estimand, str) or estimand not in allowed or type(adjustment) is not tuple
            or adjustment != allowed[estimand]):
        raise InputError("unsupported estimand/adjustment roles; a total-effect plan must not condition on the intermediary")
    if not isinstance(records, list) or not 4 <= len(records) <= 8192:
        raise InputError("observational rows outside the fixed rehearsal bound")
    devices, contexts = set(), {}
    for row in records:
        if not isinstance(row, dict) or set(row) != set(OBS_COLUMNS):
            raise InputError("estimation accepts only exact observed columns, never latent or oracle fields")
        for name in ("device_id", "batch_id"):
            value = row[name]
            if not isinstance(value, str) or not value or len(value) > 128 or value != value.strip() or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value):
                raise InputError("observational identity must be a bounded exact control-free string")
        if row["device_id"] in devices:
            raise InputError("one observed row per invented device is required")
        devices.add(row["device_id"])
        if any(type(row[name]) not in (float, int) or not math.isfinite(row[name]) or abs(row[name]) > 1e6 for name in OBS_COLUMNS[2:]):
            raise InputError("observed numeric test values must be finite and bounded")
        if row["batch_id"] in contexts and row["batch_context"] != contexts[row["batch_id"]]:
            raise InputError("the fixed simulation requires one shared context value per batch")
        contexts[row["batch_id"]] = row["batch_context"]
    columns = ("intercept", "process_setting", *adjustment)
    design = np.array([[1.0, row["process_setting"], *[row[k] for k in adjustment]] for row in records])
    target = np.array([row["readout"] for row in records])
    with threadpool_limits(limits=1):
        singular = np.linalg.svd(design, compute_uv=False)
        cutoff = RANK_RTOL * float(singular[0])
        rank = int(np.sum(singular > cutoff))
        result = {"requested_estimand": estimand, "adjustment": list(adjustment), "columns": list(columns),
                  "rows": len(records), "declared_batches": len({row["batch_id"] for row in records}),
                  "rank": rank, "required_rank": len(columns), "relative_rank_tolerance": RANK_RTOL,
                  "absolute_singular_value_cutoff": cutoff, "singular_values": singular.tolist(),
                  "full_rank_condition_number": float(singular[0] / singular[-1]) if rank == len(columns) else None,
                  "causal_identification_checked_by_ols": False,
                  "status": "NUMERICAL_RANK_DEFICIENT" if rank < len(columns) else "OLS_COEFFICIENT_AVAILABLE",
                  "setting_coefficient": None, "coefficients": None}
        if rank < len(columns):
            # Do not expose a pseudoinverse/regularized coefficient as an effect.
            return result
        coefficients, _, solved_rank, _ = np.linalg.lstsq(design, target, rcond=RANK_RTOL)
    if solved_rank != len(columns) or not np.isfinite(coefficients).all():
        raise InputError("full-rank OLS could not produce finite coefficients")
    result["coefficients"] = dict(zip(columns, coefficients.tolist()))
    result["setting_coefficient"] = float(coefficients[1])
    return result


def scenario_contract(scenario):
    if scenario not in SCENARIOS:
        raise InputError("unsupported fixed scenario")
    setting = {"measured_context": "A = C + UA", "omitted_device_confounder": "A = C + UA + U", "deterministic_assignment": "A = C"}[scenario]
    readout = "Y = A + 3*M + 2*C + UY" + (" + 2*U" if scenario == "omitted_device_confounder" else "")
    return {"equations": ["C = UC, shared within each invented batch", setting, "M = 2*A + C + UM", readout],
            "graph_edges": ["C->A", "C->M", "C->Y", "A->M", "A->Y", "M->Y"] + (["U->A", "U->Y"] if scenario == "omitted_device_confounder" else []),
            "exogenous_distribution": "mutually independent standard normal draws; UC is batch-level, UA/UM/UY/U are device-level",
            "unused_exogenous_draws": ["UA", "U"] if scenario == "deterministic_assignment" else ["U"] if scenario == "measured_context" else [],
            "assignment_noise_present": scenario != "deterministic_assignment",
            "observed_context_suffices_by_stipulated_structure": scenario == "measured_context",
            "identification_note": {
                "measured_context": "Under this exact additive SCM, pre-setting C blocks confounding; full conditional Gaussian setting support is stipulated, not certified by finite-sample rank. Conditioning on M instead targets a controlled direct effect under this SCM only.",
                "omitted_device_confounder": "U affects both A and Y and is excluded from observational regressions. C adjustment does not identify the total effect; mediator-adjusted OLS does not identify the controlled direct effect. Numeric coefficients remain descriptive, even when close to oracle truth.",
                "deterministic_assignment": "A equals C exactly. Treatment is determined by the required context, so the requested conditional comparison lacks assignment support and adjusted columns are aliased. Intervention truth remains stipulated by the simulator, not recovered from observations."
            }[scenario],
            "base_population_observational_coefficients": {"naive_association": 9.5, "context_adjusted": 7.0, "intermediary_adjusted": 1.0} if scenario == "measured_context" else None,
            "population_coefficient_note": "Base targets follow Y=7*A+5*C+3*UM+UY and Cov(A,C)/Var(A)=1/2. They are population quantities, not finite-sample estimates or physical coefficients.",
            "analytic_oracle_for_this_stipulated_scenario": {"do_A_1_minus_0_recomputing_M": 7.0, "do_A_1_minus_0_with_do_M_0": 1.0},
            "oracle_derivation": "For this scenario's equations, replacing A by a changes M by 2 per unit and Y by 1+3*2=7. Also intervening to fix M removes that path and gives 1. This algebra says nothing about empirical devices."}


def interpretation(report):
    text = ["SIMULATION-ONLY CAUSAL-DESIGN REHEARSAL", "",
            "Illustrative dimensionless context C, device-level setting A, intermediary M and readout Y.",
            "No physical coefficients, calibrated device simulation or real-data causal validation.",
            f"{report['configuration']['batches']} invented batches and {report['configuration']['devices']} device labels per scenario; not an unconditional independent sample count.",
            "Three fixed analysis plans and all scenarios are retained; nothing is selected by predictive error or by closeness to oracle truth.", ""]
    for name in SCENARIOS:
        item = report["scenarios"][name]
        text += [name.upper(), item["contract"]["identification_note"]]
        for model, estimate in item["observational_estimates"].items():
            coefficient = estimate["setting_coefficient"]
            shown = "unavailable at declared numerical rank tolerance" if coefficient is None else format(coefficient, ".8g")
            text.append(model + ": setting coefficient " + shown + "; requested quantity " + estimate["requested_estimand"] + ".")
        text.append("Paired simulator-only total contrast: " + format(item["paired_oracle_summary"]["mean_total_contrast"], ".8g") + "; controlled direct contrast at do(M=0): " + format(item["paired_oracle_summary"]["mean_controlled_direct_contrast"], ".8g") + ".")
        if name == "measured_context":
            text.append("Population regression targets under this base SCM are 9.5, 7 and 1; finite seeded observations need not equal them. The latter two answer different questions.")
        text += ["Oracle contrasts were generated by intervention surgery, not estimated by these regressions.", ""]
    text += ["HOW TO USE THE REHEARSAL", "Check roles and target quantity before analysis: a total-effect request with the intermediary in its adjustment list is rejected.",
             "The estimators accept only observed columns. Latent draws and paired oracle outcomes are separate and never given to OLS.",
             "A Gaussian SCM with assignment noise stipulates support; finite rank does not verify causal positivity. A numerical near-collinearity rejection is not proof that assignment noise is exactly zero.",
             "Device-specific interventions and no interference are assumptions of this fictional design, not a model of shared batch annealing.",
             "For real research, justify intervention unit, timing, estimand, adjustment, confounding, support, dependence and uncertainty separately. This command does not load real data or identify a real effect.", "LIMITATIONS", *report["limitations"]]
    return ("\n".join(text) + "\n").encode()


def rehearse(*, batches=32, devices_per_batch=16, seed=7):
    exogenous = draw_exogenous(batches, devices_per_batch, seed)
    files = {}
    report = {"schema_version": "causal-design-rehearsal-v1", "status": "COMPLETE_SIMULATION_REHEARSAL",
              "configuration": {"batches": batches, "devices_per_batch": devices_per_batch, "devices": batches * devices_per_batch, "seed": seed, "units": "dimensionless_invented_indices", "scenarios": list(SCENARIOS), "same_exogenous_draws_across_scenarios": True},
              "variable_roles": {"C/batch_context": "observed pre-setting shared batch context", "A/process_setting": "fictional adjustable device-level setting", "M/intermediary": "observed after setting, before readout", "Y/readout": "invented device readout", "U": "device-level latent confounder used only in the omitted-confounder scenario"},
              "prespecified_plans": [{"name": n, "requested_estimand": e, "adjustment": list(a)} for n, e, a in PLANS],
              "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
              "scenarios": {}, "limitations": LIMITATIONS.copy()}
    for scenario in SCENARIOS:
        observed = observation_rows(structural_values(exogenous, scenario), batches, devices_per_batch)
        estimates = {name: estimate_observational(observed, estimand=estimand, adjustment=adjustment) for name, estimand, adjustment in PLANS}
        low = structural_values(exogenous, scenario, do_setting=0)
        high = structural_values(exogenous, scenario, do_setting=1)
        direct_low = structural_values(exogenous, scenario, do_setting=0, do_intermediary=0)
        direct_high = structural_values(exogenous, scenario, do_setting=1, do_intermediary=0)
        pairs = [{"device_id": row["device_id"], "batch_id": row["batch_id"],
                  "intermediary_do_0": float(low["intermediary"][i]), "intermediary_do_1": float(high["intermediary"][i]),
                  "readout_do_0": float(low["readout"][i]), "readout_do_1": float(high["readout"][i]),
                  "total_contrast": float(high["readout"][i] - low["readout"][i]),
                  "controlled_direct_contrast_at_intermediary_0": float(direct_high["readout"][i] - direct_low["readout"][i])} for i, row in enumerate(observed)]
        total = [row["total_contrast"] for row in pairs]
        direct = [row["controlled_direct_contrast_at_intermediary_0"] for row in pairs]
        if max(abs(v - 7) for v in total) > 1e-9 or max(abs(v - 1) for v in direct) > 1e-9:
            raise InputError("paired intervention arithmetic disagrees with the fixed declared equations")
        files[scenario + '.observations.csv'] = csv_bytes(OBS_COLUMNS, observed)
        files[scenario + '.paired_interventions.csv'] = csv_bytes(PAIR_COLUMNS, pairs)
        report["scenarios"][scenario] = {"contract": scenario_contract(scenario), "observational_estimates": estimates,
            "paired_oracle_summary": {"devices": len(pairs), "mean_total_contrast": math.fsum(total) / len(total),
                "mean_controlled_direct_contrast": math.fsum(direct) / len(direct),
                "maximum_absolute_total_algebra_discrepancy": max(abs(v - 7) for v in total),
                "maximum_absolute_direct_algebra_discrepancy": max(abs(v - 1) for v in direct),
                "truth_source": "declared simulation equations and paired intervention surgery, never observed-regression fit"}}
    files["SCM.json"] = json_bytes({"schema_version": "causal-design-rehearsal-scm-v1", "simulation_only": True, "configuration": report["configuration"], "roles": report["variable_roles"], "scenarios": {name: report["scenarios"][name]["contract"] for name in SCENARIOS}})
    files["REPORT.json"] = json_bytes(report)
    files["INTERPRETATION.txt"] = interpretation(report)
    return files, report


def write_new(path, blob):
    if len(blob) > MAX_FILE_BYTES:
        raise InputError("complete output file exceeds byte bound")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(blob)


def run(output, *, batches=32, devices_per_batch=16, seed=7, simulation_declared=False):
    if simulation_declared is not True:
        raise InputError("literal simulation-only acknowledgement is required")
    output = Path(output)
    if os.path.lexists(output):
        raise InputError("output directory must not already exist")
    source_sha = sha(Path(__file__).read_bytes())
    # Complete computation/serialization precedes creation of the output tree.
    files, report = rehearse(batches=batches, devices_per_batch=devices_per_batch, seed=seed)
    if sha(Path(__file__).read_bytes()) != source_sha:
        raise InputError("source changed during computation")
    receipt = {"schema_version": "causal-design-rehearsal-receipt-v1", "status": report["status"],
               "source_sha256": source_sha, "output_files_sha256": {name: sha(data) for name, data in files.items()},
               "simulation_only": True, "empirical_effect_identified": False, "network_or_downloads": False,
               "integrity_not_authentication": True}
    output.mkdir(mode=0o700)
    try:
        for name, data in files.items():
            write_new(output / name, data)
        write_new(output / "COMPLETE.json", json_bytes(receipt))
    except OSError:
        raise InputError("new output is incomplete; do not treat it as a completed rehearsal") from None
    return receipt


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--devices-per-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--accept-simulation-only", action="store_true", required=True)
    try:
        args = parser.parse_args(argv)
        run(args.output_dir, batches=args.batches, devices_per_batch=args.devices_per_batch, seed=args.seed, simulation_declared=args.accept_simulation_only)
    except (InputError, OSError, ValueError, np.linalg.LinAlgError):
        print("Error: rehearsal rejected; use --help, supported counts and a fresh local output directory", file=sys.stderr)
        return 2
    print("Completed invented causal-design rehearsal. Read INTERPRETATION.txt; no empirical device effect is identified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
