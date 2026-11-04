"""Synthetic warm prediction-planning benchmark; no fits, downloads or writes.

Use PYTHONPATH to select a checkout. This script does not change import paths or
patch validation. Measurements apply to this host, fixture and dependency set,
not overall model-training speed or scientific predictive performance.
"""

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np
import pandas as pd
import sklearn

from groupbench import plan_prediction
from groupbench.data import prepare_data
from groupbench.input_support import prediction_input_support
from groupbench.package_source import observe_package_source


CASES = {"small": (288, 32, 12, 4), "large": (20_000, 2_000, 32, 8),
         "bounded-max": (20_000, 20_000, 100, 20)}


def fixture(n_training, n_prediction, n_features, n_categorical):
    """Invented material/process names and values, not a physical model."""
    rng = np.random.default_rng(731)
    names = ["deposition_temperature_c", "pressure_pa", "thickness_nm", "anneal_duration_s",
             "dopant_fraction", "gas_flow_sccm", "substrate_temperature_c", "growth_rate_nm_s"]

    def table(rows, query=False):
        values = {}
        for column in range(n_features - n_categorical):
            numbers = rng.uniform(100, 900, rows)
            numbers[rng.random(rows) < .04] = np.nan
            if query:
                numbers[0] = 1000.
            if column == n_features - n_categorical - 1:
                numbers[:] = 1. if query else np.nan
            values[f"{names[column % len(names)]}_{column}"] = numbers
        for column in range(n_categorical):
            labels = rng.choice(["Si", "SiC", "GaN", "Ga2O3", "AlN", "IGZO"], rows).astype(object)
            labels[rng.random(rows) < .04] = None
            if query:
                labels[0] = "new_material"
            values[f"material_or_process_family_{column}"] = labels
        return pd.DataFrame(values)

    training = table(n_training)
    training["mobility_cm2_Vs"] = rng.uniform(1, 100, n_training)
    training["batch"] = [f"lot-{row // 12:05d}" for row in range(n_training)]
    return training, table(n_prediction, query=True)


def measure(function, repeats):
    expected = function()
    assert function() == expected
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        actual = function()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        assert actual == expected, "Deterministic plan/diagnostic output changed across repetitions."
        samples.append(elapsed)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples),
            "max_ms": max(samples), "samples_ms": samples}, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="small")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 25:
        parser.error("--repeats must be between 1 and 25")
    report = {
        "kind": "synthetic_warm_prediction_preflight", "fits_performed": 0,
        "warmup_calls": 2, "repetitions": args.repeats,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                        "thread_settings": {key: os.environ.get(key) for key in (
                            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "package_source": observe_package_source(), "cases": [],
        "limitations": ["Synthetic measurements do not establish real-data predictive performance.",
                        "Wall-clock timing depends on host load and dependency versions.",
                        "No estimator is fitted; this does not measure or promise model-training acceleration."],
    }
    for name in CASES if args.case == "all" else [args.case]:
        rows, new_rows, columns, categorical = CASES[name]
        frame, query = fixture(rows, new_rows, columns, categorical)
        training, _, _ = prepare_data(frame, "mobility_cm2_Vs", "batch")
        plan_timing, plan = measure(lambda: plan_prediction(
            frame, query, target="mobility_cm2_Vs", group="batch"), args.repeats)
        diagnostic_timing, diagnostic = measure(lambda: prediction_input_support(training, query), args.repeats)
        assert plan["input_support"] == diagnostic
        assert plan["fits_performed"] == 0
        report["cases"].append({
            "name": name, "training_rows": rows, "prediction_rows": new_rows,
            "features": columns, "categorical_features": categorical,
            "prediction_plan": plan_timing, "public_diagnostic": diagnostic_timing,
            "input_support": diagnostic, "fit_budget": plan["fit_budget"],
        })
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
