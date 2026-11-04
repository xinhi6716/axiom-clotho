"""Compare explicit numeric diagnostic APIs on synthetic tables, without fits.

Install the optional package first. No monkeypatching or default dispatch is
used. Results include Python validation, byte snapshots, native work and report
construction; they do not measure model fitting or prediction planning.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

import clotho_numeric_support
from clotho_numeric_support import prediction_input_support_numeric
from groupbench import input_support

CASES = {"small": (288, 32, 12), "large": (20_000, 2_000, 100),
         "bounded-max": (20_000, 20_000, 100)}


def fixture(n, m, p):
    rng = np.random.default_rng(731)
    def table(rows, query):
        values = rng.uniform(100, 900, (rows, p))
        values[rng.random((rows, p)) < .04] = np.nan
        values[:, -1] = 1.0 if query else np.nan
        if query:
            values[0, 0] = 1000.0
        return pd.DataFrame(values, columns=[f"synthetic_process_parameter_{i}" for i in range(p)])
    return table(n, False), table(m, True)


def measure_pair(training, query, repeats):
    functions = {"python": lambda: input_support.prediction_input_support(training, query),
                 "native": lambda: prediction_input_support_numeric(training, query)}
    before = (training.copy(deep=True), query.copy(deep=True))
    expected = functions["python"]()
    for _ in range(2):
        for function in functions.values():
            if function() != expected:
                raise RuntimeError("Diagnostic parity failed during warmup.")
    samples = {name: [] for name in functions}
    for repetition in range(repeats):
        order = ("python", "native") if repetition % 2 == 0 else ("native", "python")
        for name in order:
            start = time.perf_counter_ns()
            actual = functions[name]()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if actual != expected:
                raise RuntimeError("Diagnostic parity failed during measurement.")
            samples[name].append(elapsed)
    pd.testing.assert_frame_equal(training, before[0])
    pd.testing.assert_frame_equal(query, before[1])
    result = {name: {"median_ms": statistics.median(values), "min_ms": min(values),
                     "max_ms": max(values), "samples_ms": values} for name, values in samples.items()}
    result["all_report_fields_exact"] = True
    result["inputs_unchanged"] = True
    return result


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="small")
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 25:
        parser.error("--repeats must be between 1 and 25")
    native = clotho_numeric_support._load_native()
    report = {
        "kind": "synthetic_explicit_numeric_diagnostic", "fits_performed": 0,
        "warmup_calls_per_backend": 2, "repetitions": args.repeats,
        "measurement_order": "alternating_python_native_and_native_python",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "optional_package": importlib.metadata.version("clotho-numeric-support"),
                        "thread_settings": {key: os.environ.get(key) for key in (
                            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "sha256": {"benchmark": file_hash(__file__), "python_diagnostic": file_hash(input_support.__file__),
                   "optional_wrapper": file_hash(clotho_numeric_support.__file__),
                   "native_binary": file_hash(native.__file__)},
        "cases": [],
        "limitations": ["Warm synthetic host-specific timings, not model accuracy or model-training speed.",
                        "The existing planning and prediction routes do not select this backend.",
                        "Costs include raw query validation, immutable snapshots and report construction.",
                        "Native import and source-build time are excluded; timing varies with host load."]}
    for name in CASES if args.case == "all" else [args.case]:
        n, m, p = CASES[name]
        training, query = fixture(n, m, p)
        report["cases"].append({"name": name, "training_rows": n, "prediction_rows": m,
                                "numeric_features": p, **measure_pair(training, query, args.repeats)})
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
