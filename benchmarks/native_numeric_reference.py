"""Measure explicit training-reference reuse, including construction, with no fits.

All inputs are invented. Compares existing Python and native one-shot APIs with
an explicitly created native reference over the same successive query batches.
No default backend, hidden cache, model, target, file or network I/O is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import pandas as pd

import clotho_numeric_support as adapter
from groupbench import input_support

CASES = {"small": (288, 32, 12), "repeated-small-query": (20_000, 128, 100),
         "bounded-max": (20_000, 20_000, 100)}


def fixture(n, m, p, batches):
    rng = np.random.default_rng(20261105)
    def table(rows, query):
        data = rng.uniform(100., 900., size=(rows, p))
        data[rng.random(data.shape) < .04] = np.nan
        data[:, -1] = 1. if query else np.nan
        if query:
            data[0, 0] = 1000.
        return pd.DataFrame(data, columns=[f"synthetic_parameter_{i}" for i in range(p)])
    return table(n, False), [table(m, True) for _ in range(batches)]


def measure(training, queries, repeats):
    expected = [input_support.prediction_input_support(training, query) for query in queries]
    training_before = training.copy(deep=True)
    queries_before = [q.copy(deep=True) for q in queries]
    retained = adapter.NumericSupportReference(training)
    def reference_with_creation():
        reference = adapter.NumericSupportReference(training)
        return [reference.compare(q) for q in queries]
    functions = {
        "python_one_shot_batches": lambda: [input_support.prediction_input_support(training, q) for q in queries],
        "native_one_shot_batches": lambda: [adapter.prediction_input_support_numeric(training, q) for q in queries],
        "reference_including_creation": reference_with_creation,
        "reference_queries_only": lambda: [retained.compare(q) for q in queries],
    }
    for _ in range(2):
        for function in functions.values():
            if function() != expected:
                raise RuntimeError("Full-report equality failed during warmup.")
    samples = {name: [] for name in functions}
    samples["reference_creation_only"] = []
    names = list(functions)
    for repeat in range(repeats):
        # Rotate all four positions, avoiding a fixed first/warmest implementation.
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in order:
            start = time.perf_counter_ns()
            result = functions[name]()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if result != expected:
                raise RuntimeError("Full-report equality failed during measurement.")
            samples[name].append(elapsed)
        start = time.perf_counter_ns()
        new_reference = adapter.NumericSupportReference(training)
        samples["reference_creation_only"].append((time.perf_counter_ns() - start) / 1e6)
        if new_reference.compare(queries[0]) != expected[0]:
            raise RuntimeError("Construction check failed.")
    pd.testing.assert_frame_equal(training, training_before)
    for actual, before in zip(queries, queries_before):
        pd.testing.assert_frame_equal(actual, before)
    return {"all_report_fields_exact": True, "inputs_unchanged": True,
            "timings": {name: {"median_ms": statistics.median(v), "samples_ms": v}
                        for name, v in samples.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="small")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 25 or not 1 <= args.batches <= 20:
        parser.error("--repeats must be 1..25 and --batches 1..20")
    native = adapter._load_native()
    report = {
        "kind": "synthetic_explicit_numeric_reference", "fits_performed": 0,
        "batches_per_sample": args.batches, "repetitions": args.repeats,
        "warmup_calls_per_path": 2, "measurement_order": "rotating_four_paths",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "thread_settings": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in (
            ("benchmark", __file__), ("python_diagnostic", input_support.__file__),
            ("optional_wrapper", adapter.__file__), ("native_binary", native.__file__))},
        "cases": [],
        "limitations": ["Synthetic host-specific warm timings; no accuracy, estimator training or default-runtime improvement is claimed.",
                        "Each reference_including_creation sample creates a fresh reference and then evaluates every batch.",
                        "reference_queries_only excludes creation, reported separately; import and compilation are excluded.",
                        "All returned fields are checked exactly after each measured call; correctness comparisons are outside timed intervals.",
                        "No RSS or cross-platform measurements; caller-owned frames and query snapshots still consume memory."]}
    for name in CASES if args.case == "all" else [args.case]:
        n, m, p = CASES[name]
        training, queries = fixture(n, m, p, args.batches)
        report["cases"].append({"name": name, "training_rows": n, "query_rows_per_batch": m,
                                "features": p, **measure(training, queries, args.repeats)})
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
