"""Original, fixed diagnostic of constant baselines; not a model API."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import groupbench
from groupbench import GroupKFold, benchmark
from groupbench.data import validate_groups, validate_target
from groupbench.metrics import grouped_metrics


HERE = Path(__file__).resolve().parent


def fixture():
    """No random draw, external source, private data, or fitted parameter."""
    rows = []
    for index in range(12):
        count, target = (40, 10.0) if index < 3 else (4, 0.0)
        for row in range(count):
            rows.append({"row_id": f"g{index:02d}_r{row:02d}",
                         "group": f"g{index:02d}", "constant_feature": 1.0,
                         "target": target})
    return pd.DataFrame(rows)


def training_constants(target, groups):
    """Equal-group absolute error weights rows by inverse *training* group size.

    The lower weighted median minimizes this training objective. It is not the
    median of group medians; neither property guarantees lower held-out loss.
    Fraction weights make the 50% tie rule exact on this bounded companion.
    """
    y = validate_target(target, len(groups))
    g = validate_groups(groups, len(y))
    counts = Counter(g)
    weights = [Fraction(1, counts[group]) for group in g]
    total = sum(weights, Fraction())
    cumulative = Fraction()
    for value, weight in sorted(zip(y, weights), key=lambda item: item[0]):
        cumulative += weight
        if cumulative * 2 >= total:
            group_median = float(value)
            break
    group_mean = float(sum((Fraction(float(value)) * weight
                           for value, weight in zip(y, weights)), Fraction()) / total)
    return {"row_median": float(np.median(y)), "group_mean": group_mean,
            "group_median": group_median}


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, document):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")


def preflight(protocol):
    if versions() != protocol["required_versions"]:
        raise ValueError("Reference runtime mismatch; no fitting performed")
    package = Path(groupbench.__file__).resolve().parent
    observed = {name: sha256(package / name)
                for name in protocol["required_source_sha256"]}
    if observed != protocol["required_source_sha256"]:
        raise ValueError("Reference product source mismatch; no fitting performed")
    return observed


def run(output):
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    observed_source = preflight(protocol)
    output = Path(output)
    output.mkdir()  # Existing directories and live/dangling symlinks fail.
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    deadline = int(time.process_time()) + protocol["resource_cap"]["cpu_seconds"]
    bounds = [deadline] + [value for value in (soft, hard) if value != resource.RLIM_INFINITY]
    resource.setrlimit(resource.RLIMIT_CPU, (min(bounds), hard))
    json_write(output / "run-started.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(HERE / "protocol.json"),
        "runner_sha256": sha256(HERE / "run.py"),
        "source_sha256": observed_source, "versions": versions(),
        "evidence_kind": "actual author execution of original synthetic fixture",
        "complete": False})
    try:
        frame = fixture()
        frame.to_csv(output / "fixture.csv", index=False, mode="x")
        json_write(output / "protocol.json", protocol)
        splits = list(GroupKFold(protocol["outer_splits"], protocol["random_state"])
                      .split(frame, groups=frame.group))
        planned_folds = np.full(len(frame), -1, dtype=int)
        constants = {name: np.full(len(frame), np.nan)
                     for name in protocol["comparison_constants"]}
        constant_audit = []
        for fold, (train, test) in enumerate(splits):
            values = training_constants(frame.target.to_numpy()[train], frame.group.to_numpy()[train])
            planned_folds[test] = fold
            for name, value in values.items():
                constants[name][test] = value
            constant_audit.append({"fold": fold, "train_row_positions": train.tolist(),
                                   "test_row_positions": test.tolist(), "constants": values,
                                   "training_group_sizes": dict(sorted(Counter(frame.group.to_numpy()[train]).items()))})
        json_write(output / "constant_audit.json", constant_audit)
        fit_calls = 0
        original_fit = Pipeline.fit

        def counted_fit(estimator, *args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            return original_fit(estimator, *args, **kwargs)

        with threadpool_limits(limits=1), patch.object(Pipeline, "fit", counted_fit):
            result = benchmark(frame, target="target", group="group", features=["constant_feature"],
                               outer_splits=protocol["outer_splits"], inner_splits=protocol["inner_splits"],
                               random_state=protocol["random_state"], baselines=tuple(protocol["baselines"]),
                               objective=protocol["selection_objective"])
        np.testing.assert_array_equal(result.predictions.outer_fold, planned_folds)
        if fit_calls != 45:
            raise AssertionError("Actual pipeline fit count differs from fixed plan")
        predictions = result.predictions.copy()
        predictions.insert(1, "row_id", frame.row_id)
        for name, values in constants.items():
            predictions["prediction_" + name] = values
        if not np.isfinite(predictions.filter(like="prediction_").to_numpy()).all():
            raise AssertionError("Incomplete or nonfinite predictions")
        predictions.to_csv(output / "predictions.csv", index=False, mode="x", float_format="%.17g")
        metrics = {}
        for column in predictions.filter(like="prediction_"):
            overall, by_group = grouped_metrics(frame.target, predictions[column], frame.group)
            metrics[column.removeprefix("prediction_")] = {
                "overall": overall, "per_group": by_group}
        json_write(output / "product_manifest.json", result.manifest)
        json_write(output / "summary.json", {
            "schema_version": protocol["schema_version"], "status": "complete_synthetic_diagnostic",
            "n_rows": len(frame), "n_declared_groups": frame.group.nunique(),
            "observed_pipeline_fit_calls": fit_calls, "analytic_constants": 9,
            "outer_membership_parity": True, "metrics": metrics,
            "constant_audit": constant_audit,
            "interpretation": protocol["interpretation"]})
        json_write(output / "cost.json", {
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "timing_boundary": "after output directory reservation through summary serialization; excludes imports and preflight",
            "max_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rlimit_cpu_soft_seconds": min(bounds), "threads_per_library": 1,
            "download_bytes": 0, "fit_calls": fit_calls, "analytic_constants": 9,
            "not_billed_platform_cost": True})
        json_write(output / "manifest.json", {
            "complete": True, "files": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
            "limitations": "Hashes are integrity observations, not proof of chronology or independent replication."})
    except BaseException as exc:
        json_write(output / "failure.json", {"complete": False, "error_type": type(exc).__name__})
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args.output)
    except (ValueError, OSError, AssertionError) as exc:
        print(f"Diagnostic rejected or incomplete: {exc}", file=sys.stderr)
        return 2
    print("Complete synthetic diagnostic; not real-data performance validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
