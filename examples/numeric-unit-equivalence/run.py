"""Run twelve fixed fits on an already-exposed synthetic unit-equivalence example."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from importlib.metadata import version as distribution_version
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_info, threadpool_limits

from groupbench.model import Candidate, make_pipeline
import groupbench.model as product_model
from generate_synthetic import SEED, TRAIN_GROUPS, TEST_GROUPS, generate, write_fixture

FAMILIES = ("dummy", "ridge", "extra_trees")
ARMS = ("raw", "calculator_scaled")
UNITS = ("A", "nA")
PARAMETERS = {
    "dummy": {"strategy": "mean"},
    "ridge": {"alpha": 1.0},
    "extra_trees": {"n_estimators": 64, "max_depth": 6, "min_samples_leaf": 2},
}
REFERENCE_SKLEARN = "1.8.0"
PUBLIC_SOURCE = "26b6eb8f6a149d760aec0aff1fbd680f0f32dbb6"
MODEL_SOURCE_SHA256 = "a9c292336c85523e667b6b8417c89ea1421b97d51f83cf9394ca5673190c242a"


def validate_fixture(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    expected = ["row_id", "group", "split", "current_A", "current_nA", "y"]
    if list(frame.columns) != expected or len(frame) != 384:
        raise ValueError("Unexpected fixture schema or row count")
    if not np.array_equal(frame["row_id"], np.arange(384)):
        raise ValueError("Unexpected fixture row identities")
    if not np.array_equal(frame["group"], np.repeat(np.arange(12), 32)):
        raise ValueError("Unexpected fixed group assignment")
    train = frame["group"].to_numpy() < 8
    test = ~train
    if not np.array_equal(frame["split"], np.where(train, "train", "test")):
        raise ValueError("Unexpected split labels")
    if not np.isfinite(frame[["current_A", "current_nA", "y"]].to_numpy()).all():
        raise ValueError("Nonfinite synthetic input")
    if not np.array_equal(frame["current_nA"], frame["current_A"] * 1e9):
        raise ValueError("Unit representations must contain identical information")
    if set(frame.loc[train, "group"]) & set(frame.loc[test, "group"]):
        raise ValueError("Group overlap")
    return train, test


def build_model(family: str, arm: str, X_train: pd.DataFrame):
    params = PARAMETERS[family].copy()
    if arm == "calculator_scaled":
        return make_pipeline(X_train, Candidate(family, family, params), SEED)
    if family == "dummy":
        return DummyRegressor(**params)
    if family == "ridge":
        return Ridge(**params, solver="lsqr")
    return ExtraTreesRegressor(**params, random_state=SEED, n_jobs=1)


def train_predict(frame: pd.DataFrame, *, factory=build_model) -> tuple[dict, dict, int]:
    """Finish every prediction before score_predictions accesses evaluation targets."""
    train, test = validate_fixture(frame)
    predictions, diagnostics = {}, {}
    fit_count = 0
    for family in FAMILIES:
        for arm in ARMS:
            for unit in UNITS:
                name = f"{family}__{arm}__{unit}"
                X = pd.DataFrame({"current": frame[f"current_{unit}"].to_numpy()})
                X_train, X_test = X.loc[train], X.loc[test]
                model = factory(family, arm, X_train)
                model.fit(X_train, frame.loc[train, "y"].to_numpy())
                fit_count += 1
                prediction = np.asarray(model.predict(X_test), dtype=float)
                if prediction.shape != (128,) or not np.isfinite(prediction).all():
                    raise ValueError("Unexpected or nonfinite prediction")
                predictions[name] = prediction
                fitted = model
                detail = {}
                if arm == "calculator_scaled":
                    numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
                    scaler = numeric.named_steps["scale"]
                    if not np.allclose(scaler.mean_, X_train.mean().to_numpy(), rtol=1e-14, atol=0):
                        raise ValueError("Scaler statistics do not match training features")
                    if int(scaler.n_samples_seen_) != 256:
                        raise ValueError("Scaler saw an unexpected number of samples")
                    detail.update(scaler_mean=scaler.mean_.tolist(),
                                  scaler_scale=scaler.scale_.tolist(),
                                  scaler_n=int(scaler.n_samples_seen_))
                    fitted = model.named_steps["model"]
                detail["estimator_parameters"] = fitted.get_params(deep=False)
                if family == "extra_trees":
                    detail["tree_node_counts"] = [int(tree.tree_.node_count) for tree in fitted.estimators_]
                diagnostics[name] = detail
    if fit_count != 12:
        raise ValueError("Unexpected fit schedule")
    return predictions, diagnostics, fit_count


def score_predictions(frame: pd.DataFrame, predictions: dict, diagnostics: dict) -> dict:
    train, test = validate_fixture(frame)
    expected = {f"{family}__{arm}__{unit}" for family in FAMILIES for arm in ARMS for unit in UNITS}
    if set(predictions) != expected or set(diagnostics) != expected:
        raise ValueError("All twelve prediction arrays are required before scoring")
    y_train = frame.loc[train, "y"].to_numpy()
    y_test = frame.loc[test, "y"].to_numpy()
    groups = frame.loc[test, "group"].to_numpy()
    threshold = 1e-10 * max(1.0, float(np.max(np.abs(y_train))))
    scores, pairs = {}, {}
    for name, values in predictions.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (128,) or not np.isfinite(values).all():
            raise ValueError("Unexpected or nonfinite predictions for scoring")
        errors = np.abs(values - y_test)
        per_group = {str(g): float(errors[groups == g].mean()) for g in TEST_GROUPS}
        group_mae = float(np.mean(list(per_group.values())))
        row_mae = float(errors.mean())
        if not np.isclose(group_mae, row_mae, atol=1e-14, rtol=1e-14):
            raise ValueError("Equal-size group and row means disagree")
        scores[name] = {"group_mae": group_mae, "row_mae": row_mae, "per_group_mae": per_group}
    for family in FAMILIES:
        for arm in ARMS:
            delta = np.abs(predictions[f"{family}__{arm}__A"] - predictions[f"{family}__{arm}__nA"])
            pairs[f"{family}__{arm}"] = {
                "max_prediction_gap": float(delta.max()),
                "mean_prediction_gap": float(delta.mean()),
                "equivalent_at_fixed_threshold": bool(delta.max() <= threshold),
            }
    nodes = diagnostics["extra_trees__raw__A"]["tree_node_counts"]
    raw_check = bool(pairs["extra_trees__raw"]["max_prediction_gap"] > 1e-6
                     and len(nodes) == 64 and set(nodes) == {1})
    scaled_check = all(pairs[f"{f}__calculator_scaled"]["equivalent_at_fixed_threshold"] for f in FAMILIES)
    return {"scores": scores, "unit_pairs": pairs, "prediction_equivalence_threshold": threshold,
            "checks": {"raw_tree_gap_and_stumps": raw_check, "all_scaled_pairs_equivalent": scaled_check}}


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(output: Path) -> int:
    """Create a separate result directory; never overwrite or silently repeat."""
    if os.path.lexists(output):
        raise ValueError("Output already exists; choose a new result directory")
    if sklearn.__version__ != REFERENCE_SKLEARN:
        raise ValueError("This fixed reference example requires scikit-learn 1.8.0; the package supports a broader range")
    source_hash = hashlib.sha256(Path(product_model.__file__).read_bytes()).hexdigest()
    if source_hash != MODEL_SOURCE_SHA256:
        raise ValueError("The pipeline source differs from this version-pinned example; do not interpret this as a fresh validation")
    # Refuse nonexistent parents before generation or fitting.
    output.mkdir(mode=0o700)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    cpu_limit = None
    try:
        if sys.platform.startswith("linux"):
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
            cpu_limit = 600
        write_json(output / "run-started.json", {"scope": "EXPOSED_SYNTHETIC_SOFTWARE_EXAMPLE",
                   "planned_fits": 12, "completion_requires": "manifest.json"})
        frame = generate()
        validate_fixture(frame)
        write_fixture(frame, output / "synthetic_fixture.csv")
        with threadpool_limits(limits=1):
            pools = [{k: p[k] for k in ("internal_api", "num_threads", "version") if k in p}
                     for p in threadpool_info()]
            predictions, diagnostics, fit_count = train_predict(frame)
        # No evaluation-target scoring occurs above this line.
        summary = score_predictions(frame, predictions, diagnostics)
        summary.update(
            scope="EXPOSED_SYNTHETIC_SOFTWARE_EXAMPLE", seed=SEED, fit_count=fit_count,
            unit_factor=1e9, target_units="arbitrary_response_units_unchanged_between_arms",
            split={"train_groups": list(TRAIN_GROUPS), "test_groups": list(TEST_GROUPS),
                   "train_rows": 256, "test_rows": 128},
            preprocessing_diagnostics=diagnostics, parameters=PARAMETERS,
            source={"repository": "xinhi6716/ml-super-calculator", "reference_revision": PUBLIC_SOURCE,
                    "model_py_sha256": source_hash,
                    "package_py_sha256": {str(p.relative_to(Path(product_model.__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(Path(product_model.__file__).parent.rglob("*.py"))}},
            versions={"python": platform.python_version(), "numpy": np.__version__,
                      "pandas": pd.__version__, "sklearn": sklearn.__version__,
                      "scipy": distribution_version("scipy"), "threadpoolctl": distribution_version("threadpoolctl"),
                      "platform": platform.system()},
            threadpools=pools, dataset_download_bytes=0,
        )
        summary["status"] = "CHECKS_MET" if all(summary["checks"].values()) else "CHECKS_NOT_MET"
        write_json(output / "summary.json", summary)
        _, test = validate_fixture(frame)
        prediction_frame = frame.loc[test, ["row_id", "group", "y"]].copy()
        for name, values in predictions.items():
            prediction_frame[name] = values
        with (output / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
            prediction_frame.to_csv(stream, index=False, float_format="%.17g")
        write_json(output / "cost.json", {
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "cpu_limit_seconds": cpu_limit,
            "timing_scope": "generation through prediction/summary writes; excludes imports, preflight and manifest",
            "thread_limit": 1, "dataset_download_bytes": 0,
        })
        manifest = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                    for p in sorted(output.iterdir()) if p.is_file()}
        write_json(output / "manifest.json", {"files": manifest, "status": "COMPLETE", "scope": "synthetic example outputs only"})
        return 0 if summary["status"] == "CHECKS_MET" else 1
    except Exception as error:
        # Preserve incomplete output for diagnosis; never label it a complete report.
        write_json(output / "failure.json", {"status": "INCOMPLETE", "error_type": type(error).__name__})
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New result directory with an existing parent")
    args = parser.parse_args(argv)
    try:
        code = run(args.output)
    except (OSError, ValueError) as error:
        print(f"Example not completed: {error}", file=sys.stderr)
        return 2
    print("Synthetic example checks met." if code == 0 else "Fixed checks not met; outputs retained, do not retune this fixture.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
