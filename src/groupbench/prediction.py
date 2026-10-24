"""Fit an inner-selected local model and predict new rows, with provenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .categorical import LEGACY, DISTINCT, encoding_options, preprocessing_description
from .data import (data_fingerprint, duplicate_diagnostics, prepare_data,
                   validate_features, validate_prediction_features)
from .model import GroupBenchRegressor, selection_plan
from .input_support import prediction_input_support


@dataclass
class PredictionResult:
    predictions: pd.DataFrame
    manifest: dict
    estimator: GroupBenchRegressor

    def write(self, output_dir, *, overwrite: bool = False):
        ensure_prediction_output(output_dir)
        from .report import write_payloads
        payloads = {
            "predictions.csv": self.predictions.to_csv(index=False, lineterminator="\n"),
            "manifest.json": json.dumps(self.manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        }
        write_payloads(payloads, output_dir, overwrite=overwrite)
        from pathlib import Path
        return Path(output_dir) / "predictions.csv"


def ensure_prediction_output(output_dir):
    """Do not leave an old evaluation report beside a new inference bundle."""
    from pathlib import Path
    out = Path(output_dir)
    if any((out / name).exists() or (out / name).is_symlink() for name in ("report.html", "summary.json")):
        raise ValueError("Prediction output contains evaluation artifacts. Choose a separate output directory, even with overwrite enabled.")


def _prepare_prediction(frame, new_rows, *, target, group, features, inner_splits,
                        random_state, baselines, objective, categorical_encoding=LEGACY):
    """Shared preflight of training, new-row schema and full-data fit schedule."""
    X, y, groups = prepare_data(frame, target, group, features)
    incoming = validate_features(new_rows)
    if set(incoming.columns) != set(X.columns):
        raise ValueError("New rows must contain exactly the training feature columns, without target or group columns.")
    # No imputation/encoding is fitted; match the estimator's prediction type
    # rules, including its all-missing-column accommodation, before training.
    validate_prediction_features(incoming, X.columns, [str(X[c].dtype) for c in X])
    plan = selection_plan(X, groups, baselines=baselines, inner_splits=inner_splits,
                          random_state=random_state, objective=objective, categorical_encoding=categorical_encoding)
    return X, y, groups, incoming, plan


def fit_predict(frame: pd.DataFrame, new_rows: pd.DataFrame, *, target: str, group: str,
                features=None, inner_splits: int = 3, random_state: int = 0,
                baselines=("dummy", "ridge", "extra_trees"), objective="group_mae",
                source_sha256: str | None = None, prediction_source_sha256: str | None = None,
                categorical_encoding: str = LEGACY) -> PredictionResult:
    """Fit using all labeled rows and return local predictions for new rows.

    This is inference, not an independent performance estimate. ``new_rows``
    must contain exactly the selected feature columns, never a target or group
    column. Use ``benchmark`` on training data before trusting this prediction.
    No executable model file is loaded or saved.
    """
    X, y, groups, incoming, plan = _prepare_prediction(
        frame, new_rows, target=target, group=group, features=features,
        inner_splits=inner_splits, random_state=random_state, baselines=baselines, objective=objective, categorical_encoding=categorical_encoding)
    random_state = plan.seed
    estimator = GroupBenchRegressor(baselines=baselines, inner_splits=inner_splits,
                                    random_state=random_state, objective=objective, categorical_encoding=categorical_encoding)
    estimator._fit_prepared(X, y, groups, plan)
    inner_splits = len(plan.splits)
    predictions = estimator.predict(incoming)
    new_hash = hashlib.sha256(pd.util.hash_pandas_object(incoming[list(X.columns)], index=False).values.tobytes()).hexdigest()
    manifest = {
        "schema_version": "1.1", "groupbench_version": __version__, "mode": "fit_predict",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_sha256": data_fingerprint(X, y, groups), "new_features_sha256": new_hash,
        "source_file_sha256": source_sha256, "prediction_source_file_sha256": prediction_source_sha256,
        "n_training_rows": len(X), "n_training_groups": len(pd.unique(groups)), "n_predictions": len(predictions),
        "feature_schema": [{"name": c, "dtype": str(X[c].dtype)} for c in X],
        "config": {"target": target, "group": group, "features": list(X.columns), "inner_splits": inner_splits,
                   "random_state": random_state, "baselines": list(baselines), "selection_objective": objective, "threads": 1},
        "selected_candidate": estimator.best_candidate_, "selected_family": estimator.best_family_,
        "inner_selection": estimator.selection_results_, "inner_folds": estimator.inner_audit_,
        "data_diagnostics": duplicate_diagnostics(X, y, groups),
        "input_support": prediction_input_support(X, incoming),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "limitations": [
            "These are new-row predictions from a full-data refit, not held-out evaluation metrics.",
            "Inner validation scores are selection diagnostics, not unbiased test-set performance.",
            "No uncertainty intervals or extrapolation guarantees are provided.",
            "Input-support flags describe individual source features only, not uncertainty, calibration, multivariate support or prediction reliability; no flags is not evidence of safety.",
            "No serialized executable model is saved; repeat fit_predict to refit or retain the in-memory estimator.",
            "Manifest contains training group identifiers; review before sharing.",
        ],
    }
    if categorical_encoding == DISTINCT:
        manifest.update(schema_version="1.2", protocol="group-fit-predict-distinct-v2",
                        preprocessing=preprocessing_description())
        manifest["config"].update(encoding_options(categorical_encoding))
    return PredictionResult(pd.DataFrame({"row_position": np.arange(len(predictions)), "prediction": predictions}), manifest, estimator)
