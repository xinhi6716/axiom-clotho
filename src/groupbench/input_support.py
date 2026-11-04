"""Aggregate source-feature checks, separate from fitted models and transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import validate_prediction_features, _align_normalized_prediction_features


# Explicit handshake for the separate optional numeric-only extension package.
# Defining the contract does not import, build, or select a native backend.
NUMERIC_SUPPORT_PROTOCOL = "numeric-source-support-v1"


def prediction_input_support(training: pd.DataFrame, new_rows: pd.DataFrame) -> dict:
    """Compare new rows with validated, normalized training features only.

    Callers supply training features from ``prepare_data``. Apply the existing
    prediction validation to reorder columns and accommodate all-missing input
    columns, without mutating either frame. No target/group, estimator state,
    imputed values or prediction-batch statistics define the reference.

    Work is bounded by the existing row/feature limits. Only per-feature counts
    and aggregate row counts leave this function, never category inventories,
    numeric extrema, row identifiers or feature values. Counts may overlap.
    """
    incoming = validate_prediction_features(
        new_rows, training.columns, [str(training[c].dtype) for c in training])
    return _input_support_aligned(training, incoming)


def _prediction_input_support_normalized(training, new_rows):
    """Internal fast path for unchanged ``validate_features`` output only.

    Public diagnostics always validate raw inputs above. Planning and inference
    already own normalized copies; repeat only schema/type alignment here.
    """
    incoming = _align_normalized_prediction_features(
        new_rows, training.columns, [str(training[c].dtype) for c in training])
    return _input_support_aligned(training, incoming)


def _input_support_aligned(training, incoming):
    """Count support against normalized training and aligned prediction copies."""
    flags = {name: np.zeros(len(incoming), dtype=bool) for name in (
        "missing_value", "numeric_outside_training_range", "unseen_category",
        "non_missing_value_in_all_missing_training_feature")}
    features = []
    for name in training:
        source, query = training[name], incoming[name]
        source_missing, query_missing = source.isna(), query.isna().to_numpy()
        observed = source[~source_missing]
        all_missing = len(observed) == 0
        unsupported = ~query_missing if all_missing else np.zeros(len(incoming), dtype=bool)
        flags["missing_value"] |= query_missing
        flags["non_missing_value_in_all_missing_training_feature"] |= unsupported
        numeric = pd.api.types.is_numeric_dtype(source)
        feature = {
            "name": name, "kind": "numeric" if numeric else "categorical",
            "training_missing_count": int(source_missing.sum()),
            "training_support": "unavailable_all_missing" if all_missing else "observed_non_missing_values",
            "prediction_missing_count": int(query_missing.sum()),
            "prediction_non_missing_without_training_values_count": int(unsupported.sum()),
        }
        if numeric:
            # Normalization already rejects infinities. Direct comparisons
            # avoid overflow from subtracting extreme but finite endpoints.
            feature["below_training_min_count"] = None
            feature["above_training_max_count"] = None
            if not all_missing:
                below = query.to_numpy() < observed.min()
                above = query.to_numpy() > observed.max()
                feature["below_training_min_count"] = int(below.sum())
                feature["above_training_max_count"] = int(above.sum())
                flags["numeric_outside_training_range"] |= below | above
        else:
            # Source categories, not imputed/encoded categories: a literal
            # missing marker is a normal string; NaN is counted separately.
            unseen = ~query_missing & ~query.isin(observed.unique()).to_numpy()
            feature["unseen_category_count"] = int(unseen.sum())
            flags["unseen_category"] |= unseen
        features.append(feature)
    any_flag = np.zeros(len(incoming), dtype=bool)
    for mask in flags.values():
        any_flag |= mask
    return {
        "schema_version": "1.0",
        "reference": "full_training_features_before_imputation",
        "action": "diagnostic_only_no_input_changes",
        "n_training_rows": len(training), "n_prediction_rows": len(incoming),
        "rows_with_any_flag": int(any_flag.sum()),
        "row_flag_counts": {name: int(mask.sum()) for name, mask in flags.items()},
        "features": features,
    }


def outer_fold_input_support(X: pd.DataFrame, groups, *, outer_splits: int = 5,
                             random_state: int = 0) -> dict:
    """Return optional target-free support diagnostics for benchmark outer folds.

    Pass exactly the selected feature frame, row order and group labels used by
    the benchmark. This uses its GroupKFold splitter, but does not validate inner
    selection feasibility or fit models. Marginal ranges do not detect holes,
    joint feature support, target shift, or establish future accuracy.
    """
    from .data import validate_features, validate_groups, validate_seed
    from .splitting import GroupKFold

    features = validate_features(X)
    labels = validate_groups(groups, len(features))
    seed = validate_seed(random_state)
    folds = []
    for fold, (train, test) in enumerate(
            GroupKFold(outer_splits, seed).split(features, groups=labels)):
        support = prediction_input_support(features.iloc[train], features.iloc[test])
        support["reference"] = "outer_training_features_before_imputation"
        folds.append({
            "fold": fold,
            "train_group_count": len(set(labels[train])),
            "test_group_count": len(set(labels[test])),
            "flagged_row_fraction": support["rows_with_any_flag"] / len(test),
            "input_support": support,
        })
    return {
        "schema_version": "1.0",
        "diagnostic": "outer_fold_source_support",
        "action": "diagnostic_only_no_input_changes",
        "splitter": "GroupKFold",
        "random_state": seed,
        "outer_splits": len(folds),
        "n_rows": len(features),
        "n_groups": len(set(labels)),
        "folds": folds,
        "limitations": [
            "Marginal feature checks do not detect interior holes or joint support gaps.",
            "Flags are not an OOD test, calibrated uncertainty, or evidence of future accuracy.",
            "No model selection, predictions, or benchmark outputs are changed.",
            "Inner selection feasibility is not checked by this target-free diagnostic.",
        ],
    }
