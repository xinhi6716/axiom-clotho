"""Aggregate source-feature checks, separate from fitted models and transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import validate_prediction_features


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
