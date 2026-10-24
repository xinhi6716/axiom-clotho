"""Explicit categorical protocols; legacy defaults remain reproducible."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

LEGACY = "legacy-v1"
DISTINCT = "distinct-v2"
ENCODINGS = (LEGACY, DISTINCT)
MISSING_MARKER = "__GROUPBENCH_MISSING__"


def validate_categorical_encoding(value):
    if not isinstance(value, str) or value not in ENCODINGS:
        raise ValueError("categorical_encoding must be 'legacy-v1' or 'distinct-v2'.")
    return value


def encoding_options(value):
    """Omit the legacy option to retain existing factories/config records."""
    return {} if value == LEGACY else {"categorical_encoding": value}


def _distinct_category_keys(X):
    """Escape the entire marker-prefix domain, never just the exact marker.

    Inserting ':' after the common prefix is injective and preserves the
    ordering of all real source strings. Escaped strings cannot collide with
    unchanged strings (which do not start with the prefix), or the bare marker
    reserved for missing. Missing keeps the legacy lexical sort boundary.
    This stateless step learns nothing from training or prediction batches.
    """
    values = np.asarray(X, dtype=object)
    result = np.empty(values.shape, dtype=object)
    for index, value in np.ndenumerate(values):
        if pd.isna(value):
            result[index] = MISSING_MARKER
        elif isinstance(value, str):
            result[index] = (MISSING_MARKER + ":" + value[len(MISSING_MARKER):]
                             if value.startswith(MISSING_MARKER) else value)
        else:
            raise ValueError("Categorical encoding requires normalized strings or missing values.")
    return result


def categorical_imputer(encoding):
    validate_categorical_encoding(encoding)
    if encoding == DISTINCT:
        return FunctionTransformer(_distinct_category_keys, feature_names_out="one-to-one")
    return SimpleImputer(strategy="constant", fill_value=MISSING_MARKER,
                         keep_empty_features=True)


def preprocessing_description():
    return ("Within each fit: numeric median imputation + scaling; categorical "
            "distinct-v2 stateless marker-prefix escaping + train-only sparse "
            "one-hot encoding (max 64 categories per feature). Unknown and "
            "infrequent category compression still applies.")
