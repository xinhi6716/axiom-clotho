"""Read an offline, bounded feature-schema declaration, never executable data.

This accepts the version-1 schema emitted by Semiconductor Toolbox's original
measurement-to-features example. It selects names, not scientific validity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

from .data import MAX_FEATURES


MAX_MANIFEST_BYTES = 64 * 1024
SCHEMA_KEYS = frozenset({"schema_version", "row_key", "model_feature_allowlist",
                         "metadata_columns", "target_columns", "feature_units", "warnings"})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Feature schema contains duplicate JSON object keys.")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("Feature schema must contain strict JSON without non-finite constants.")


def _names(value, field, *, nonempty=False):
    if (not isinstance(value, list) or len(value) > MAX_FEATURES
            or (nonempty and not value)
            or any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in value)):
        raise ValueError(f"Feature schema {field} must be a list of at most {MAX_FEATURES} non-blank, unpadded names"
                         + (" with at least one entry." if nonempty else "."))
    if len(value) != len(set(value)):
        raise ValueError(f"Feature schema {field} contains duplicate names.")
    return value


def _read_regular(path):
    """Follow local input symlinks, like CSV input; never read a special file.

    Check before opening to avoid known devices/FIFOs, then check the descriptor
    again. Nonblocking open avoids waiting on a FIFO swapped in before open on
    platforms providing O_NONBLOCK. This is not a filesystem-atomic snapshot or
    a guarantee about network-mounted storage; use trusted, idle local files.
    """
    path = Path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Feature schema input must be a regular local file.")
    if before.st_size > MAX_MANIFEST_BYTES:
        raise ValueError(f"Feature schema exceeds the {MAX_MANIFEST_BYTES}-byte limit.")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                 | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0))
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("Feature schema input must be a regular local file.")
        if opened.st_size > MAX_MANIFEST_BYTES:
            raise ValueError(f"Feature schema exceeds the {MAX_MANIFEST_BYTES}-byte limit.")
        chunks, size = [], 0
        while True:
            chunk = os.read(fd, min(8192, MAX_MANIFEST_BYTES - size + 1))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_MANIFEST_BYTES:
                raise ValueError(f"Feature schema exceeds the {MAX_MANIFEST_BYTES}-byte limit.")
    finally:
        os.close(fd)
    return b"".join(chunks)


def read_feature_manifest(path):
    """Return only the declared predictor names, in their original order.

    No referenced files, URLs, imports, executable formats, or models are loaded.
    The caller must still validate names against the actual training table and
    exclude its target and grouping columns through the usual prepare_data path.
    """
    raw = _read_regular(path)
    try:
        schema = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Feature schema must contain a bounded, valid UTF-8 JSON object.") from exc
    if not isinstance(schema, dict) or set(schema) != SCHEMA_KEYS:
        raise ValueError("Feature schema must contain exactly the documented version-1 schema fields.")
    if type(schema["schema_version"]) is not int or schema["schema_version"] != 1:
        raise ValueError("Feature schema schema_version must be the integer 1.")
    features = _names(schema["model_feature_allowlist"], "model_feature_allowlist", nonempty=True)
    row_key = _names(schema["row_key"], "row_key", nonempty=True)
    metadata = _names(schema["metadata_columns"], "metadata_columns")
    targets = _names(schema["target_columns"], "target_columns")
    if (set(features) & (set(metadata) | set(targets) | set(row_key))
            or set(metadata) & set(targets) or not set(row_key) <= set(metadata)):
        raise ValueError("Feature schema roles conflict: predictors, metadata and targets must be disjoint; row_key must be metadata.")
    units = schema["feature_units"]
    if (not isinstance(units, dict) or set(units) != set(features)
            or any(not isinstance(unit, str) or not unit.strip() for unit in units.values())):
        raise ValueError("Feature schema feature_units must declare one non-blank string for each allowed feature only.")
    warnings = schema["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(item, str) or not item.strip() for item in warnings):
        raise ValueError("Feature schema warnings must be a list of non-blank strings.")
    return list(features)
