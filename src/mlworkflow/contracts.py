"""Bounded, standard-library-only contracts for measurement regression bundles."""
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat

FEATURES = ("hrs_resistance_median_ohm", "lrs_resistance_median_ohm", "resistance_ratio_median")
FEATURE_FILES = ("features.csv", "feature_schema.json", "iv_readpoints.json", "resistance_input.csv", "rram_windows.json", "provenance.json")
INVENTORIES = {
    "prepared": {"training.csv", "future.csv", "cohort.csv", "config.json", "provenance.json"} | {"features/" + n for n in FEATURE_FILES},
    "run": {"run.json", "prediction_rows.csv", "report.html"} | {"comparison/" + n for n in ("summary.json", "manifest.json", "predictions.csv", "report.html", "checksums.json")} | {"prediction/" + n for n in ("manifest.json", "predictions.csv", "checksums.json")},
    "score": {"scores.json", "scored_rows.csv", "report.html"},
}
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 10000
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]{1,4})?\Z")
DECLARATIONS = {"global_physical_ids", "pre_outcome_features", "endpoint_not_derived_from_features", "measurement_assumptions_accepted"}
CONFIG_KEYS = {"schema_version", "endpoint_name", "target_unit", "group_by", "features", "max_bracket_span_V", "outer_splits", "inner_splits", "baselines", "seed", "objective", "max_fits", "declarations"}


class WorkflowError(ValueError):
    """Categorical errors never include user paths, IDs or endpoint values."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_regular(path):
    try:
        before = Path(path).stat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_BYTES:
            raise WorkflowError("input must be a bounded regular local file")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size > MAX_BYTES:
                raise WorkflowError("input must be a bounded regular local file")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                data = handle.read(MAX_BYTES + 1)
        finally:
            os.close(fd)
        if len(data) > MAX_BYTES:
            raise WorkflowError("input exceeds the byte limit")
        return data
    except OSError:
        raise WorkflowError("unable to read a required local input") from None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowError("JSON contains duplicate keys")
        result[key] = value
    return result


def parse_json(raw):
    def reject(_):
        raise WorkflowError("JSON contains a nonfinite value")
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique, parse_constant=reject)
    except (UnicodeError, ValueError, RecursionError):
        raise WorkflowError("input must be strict bounded UTF-8 JSON") from None


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def identifier(value):
    if (not isinstance(value, str) or not value or len(value) > 256 or value != value.strip()
            or any(ord(c) < 32 or ord(c) == 127 or 0xD800 <= ord(c) <= 0xDFFF for c in value) or value[0] in "=+-@"):
        raise WorkflowError("identities and labels must be nonblank, unpadded, control-free and spreadsheet-safe")
    return value


def number(value):
    if not isinstance(value, str) or len(value) > 64 or not NUMBER.fullmatch(value):
        raise WorkflowError("numeric fields require finite decimal strings without units")
    try:
        exact = Decimal(value)
        if len(exact.as_tuple().digits) > 32 or abs(exact.as_tuple().exponent) > 200:
            raise WorkflowError("numeric precision exceeds the supported range; rescale explicitly")
        if exact and not Decimal("1e-100") <= abs(exact) <= Decimal("1e100"):
            raise WorkflowError("numeric magnitude exceeds the supported range; rescale explicitly")
        result = float(exact)
        if not math.isfinite(result):
            raise WorkflowError("numeric fields must be finite")
        return result
    except (InvalidOperation, OverflowError):
        raise WorkflowError("numeric fields exceed the supported range") from None


def parse_csv(raw, columns):
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        header = next(reader, None)
        if header is None or len(header) != len(columns) or set(header) != set(columns):
            raise WorkflowError("CSV must contain exactly the documented unique columns")
        rows = []
        for cells in reader:
            if len(cells) != len(header):
                raise WorkflowError("CSV row width differs from its header")
            rows.append(dict(zip(header, cells)))
            if len(rows) > MAX_ROWS:
                raise WorkflowError("CSV exceeds the row limit")
        if not rows:
            raise WorkflowError("CSV must contain observations")
        return rows
    except (csv.Error, UnicodeError):
        raise WorkflowError("input must be valid UTF-8 CSV") from None


def csv_bytes(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def keyed(rows, key="device"):
    result = {}
    for row in rows:
        value = identifier(row[key])
        if value in result:
            raise WorkflowError("duplicate device identity; no rows were joined or dropped")
        result[value] = row
    return result


def validate_config(config):
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS or config["schema_version"] != "measurement-regression-v1":
        raise WorkflowError("configuration must match measurement-regression-v1 exactly")
    identifier(config["endpoint_name"])
    identifier(config["target_unit"])
    if config["group_by"] not in ("batch", "wafer", "device"):
        raise WorkflowError("choose an explicit batch, wafer or device grouping boundary")
    features = config["features"]
    baselines = config["baselines"]
    if (not isinstance(features, list) or not features or any(not isinstance(f, str) or f not in FEATURES for f in features)
            or len(features) != len(set(features))):
        raise WorkflowError("choose a nonempty unique subset of the measurement feature allowlist")
    if (not isinstance(baselines, list) or "dummy" not in baselines or len(baselines) < 2
            or any(not isinstance(b, str) or b not in ("dummy", "ridge", "extra_trees") for b in baselines)
            or len(baselines) != len(set(baselines))):
        raise WorkflowError("comparison requires dummy and at least one supported unique model family")
    for key, low, high in (("outer_splits", 2, 100), ("inner_splits", 2, 100), ("seed", 0, 2 ** 32 - 101), ("max_fits", 1, 100000)):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise WorkflowError("split count, seed or fit budget is outside the supported integer range")
    if config["objective"] != "group_mae" or number(config["max_bracket_span_V"]) <= 0:
        raise WorkflowError("v1 requires group_mae and a positive declared readpoint bracket bound")
    declarations = config["declarations"]
    if not isinstance(declarations, dict) or set(declarations) != DECLARATIONS or any(v is not True for v in declarations.values()):
        raise WorkflowError("all scientific input declarations must be explicit true booleans; they are not verified facts")
    return config


def check_fresh_output(path):
    path = Path(path)
    if os.path.lexists(path) or not path.parent.is_dir():
        raise WorkflowError("output must be a fresh directory with an existing parent")
    return path


def protect_output(path, *input_roots):
    """Never nest new artifacts inside a protected source/receipt bundle."""
    destination = check_fresh_output(path)
    try:
        resolved = destination.resolve()
        if any(resolved.is_relative_to(Path(root).resolve()) for root in input_roots):
            raise WorkflowError("output must be outside the input measurement or saved-bundle directory; use a sibling destination")
    except (OSError, RuntimeError):
        raise WorkflowError("unable to resolve an input or output directory safely") from None
    return destination


def write_new(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def begin_bundle(path):
    path = check_fresh_output(path)
    path.mkdir(mode=0o700)
    write_new(path / "INCOMPLETE.json", json_bytes({"status": "INCOMPLETE_DO_NOT_USE"}))
    return path


def finish_bundle(path, kind):
    payloads = {name: read_regular(path / name) for name in INVENTORIES[kind]}
    receipt = {"schema_version": "measurement-workflow-bundle-v1", "kind": kind,
               "files_sha256": {name: sha(data) for name, data in sorted(payloads.items())},
               "integrity_not_authenticity": True}
    write_new(path / "COMPLETE.json", json_bytes(receipt))
    (path / "INCOMPLETE.json").unlink()
    return receipt


def write_bundle(path, kind, payloads):
    if set(payloads) != INVENTORIES[kind]:
        raise WorkflowError("internal output inventory mismatch")
    path = begin_bundle(path)
    for name, data in sorted(payloads.items()):
        (path / name).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_new(path / name, data)
    return finish_bundle(path, kind)


def load_bundle(path, kind):
    path = Path(path)
    expected = INVENTORIES[kind]
    try:
        if path.is_symlink() or not path.is_dir():
            raise WorkflowError("bundle must be a regular local directory")
        actual = set()
        for item in path.rglob("*"):
            if item.is_symlink():
                raise WorkflowError("saved bundles cannot contain symbolic links")
            if not item.is_dir():
                # Receipt keys use '/', independently of host path separators.
                actual.add(item.relative_to(path).as_posix())
        if actual != expected | {"COMPLETE.json"}:
            raise WorkflowError("bundle is incomplete or has unexpected files; do not repair it silently")
        receipt_raw = read_regular(path / "COMPLETE.json")
        receipt = parse_json(receipt_raw)
        if (not isinstance(receipt, dict) or set(receipt) != {"schema_version", "kind", "files_sha256", "integrity_not_authenticity"}
                or receipt["schema_version"] != "measurement-workflow-bundle-v1" or receipt["kind"] != kind
                or receipt["integrity_not_authenticity"] is not True or not isinstance(receipt["files_sha256"], dict)
                or set(receipt["files_sha256"]) != expected):
            raise WorkflowError("saved bundle receipt does not match its fixed inventory")
        payloads = {name: read_regular(path / name) for name in expected}
        if any(sha(data) != receipt["files_sha256"][name] for name, data in payloads.items()):
            raise WorkflowError("saved bundle bytes changed; checksum agreement is not authentication")
        return payloads, sha(receipt_raw)
    except OSError:
        raise WorkflowError("unable to read a complete saved bundle") from None
