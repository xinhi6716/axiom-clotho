"""Compile synthetic golden tests against the existing Python support oracle.

The default Python workflow does not require Rust. This development check needs
an installed rustc and existing Python dependencies; it installs/downloads nothing. Rust
assertions compare every field, with no floating-point comparison tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import struct
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from groupbench.data import validate_features
from groupbench.input_support import prediction_input_support


def _rust_string(value):
    """Encode a Rust string literal without inferring or repairing its content."""
    if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError("Unpaired surrogate cannot be represented in a Rust UTF-8 string literal.")
    escapes = {'"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t'}
    return '"' + ''.join(escapes.get(c, f'\\u{{{ord(c):x}}}' if ord(c) < 32 or ord(c) == 127 else c)
                         for c in value) + '"'


def _columns(frame):
    columns = []
    for name in frame:
        numeric = pd.api.types.is_numeric_dtype(frame[name])
        values = []
        for value in frame[name]:
            if pd.isna(value):
                values.append("None")
            elif numeric:
                values.append(f"Some({float(value)!r}_f64)")
            else:
                values.append(f"Some(String::from({_rust_string(value)}))")
        variant = "Numeric" if numeric else "Categorical"
        columns.append(f"FeatureColumn {{ name: String::from({_rust_string(name)}), "
                       f"values: FeatureValues::{variant}(vec![{','.join(values)}]) }}")
    return 'vec![' + ','.join(columns) + ']'


def _option(value):
    return "None" if value is None else f"Some({value})"


def _require_keys(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Unsupported Python oracle {context} keys; review the parity contract explicitly.")


def _expected(report):
    _require_keys(report, {
        "schema_version", "reference", "action", "n_training_rows", "n_prediction_rows",
        "rows_with_any_flag", "row_flag_counts", "features",
    }, "report")
    _require_keys(report["row_flag_counts"], {
        "missing_value", "numeric_outside_training_range", "unseen_category",
        "non_missing_value_in_all_missing_training_feature",
    }, "row flag")
    features = []
    for feature in report["features"]:
        kind = feature.get("kind")
        if kind not in ("numeric", "categorical"):
            raise ValueError("Unsupported Python oracle feature kind; review the parity contract explicitly.")
        keys = {"name", "kind", "training_missing_count", "training_support",
                "prediction_missing_count", "prediction_non_missing_without_training_values_count"}
        keys.update({"below_training_min_count", "above_training_max_count"} if kind == "numeric"
                    else {"unseen_category_count"})
        _require_keys(feature, keys, "feature")
        support = {"observed_non_missing_values": "ObservedNonMissingValues",
                   "unavailable_all_missing": "UnavailableAllMissing"}.get(feature["training_support"])
        if support is None:
            raise ValueError("Unsupported Python oracle training support; review the parity contract explicitly.")
        if kind == "numeric":
            counts = ("FeatureCounts::Numeric { below_training_min_count: "
                      + _option(feature["below_training_min_count"])
                      + ", above_training_max_count: "
                      + _option(feature["above_training_max_count"]) + " }")
        else:
            counts = ("FeatureCounts::Categorical { unseen_category_count: "
                      + str(feature["unseen_category_count"]) + " }")
        fields = [f"name: String::from({_rust_string(feature['name'])})",
                  f"training_support: TrainingSupport::{support}", f"counts: {counts}"]
        for key in ("training_missing_count", "prediction_missing_count",
                    "prediction_non_missing_without_training_values_count"):
            fields.append(f"{key}: {feature[key]}")
        features.append("FeatureSupport { " + ','.join(fields) + " }")
    fields = [f"{key}: {_rust_string(report[key])}" for key in ("schema_version", "reference", "action")]
    fields += [f"{key}: {report[key]}" for key in ("n_training_rows", "n_prediction_rows", "rows_with_any_flag")]
    flags = ','.join(f"{key}: {value}" for key, value in report["row_flag_counts"].items())
    fields += [f"row_flag_counts: RowFlagCounts {{ {flags} }}", "features: vec![" + ','.join(features) + "]"]
    return "InputSupport { " + ','.join(fields) + " }"


def golden_cases():
    """Invented values only. Cases are deterministic and need no external data."""
    yield "numeric_boundaries", pd.DataFrame({"x": [1., 3., np.nan], "constant": [2.] * 3}), pd.DataFrame({
        "constant": [2., 1., 3., 2., np.nan], "x": [1., 3., 0., 4., np.nan]})
    yield "all_missing_training", pd.DataFrame({"x": [np.nan] * 2, "kind": pd.Series([None] * 2, dtype=object)}), pd.DataFrame({
        "x": [1., np.nan, -1.], "kind": ["new", None, "__GROUPBENCH_MISSING__"]})
    yield "missing_marker_literal", pd.DataFrame({"kind": ["A", None, "__GROUPBENCH_MISSING__"]}), pd.DataFrame({
        "kind": ["__GROUPBENCH_MISSING__", None, "A", "never-seen", ""]})
    yield "stateless_normalization", pd.DataFrame({"kind": pd.Series([True, 7, None], dtype=object),
                                                   "flag": [True, False, True]}), pd.DataFrame({
        "flag": ["True", "False", "unknown"], "kind": ["True", "7", "unseen"]})
    yield "extreme_binary64", pd.DataFrame({"x": [-np.finfo(float).max, np.finfo(float).max],
                                           "constant": [np.finfo(float).max] * 2}), pd.DataFrame({
        "x": [-np.finfo(float).max, 0., np.finfo(float).max],
        "constant": [-np.finfo(float).max, 0., np.finfo(float).max]})
    yield "signed_zero_subnormal", pd.DataFrame({"x": [-0., np.nextafter(0., 1.)]}), pd.DataFrame({
        "x": [0., -0., np.nextafter(0., 1.), np.nextafter(0., -1.)]})
    yield "text_identity", pd.DataFrame({'製程\t"\\': ["材料", "", "A ", "line\nbreak"], "": [1., 1., 2., 2.]}), pd.DataFrame({
        "": [0., 1., 2., 3.], '製程\t"\\': ["材料", "", "A", "line\nbreak"]})
    yield "all_missing_type_accommodation", pd.DataFrame({"x": [1.], "kind": ["A"]}), pd.DataFrame({
        "kind": [np.nan, np.nan], "x": pd.Series([None, None], dtype=object)})
    yield "duplicate_indexes", pd.DataFrame({"x": [1., 2.], "kind": ["A", "B"]}, index=[99, 99]), pd.DataFrame({
        "kind": ["unseen", "B"], "x": [3., 2.]}, index=[-7, -7])
    yield "large_inventory", pd.DataFrame({"kind": [f"private-category-{i}" for i in range(20_000)]}), pd.DataFrame({
        "kind": ["private-category-0", "private-category-19999", "new-secret"]})
    yield "feature_limit", pd.DataFrame({f"x{i}": [0., np.nan] for i in range(100)}), pd.DataFrame({
        f"x{i}": [0., 1., np.nan] for i in reversed(range(100))})
    rng = random.Random(20261104)
    for case in range(32):
        train, query = {}, {}
        rows, new_rows = rng.randrange(1, 25), rng.randrange(1, 25)
        for column in range(rng.randrange(1, 9)):
            name = f"feature_{column}"
            if rng.randrange(2):
                train[name] = pd.Series([rng.choice([None, -2., 0., 1., 3.]) for _ in range(rows)], dtype=float)
                query[name] = pd.Series([rng.choice([None, -3., -2., 0., 3., 4.]) for _ in range(new_rows)], dtype=float)
            else:
                train[name] = pd.Series([rng.choice([None, "A", "B", "__GROUPBENCH_MISSING__", "材料"]) for _ in range(rows)], dtype=object)
                query[name] = pd.Series([rng.choice([None, "A", "B", "new", "", "材料"]) for _ in range(new_rows)], dtype=object)
        yield f"random_{case:02d}", pd.DataFrame(train), pd.DataFrame(query)[list(reversed(query))]


def generated_tests():
    tests = ["use clotho_core::input_support::*;\n"]
    cases = []
    for name, training, incoming in golden_cases():
        source, query = validate_features(training), validate_features(incoming)
        report = prediction_input_support(source, incoming)
        # Preserve incoming order and all-missing dtype until the Rust boundary.
        tests.append(f"#[test]\nfn {name}() {{\n"
                     f"let training = {_columns(source)};\nlet incoming = {_columns(query)};\n"
                     f"let expected = {_expected(report)};\n"
                     "assert_eq!(prediction_input_support(&training, &incoming).unwrap(), expected);\n}\n")
        cases.append(name)
    return '\n'.join(tests), cases


def numeric_byte_cases():
    """Independent byte-boundary cases; the original 43 cases stay unchanged."""
    for name, training, incoming in golden_cases():
        source, query = validate_features(training), validate_features(incoming)
        if all(pd.api.types.is_numeric_dtype(source[c]) and
               pd.api.types.is_numeric_dtype(query[c]) for c in source):
            yield "bytes_" + name, source, query
    yield "bytes_all_missing", pd.DataFrame({"x": [np.nan, np.nan]}), pd.DataFrame({
        "x": [np.nan, 0., -1., np.nan]})
    yield "bytes_all_missing_query", pd.DataFrame({"x": [0., 1.]}), pd.DataFrame({"x": [np.nan] * 3})
    yield "bytes_row_limit", pd.DataFrame({"x": np.arange(20_000, dtype=float)}), pd.DataFrame({
        "x": np.full(20_000, np.nan)})
    yield "bytes_nan_payloads", pd.DataFrame({"x": [np.nan]}), pd.DataFrame({"x": [
        struct.unpack("<d", struct.pack("<Q", bits))[0]
        for bits in (0x7ff0000000000001, 0xfff8000000000001, 0x7ff8000000000042)]})
    rng = random.Random(20261105)
    def values(n):
        # Force one exponent bit off: finite binary64, retaining both signs.
        return [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64) & 0xffefffffffffffff))[0]
                for _ in range(n)]
    for case in range(16):
        training = {f"n{i}": values(31) for i in range(4)}
        incoming = {f"n{i}": values(43) for i in reversed(range(4))}
        yield f"bytes_random_bits_{case:02d}", pd.DataFrame(training), pd.DataFrame(incoming)


def _byte_storage(frame):
    columns = []
    for name in frame:
        # Explicit binary64 little-endian storage, including exact NaN payloads.
        bits = [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in frame[name]]
        columns.append("[" + ",".join(f"{v}_u64" for v in bits)
                       + "].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()")
    return "vec![" + ",".join(columns) + "]"


def generated_numeric_byte_tests():
    tests = ["use clotho_core::numeric_bytes::*;\n"]
    cases = []
    for name, training, incoming in numeric_byte_cases():
        source, query = validate_features(training), validate_features(incoming)
        report = prediction_input_support(source, incoming)
        def columns(frame, storage):
            return "vec![" + ",".join(
                f"NumericBytesColumn {{ name: {_rust_string(n)}, values: &{storage}[{i}] }}"
                for i, n in enumerate(frame)) + "]"
        tests.append(f"#[test]\nfn {name}() {{\n"
                     f"let t = {_byte_storage(source)};\nlet q = {_byte_storage(query)};\n"
                     f"let training = {columns(source, 't')};\nlet incoming = {columns(query, 'q')};\n"
                     f"let expected = {_expected(report)};\n"
                     "assert_eq!(prediction_input_support_bytes(&training, &incoming).unwrap(), expected);\n}\n")
        cases.append(name)
    return '\n'.join(tests), cases


def check_parity(rustc=None):
    compiler = rustc or os.environ.get("RUSTC") or shutil.which("rustc")
    if not compiler:
        raise RuntimeError("rustc is required for Rust parity; no Rust checks were run.")
    source, cases = generated_tests()
    byte_source, byte_cases = generated_numeric_byte_tests()
    source += "\n" + byte_source
    with tempfile.TemporaryDirectory(prefix="clotho-rust-parity-") as tmp:
        directory = Path(tmp)
        library = directory / "libclotho_core.rlib"
        generated = directory / "parity.rs"
        binary = directory / ("parity.exe" if os.name == "nt" else "parity")
        generated.write_text(source, encoding="utf-8")
        commands = [
            [compiler, "--edition=2021", "--crate-name=clotho_core", "--crate-type=rlib", "-D", "warnings",
             str(ROOT / "rust/clotho-core/src/lib.rs"), "-o", str(library)],
            [compiler, "--edition=2021", "--test", "-D", "warnings", str(generated),
             "--extern", f"clotho_core={library}", "-o", str(binary)],
            [str(binary)],
        ]
        for command in commands:
            subprocess.run(command, check=True, timeout=180)
    return {"status": "pass", "golden_cases": len(cases), "comparison": "all_report_fields_exact",
            "source": "groupbench.input_support.prediction_input_support", "cases": cases,
            "numeric_byte_cases": len(byte_cases), "numeric_byte_case_names": byte_cases,
            "total_cases": len(cases) + len(byte_cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rustc", help="Path to an already installed rustc executable.")
    args = parser.parse_args()
    try:
        result = check_parity(args.rustc)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Rust parity failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
