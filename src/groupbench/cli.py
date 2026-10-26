"""Command-line entry point; inputs and outputs are local files only."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import io
import json
import os
import stat
from pathlib import Path
import sys

import pandas as pd

from . import __version__
from .categorical import ENCODINGS, LEGACY
from .data import MAX_ROWS, make_materials_fixture
from .evaluation import benchmark
from .feature_manifest import read_feature_manifest
from .prediction import ensure_prediction_output, fit_predict
from .planning import plan_benchmark, plan_prediction
from .verification import verify_bundle
from .prediction_schema import validate_prediction_bundle_schema

MAX_CSV_BYTES = 64 * 1024 * 1024
CSV_READ_CHUNK_BYTES = 64 * 1024

OUTPUT_NAMES = ("report.html", "manifest.json", "summary.json", "predictions.csv", "checksums.json")


def _add_protocol(parser, *, outer=True, output=True):
    if output:
        parser.add_argument("--out", type=Path, required=True, help="Local output directory")
    if outer:
        parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baselines", default="dummy,ridge,extra_trees", help="Comma-separated families: dummy,ridge,extra_trees")
    parser.add_argument("--categorical-encoding", choices=ENCODINGS, default=LEGACY,
                        help="Use distinct-v2 to distinguish literal missing-marker strings; default preserves legacy-v1")
    parser.add_argument("--objective", choices=("group_mae", "mae"), default="group_mae")
    if output:
        parser.add_argument("--overwrite", action="store_true", help="Replace known output files in --out")


def _add_columns(parser):
    parser.add_argument("--target", required=True)
    parser.add_argument("--group", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--features", help="Optional comma-separated feature allowlist")
    selection.add_argument("--features-file", type=Path,
                           help="Local version-1 feature_schema.json; use only its model_feature_allowlist")


def protect_input_files(inputs, output_dir):
    """Refuse to replace a source through a direct, symlink, or hardlink alias."""
    for source in inputs:
        for name in OUTPUT_NAMES:
            destination = Path(output_dir) / name
            same_file = destination.exists() and source.exists() and destination.samefile(source)
            if destination.resolve() == source.resolve() or same_file:
                raise ValueError("Output would overwrite an input file; choose a separate --out directory.")


@contextmanager
def _open_csv_source(path):
    """Open one regular input; local symlinks remain allowed.

    Pre-open checks reject known special/oversized inputs. Descriptor checks
    reject special/oversized replacements at open. O_NONBLOCK prevents a swapped FIFO from
    waiting where available; this is not a network-filesystem timeout or lock.
    """
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("CSV input must be a regular local file.")
    if before.st_size > MAX_CSV_BYTES:
        raise ValueError(f"CSV exceeds the {MAX_CSV_BYTES:,}-byte input limit.")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                 | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0))
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("CSV input must be a regular local file.")
        if opened.st_size > MAX_CSV_BYTES:
            raise ValueError(f"CSV exceeds the {MAX_CSV_BYTES:,}-byte input limit.")
        # Transfer descriptor ownership only after fdopen succeeds.
        source = os.fdopen(fd, "rb", buffering=0)
        fd = None
        with source:
            yield source
    finally:
        if fd is not None:
            os.close(fd)


class _HashingReader(io.RawIOBase):
    """Hash only bytes returned to the decoder, without reopening the source."""

    def __init__(self, source, digest):
        super().__init__()
        self.source = source
        self.digest = digest
        self.bytes_read = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        # Read at most one byte beyond the cap to distinguish exact-limit EOF.
        size = min(len(buffer), CSV_READ_CHUNK_BYTES,
                   MAX_CSV_BYTES - self.bytes_read + 1)
        count = self.source.readinto(memoryview(buffer)[:size])
        if count:
            self.bytes_read += count
            if self.bytes_read > MAX_CSV_BYTES:
                raise ValueError(f"CSV exceeds the {MAX_CSV_BYTES:,}-byte input limit.")
            self.digest.update(memoryview(buffer)[:count])
        return count


def _read_csv(path: Path, *, group=None):
    """Keep the existing DataFrame-only interface for CSV input callers."""
    return _read_csv_with_hash(path, group=group)[0]


def _read_csv_with_hash(path: Path, *, group=None):
    """Return validated data and SHA-256 of its exact, single-read byte stream.

    Successful validation consumes the stream to EOF, including the BOM and
    ignored blank lines in the digest. Invalid/oversized input returns no hash.
    This binds provenance to the bytes read, not to a later pathname lookup;
    it does not promise filesystem-atomic reads of concurrently edited files.
    """
    # pandas may silently use excess fields as an index, shift the remaining
    # values under the wrong names, or pad short records. Validate a bounded
    # snapshot before type inference so those repairs cannot change the data.
    snapshot = io.StringIO(newline="")
    digest = hashlib.sha256()
    with _open_csv_source(path) as source, io.TextIOWrapper(
        _HashingReader(source, digest), encoding="utf-8-sig", newline=""
    ) as handle:
        record_lines = []

        def source_lines():
            for physical_line, line in enumerate(handle, start=1):
                # The downstream pandas C parser can silently truncate fields
                # at NUL, merging group IDs or changing features/targets.
                if "\x00" in line:
                    raise ValueError(
                        f"CSV contains an unsupported NUL character at physical line {physical_line}. "
                        "Re-export the file as a clean UTF-8 CSV."
                    )
                record_lines.append(line)
                yield line

        reader = csv.reader(source_lines(), strict=True)
        try:
            header = next(reader, [])
            if not header or len(set(header)) != len(header) or any(not name.strip() for name in header):
                raise ValueError("CSV must have a non-empty header with unique column names; names cannot be blank.")
            snapshot.write("".join(record_lines))
            record_lines.clear()
            n_rows = 0
            for record in reader:
                raw_record = "".join(record_lines)
                record_lines.clear()
                # Match pandas' blank-line behavior without dropping quoted
                # empty/whitespace values or blank lines inside quoted values.
                if not raw_record.strip(" \t\r\n"):
                    continue
                n_rows += 1
                if n_rows > MAX_ROWS:
                    raise ValueError(f"CSV exceeds the MVP limit of {MAX_ROWS:,} rows.")
                if len(record) != len(header):
                    raise ValueError(
                        f"CSV record ending at physical line {reader.line_num} has {len(record)} fields; "
                        f"the header defines {len(header)}. Use explicit empty fields for missing values "
                        "and quote values containing commas."
                    )
                snapshot.write(raw_record)
        except csv.Error as exc:
            raise ValueError(
                f"CSV could not be parsed near physical line {reader.line_num}. "
                f"Check quoting and the CSV field-size limit ({csv.field_size_limit():,} characters)."
            ) from exc
        except UnicodeDecodeError as exc:
            raise ValueError("CSV must use UTF-8 encoding (an optional UTF-8 BOM is supported).") from exc
    # Group labels are identifiers: preserve leading zeros and literal 'NA'.
    converters = {group: str} if group else None
    snapshot.seek(0)
    return pd.read_csv(snapshot, converters=converters), digest.hexdigest()


def _read_training(args):
    """Shared CSV/feature-allowlist handling for execution and no-fit plans."""
    frame, source_hash = _read_csv_with_hash(args.csv_path, group=args.group)
    if args.features_file is not None:
        features = read_feature_manifest(args.features_file)
    else:
        features = [c.strip() for c in args.features.split(",")] if args.features else None
    return frame, features, source_hash


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="groupbench", description="ML Super Calculator: local grouped regression, powered by GroupBench.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Evaluate the wholly synthetic materials-process fixture")
    _add_protocol(demo)
    demo.add_argument("--groups", type=int, default=24)
    demo.add_argument("--rows-per-group", type=int, default=12)
    run = sub.add_parser("run", help="Evaluate a local CSV; target and group are never features")
    run.add_argument("csv_path", type=Path)
    _add_columns(run)
    _add_protocol(run)
    predict = sub.add_parser("predict", help="Fit a local model on training CSV and predict feature-only new CSV")
    predict.add_argument("csv_path", type=Path, help="Labeled training CSV")
    predict.add_argument("new_csv_path", type=Path, help="New rows: exactly the feature columns")
    _add_columns(predict)
    _add_protocol(predict, outer=False)
    plan = sub.add_parser("plan", help="Validate inputs and preview exact split/fit counts without fitting or writing",
                          description="Print a JSON execution plan to stdout only. No models, predictions or result files are created.")
    plan_sub = plan.add_subparsers(dest="plan_mode", required=True)
    for mode in ("run", "predict"):
        preview = plan_sub.add_parser(mode, help=f"Preview {mode} without fitting or writing", allow_abbrev=False)
        preview.add_argument("csv_path", type=Path, help="Labeled training CSV")
        if mode == "predict":
            preview.add_argument("new_csv_path", type=Path, help="New rows: exactly the feature columns")
        _add_columns(preview)
        _add_protocol(preview, outer=mode == "run", output=False)
    verify = sub.add_parser("verify", help="Check existing bundle SHA-256 byte integrity without fitting or writing",
                            description="Check SHA-256 byte integrity only, not scientific/model correctness. No fitting, writes, or model loading.",
                            epilog="Requires POSIX no-follow file access. Limits: 16 KiB checksums.json; 64 MiB per payload. Exit status: 0 verified, 2 failed.")
    verify.add_argument("run_dir", type=Path, help="Existing evaluation or prediction directory (no symlink components)")
    schema = sub.add_parser("validate-prediction-bundle", allow_abbrev=False,
                            help="Check supported regression prediction metadata and CSV consistency",
                            description="Read-only checksum and bounded structural consistency, not scientific/model validation.",
                            epilog="POSIX only. Limits: 16 KiB checksums, 1 MiB manifest, 4 MiB CSV. Exit 0 checked, 2 failed.")
    schema.add_argument("run_dir", type=Path)
    fixture = sub.add_parser("fixture", help="Write a wholly synthetic materials-process CSV")
    fixture.add_argument("--out", type=Path, required=True)
    fixture.add_argument("--groups", type=int, default=24)
    fixture.add_argument("--rows-per-group", type=int, default=12)
    fixture.add_argument("--seed", type=int, default=7)
    from .binary.cli import add_binary_parser, run_binary
    add_binary_parser(sub)
    args = parser.parse_args(argv)
    try:
        if args.command == "binary":
            return run_binary(args)
        if args.command == "verify":
            result = verify_bundle(args.run_dir)
            print(f"Verified {result.bundle_type} bundle: {len(result.files)} files, {result.total_bytes} payload bytes match checksums.json.")
            print("Byte integrity only, not scientific/model correctness, authenticity, or proof of execution.")
            print("Unlisted files are not checked. This is not an atomic directory snapshot; verify an idle directory.")
            return 0
        if args.command == "validate-prediction-bundle":
            result = validate_prediction_bundle_schema(args.run_dir)
            print(f"Prediction contract {result.contract_version} checked: schema {result.schema_version}, {result.n_predictions} rows, {result.n_features} features.")
            print(f"Input support: {result.input_support}.")
            print("Checked metadata subset and exact CSV bytes only; not scientific correctness, authenticity, model execution, or an atomic snapshot.")
            print("Additional manifest fields and unlisted files are not validated.")
            return 0
        if args.command == "fixture":
            frame = make_materials_fixture(args.groups, args.rows_per_group, args.seed)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False)
            print(f"Synthetic fixture written to {args.out}")
            return 0
        source_hash = None
        if args.command == "predict":
            ensure_prediction_output(args.out)
        if args.command in ("run", "predict"):
            sources = [args.csv_path] + ([args.new_csv_path] if args.command == "predict" else [])
            if args.features_file is not None:
                sources.append(args.features_file)
            protect_input_files(sources, args.out)
        if args.command != "plan" and not args.overwrite and any((args.out / name).exists() or (args.out / name).is_symlink() for name in OUTPUT_NAMES):
            raise FileExistsError("Output already contains a report; choose a new --out or use --overwrite.")
        if args.command == "demo":
            frame = make_materials_fixture(args.groups, args.rows_per_group, args.seed)
            target, group, features = "response", "batch", None
        else:
            frame, features, source_hash = _read_training(args)
            target, group = args.target, args.group
        options = dict(target=target, group=group, features=features,
                       inner_splits=args.inner_splits, random_state=args.seed,
                       baselines=tuple(x.strip() for x in args.baselines.split(",")),
                       objective=args.objective, source_sha256=source_hash, categorical_encoding=args.categorical_encoding)
        if args.command == "plan":
            if args.plan_mode == "predict":
                incoming, prediction_source_hash = _read_csv_with_hash(args.new_csv_path)
                result = plan_prediction(frame, incoming, **options, prediction_source_sha256=prediction_source_hash)
            else:
                result = plan_benchmark(frame, **options, outer_splits=args.outer_splits)
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "predict":
            incoming, prediction_source_hash = _read_csv_with_hash(args.new_csv_path)
            result = fit_predict(frame, incoming, **options, prediction_source_sha256=prediction_source_hash)
            destination = result.write(args.out, overwrite=args.overwrite)
            print(f"Predictions: {destination}\nInner-selected candidate: {result.manifest['selected_candidate']}")
            print("Full-data refit predictions, not held-out evaluation. Use `run` to evaluate generalization first.")
        else:
            result = benchmark(frame, **options, outer_splits=args.outer_splits)
            report = result.write(args.out, overwrite=args.overwrite)
            print(result.leaderboard[["model", "group_mae", "mae", "rmse", "r2"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
            print(f"\nOffline report: {report}\nRun key: {result.manifest['run_key']}")
            print("Selection uses only inner folds. Review outputs for sensitive values before sharing.")
        return 0
    except (ValueError, TypeError, OSError, pd.errors.ParserError) as exc:
        print(f"groupbench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
