"""Command-line entry point; inputs and outputs are local files only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path
import sys

import pandas as pd

from . import __version__
from .data import MAX_ROWS, make_materials_fixture
from .evaluation import benchmark
from .prediction import ensure_prediction_output, fit_predict
from .verification import verify_bundle

OUTPUT_NAMES = ("report.html", "manifest.json", "summary.json", "predictions.csv", "checksums.json")


def _add_protocol(parser, *, outer=True):
    parser.add_argument("--out", type=Path, required=True, help="Local output directory")
    if outer:
        parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baselines", default="dummy,ridge,extra_trees", help="Comma-separated families: dummy,ridge,extra_trees")
    parser.add_argument("--objective", choices=("group_mae", "mae"), default="group_mae")
    parser.add_argument("--overwrite", action="store_true", help="Replace known output files in --out")


def _add_columns(parser):
    parser.add_argument("--target", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--features", help="Optional comma-separated feature allowlist")


def protect_input_files(inputs, output_dir):
    """Refuse to replace a source through a direct, symlink, or hardlink alias."""
    for source in inputs:
        for name in OUTPUT_NAMES:
            destination = Path(output_dir) / name
            same_file = destination.exists() and source.exists() and destination.samefile(source)
            if destination.resolve() == source.resolve() or same_file:
                raise ValueError("Output would overwrite an input file; choose a separate --out directory.")


class _HashingReader(io.RawIOBase):
    """Hash only bytes returned to the decoder, without reopening the source."""

    def __init__(self, source, digest):
        super().__init__()
        self.source = source
        self.digest = digest

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.source.readinto(buffer)
        if count:
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
    with path.open("rb") as source, io.TextIOWrapper(
        _HashingReader(source, digest), encoding="utf-8-sig", newline=""
    ) as handle:
        record_lines = []

        def source_lines():
            for line in handle:
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
    verify = sub.add_parser("verify", help="Check existing bundle SHA-256 byte integrity without fitting or writing",
                            description="Check SHA-256 byte integrity only, not scientific/model correctness. No fitting, writes, or model loading.",
                            epilog="Requires POSIX no-follow file access. Limits: 16 KiB checksums.json; 64 MiB per payload. Exit status: 0 verified, 2 failed.")
    verify.add_argument("run_dir", type=Path, help="Existing evaluation or prediction directory (no symlink components)")
    fixture = sub.add_parser("fixture", help="Write a wholly synthetic materials-process CSV")
    fixture.add_argument("--out", type=Path, required=True)
    fixture.add_argument("--groups", type=int, default=24)
    fixture.add_argument("--rows-per-group", type=int, default=12)
    fixture.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_bundle(args.run_dir)
            print(f"Verified {result.bundle_type} bundle: {len(result.files)} files, {result.total_bytes} payload bytes match checksums.json.")
            print("Byte integrity only, not scientific/model correctness, authenticity, or proof of execution.")
            print("Unlisted files are not checked. This is not an atomic directory snapshot; verify an idle directory.")
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
            protect_input_files(sources, args.out)
        if not args.overwrite and any((args.out / name).exists() or (args.out / name).is_symlink() for name in OUTPUT_NAMES):
            raise FileExistsError("Output already contains a report; choose a new --out or use --overwrite.")
        if args.command == "demo":
            frame = make_materials_fixture(args.groups, args.rows_per_group, args.seed)
            target, group, features = "response", "batch", None
        else:
            frame, source_hash = _read_csv_with_hash(args.csv_path, group=args.group)
            target, group = args.target, args.group
            features = [c.strip() for c in args.features.split(",")] if args.features else None
        options = dict(target=target, group=group, features=features,
                       inner_splits=args.inner_splits, random_state=args.seed,
                       baselines=tuple(x.strip() for x in args.baselines.split(",")),
                       objective=args.objective, source_sha256=source_hash)
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
