#!/usr/bin/env python3
"""Audit one declared semiconductor device/wafer/lot train/test boundary.

Metadata only: no split generation, measurements, targets, fitting, or inference
of physical identity, independent samples, absent leakage, or causal validity.
"""
import argparse
import hashlib
import json
import os
import stat
import sys

COLUMNS = ("observation_id", "partition", "physical_device_id", "wafer_id", "lot_id")
LEVEL_COLUMNS = {"device": "physical_device_id", "wafer": "wafer_id", "lot": "lot_id"}
LEVELS = tuple(LEVEL_COLUMNS)
PARTITIONS = ("train", "test")
MAX_BYTES = 8 * 1024 * 1024
MAX_RECORD_BYTES = 65536
MAX_ROWS = 10000
MAX_REPORT_BYTES = 16 * 1024 * 1024
IDENTITY_CHARACTER_POLICY = "unicode-16.0-cc-cf-cs-v1"
# Pinned Unicode 16.0 General_Category Cc/Cf/Cs, independent of host UCD.
FORBIDDEN_RANGES = (
    (0x0, 0x1F),
    (0x7F, 0x9F),
    (0xAD, 0xAD),
    (0x600, 0x605),
    (0x61C, 0x61C),
    (0x6DD, 0x6DD),
    (0x70F, 0x70F),
    (0x890, 0x891),
    (0x8E2, 0x8E2),
    (0x180E, 0x180E),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x2064),
    (0x2066, 0x206F),
    (0xD800, 0xDFFF),
    (0xFEFF, 0xFEFF),
    (0xFFF9, 0xFFFB),
    (0x110BD, 0x110BD),
    (0x110CD, 0x110CD),
    (0x13430, 0x1343F),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0001, 0xE0001),
    (0xE0020, 0xE007F),
)
LIMITATIONS = [
    "Globally namespaced identities and the intended holdout level are user declarations, not verified physical facts.",
    "Identity strings are compared exactly: no normalization, case folding, alias merging, or namespace construction.",
    "Distinct aliases, Unicode normalization forms or lookalikes may hide the same real device, wafer or lot.",
    "Counts are supplied rows and distinct declared IDs, not independent objects or effective sample sizes.",
    "Only one supplied train/test assignment is audited; no rows are dropped, repaired, deduplicated or reassigned.",
    "Coarser-level overlap is descriptive when not required by the selected holdout; no scientifically correct level is selected.",
    "No observed required-boundary overlap does not establish physical independence or absence of other leakage.",
    "Omitted rows, incorrect identities, target/preprocessing leakage, time, campaigns and shared equipment are not assessed.",
    "The audit does not establish dataset completeness, causal validity, predictive usefulness or population applicability.",
]


class InputError(ValueError):
    """Categorical diagnostic, never including private IDs, values or paths."""


def _record(blob, position):
    start, fields = position, []
    while True:
        cell = bytearray()
        if position < len(blob) and blob[position] == 34:
            position += 1
            while True:
                if position >= len(blob):
                    raise InputError("Unclosed CSV quote")
                char = blob[position]
                position += 1
                if char == 34:
                    if position < len(blob) and blob[position] == 34:
                        cell.append(34)
                        position += 1
                    else:
                        break
                else:
                    cell.append(char)
                if position - start > MAX_RECORD_BYTES:
                    raise InputError("CSV record byte limit exceeded")
            if position < len(blob) and blob[position] not in (44, 10, 13):
                raise InputError("Bytes after quoted CSV field")
        else:
            while position < len(blob) and blob[position] not in (44, 10, 13):
                char = blob[position]
                position += 1
                if char == 34:
                    raise InputError("Quote inside unquoted CSV field")
                cell.append(char)
                if position - start > MAX_RECORD_BYTES:
                    raise InputError("CSV record byte limit exceeded")
        fields.append(bytes(cell))
        if len(fields) > len(COLUMNS) or position - start > MAX_RECORD_BYTES:
            raise InputError("CSV width or record limit exceeded")
        if position < len(blob) and blob[position] == 44:
            position += 1
            continue
        if position < len(blob) and blob[position] == 13:
            if position + 1 >= len(blob) or blob[position + 1] != 10:
                raise InputError("Bare CSV carriage return")
            position += 2
        elif position < len(blob) and blob[position] == 10:
            position += 1
        if position - start > MAX_RECORD_BYTES:
            raise InputError("CSV record byte limit exceeded")
        return fields, position


def _identity(raw):
    try:
        value = raw.decode("utf-8", "strict")
    except UnicodeError as error:
        raise InputError("Invalid identity UTF-8") from error
    if (not value or len(value) > 128 or value != value.strip()
            or any(any(start <= ord(char) <= end for start, end in FORBIDDEN_RANGES) for char in value)):
        raise InputError("IDs must be bounded, nonempty, unpadded and free of control/format/surrogate characters")
    return value


def parse_input(blob):
    if not isinstance(blob, bytes) or len(blob) > MAX_BYTES:
        raise InputError("CSV byte or type limit exceeded")
    position = 3 if blob.startswith(b"\xef\xbb\xbf") else 0
    fields, position = _record(blob, position)
    if len(fields) != len(COLUMNS) or set(fields) != {name.encode("ascii") for name in COLUMNS}:
        raise InputError("CSV header must contain exactly the five documented columns")
    header = [field.decode("ascii") for field in fields]
    rows, sides = [], set()
    while position < len(blob):
        fields, position = _record(blob, position)
        if len(fields) != len(COLUMNS):
            raise InputError("CSV row width mismatch; blank records are not permitted")
        if len(rows) >= MAX_ROWS:
            raise InputError("CSV row limit exceeded")
        supplied = dict(zip(header, fields))
        row = {"source_row_ordinal": len(rows) + 1}
        row.update({name: _identity(supplied[name]) for name in COLUMNS})
        if row["partition"] not in PARTITIONS:
            raise InputError("Partition must be exactly train or test")
        sides.add(row["partition"])
        rows.append(row)
    if sides != set(PARTITIONS):
        raise InputError("Both train and test must have at least one data record")
    return rows


def audit(blob, holdout_level, declare_global_identities=False):
    if holdout_level not in LEVELS:
        raise InputError("Explicit supported holdout level required")
    if declare_global_identities is not True:
        raise InputError("Explicit globally namespaced identity declaration required")
    rows = parse_input(blob)
    observations = {}
    inventories = {level: {} for level in LEVELS}
    mappings = {(child, parent): {} for child, parent in
                (("physical_device_id", "wafer_id"), ("physical_device_id", "lot_id"), ("wafer_id", "lot_id"))}
    for row in rows:
        ordinal, side = row["source_row_ordinal"], row["partition"]
        observations.setdefault(row["observation_id"], []).append(ordinal)
        for level, column in LEVEL_COLUMNS.items():
            inventories[level].setdefault(row[column], {part: [] for part in PARTITIONS})[side].append(ordinal)
        for (child, parent), mapping in mappings.items():
            mapping.setdefault(row[child], {}).setdefault(row[parent], []).append(ordinal)
    duplicates = [{"observation_id": identity, "source_row_ordinals": ordinals}
                  for identity, ordinals in observations.items() if len(ordinals) > 1]
    conflicts = [{"child_column": child, "parent_column": parent, "child_id": identity,
                  "parents": [{"parent_id": key, "source_row_ordinals": ordinals} for key, ordinals in parents.items()]}
                 for (child, parent), mapping in mappings.items()
                 for identity, parents in mapping.items() if len(parents) > 1]
    required = list(LEVELS[:LEVELS.index(holdout_level) + 1])
    overlaps = {}
    level_counts = {}
    for level, inventory in inventories.items():
        shared = [{"id": identity, "train_source_row_ordinals": sides["train"],
                   "test_source_row_ordinals": sides["test"]}
                  for identity, sides in inventory.items() if sides["train"] and sides["test"]]
        overlaps[level] = {"identity_column": LEVEL_COLUMNS[level], "required_for_holdout": level in required,
                           "overlapping_ids": shared}
        level_counts[level] = {"union": len(inventory),
                               **{side: sum(bool(sides[side]) for sides in inventory.values()) for side in PARTITIONS},
                               "overlap": len(shared)}
    invalid = bool(duplicates or conflicts)
    crossing = any(overlaps[level]["overlapping_ids"] for level in required)
    status = "INVALID_ASSIGNMENTS" if invalid else "HOLDOUT_OVERLAP" if crossing else "NO_SELECTED_HOLDOUT_OVERLAP"
    report = {
        "schema_version": "physical-holdout-audit-v1", "status": status,
        "provenance": {"csv_sha256": hashlib.sha256(blob).hexdigest(), "csv_bytes": len(blob),
                       "row_ordinal_convention": "one-based logical data record, excluding header"},
        "parameters": {"holdout_level": holdout_level, "required_levels": required,
                       "identity_contract_declared": "globally-namespaced", "identity_truth_verified": False,
                       "identity_comparison": "exact_strings_no_normalization",
                       "identity_character_policy": IDENTITY_CHARACTER_POLICY},
        "validity": {"observation_ids_unique": not duplicates, "declared_nesting_consistent": not conflicts,
                     "assessment_possible": not invalid},
        "assessment": {"required_level_overlap_observed": crossing,
                       "conclusion": "not_assessable_invalid_assignments" if invalid else
                       "declared_boundary_overlap_observed" if crossing else "no_declared_boundary_overlap_observed",
                       "scope": "supplied_metadata_only", "exit_code": int(invalid or crossing)},
        "counts": {"input_rows": len(rows), "unique_observation_ids": len(observations),
                   "partition_rows": {side: sum(row["partition"] == side for row in rows) for side in PARTITIONS},
                   "partition_unique_observation_ids": {side: len({row["observation_id"] for row in rows if row["partition"] == side}) for side in PARTITIONS},
                   "declared_ids": level_counts, "duplicate_observation_ids": len(duplicates),
                   "nesting_conflicts": len(conflicts)},
        "rows": rows, "overlaps": overlaps, "duplicate_observation_assignments": duplicates,
        "nesting_conflicts": conflicts, "limitations": LIMITATIONS.copy()}
    report_bytes(report)
    return report


def read_regular(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
                raise InputError("Input must be a bounded regular file")
            blob = stream.read(MAX_BYTES + 1)
    except (OSError, ValueError) as error:
        if isinstance(error, InputError):
            raise
        raise InputError("Cannot read requested regular input") from error
    if len(blob) > MAX_BYTES:
        raise InputError("Input byte limit exceeded")
    return blob


def report_bytes(report):
    payload = (json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode("utf-8")
    if len(payload) > MAX_REPORT_BYTES:
        raise InputError("Complete report byte limit exceeded")
    return payload


def write_report(payload, path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def write_stdout(payload):
    if sys.stdout is None:
        raise InputError("Output stream unavailable")
    try:
        sys.stdout.write(payload.decode("utf-8"))
        sys.stdout.flush()
    except (OSError, ValueError):
        try:
            sys.stdout.close()
        except (OSError, ValueError):
            pass
        raise


def emit_failure(message):
    if sys.stderr is None:
        return
    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        try:
            sys.stderr.close()
        except (OSError, ValueError):
            pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise InputError("Invalid command-line options; use --help")


def main(argv=None):
    parser = Parser(description=__doc__, allow_abbrev=False)
    parser.add_argument("csv", help="Metadata-only CSV for one already chosen train/test assignment")
    parser.add_argument("--holdout-level", choices=LEVELS, required=True)
    parser.add_argument("--declare-global-identities", action="store_true", required=True,
                        help="Declare stable globally namespaced identities; this does not verify their truth")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--stdout", action="store_true", help="Explicitly print every supplied identity and assignment")
    destination.add_argument("--output", help="New report path; never overwrite")
    try:
        args = parser.parse_args(argv)
        report = audit(read_regular(args.csv), args.holdout_level, args.declare_global_identities)
        payload = report_bytes(report)
        if args.stdout:
            write_stdout(payload)
        else:
            write_report(payload, args.output)
        return report["assessment"]["exit_code"]
    except InputError as error:
        emit_failure("Audit failed: " + str(error) + ". No complete report claimed.")
    except (OSError, ValueError, UnicodeError):
        emit_failure("Audit failed: input/output/serialization error. No complete report claimed.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
