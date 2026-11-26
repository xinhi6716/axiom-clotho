"""Audit frozen RRAM retention slots without reading conductance or fitting."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

IDENTITY = ("lot", "wafer", "die", "cell", "episode")
TIMING = ("nominal_us", "clock", "epoch", "tick_us", "error_us")
ANCHORS = (0, 1_000_000, 100_000_000_000)
EARLY_UPPER = 1_020_000
TERMINAL_LOWER, TERMINAL_UPPER = 80_000_000_000, 120_000_000_000
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 100_000
KEY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", re.ASCII)
INTEGER = re.compile(r"[0-9]{1,18}", re.ASCII)
SHA256 = re.compile(r"[a-f0-9]{64}", re.ASCII)


def records(path, fields, *, expected_sha256=None):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                         | getattr(os, "O_BINARY", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("inputs must be regular files")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input exceeds byte limit")
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("frozen roster SHA-256 mismatch")
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    header = next(reader, [])
    if len(header) != len(fields) or set(header) != set(fields):
        raise ValueError("input must have the exact declared metadata-only columns")
    positions = [header.index(field) for field in fields]
    for count, row in enumerate(reader, 1):
        if count > MAX_ROWS or len(row) != len(fields):
            raise ValueError("row limit exceeded or record has wrong field count")
        require = tuple(row[p] for p in positions)
        key = require[:len(IDENTITY)]
        if any(not KEY.fullmatch(part) for part in key):
            raise ValueError("invalid acquisition identity")
        yield key, require[len(IDENTITY):]


def integer(value):
    if not INTEGER.fullmatch(value):
        raise ValueError("times must be unsigned integral microseconds of at most 18 digits")
    return int(value)


def elapsed(reference, read):
    tick_ref, err_ref = reference[2:]
    tick_read, err_read = read[2:]
    difference = tick_read - tick_ref
    error = err_ref + err_read
    return difference - error, difference + error


def timing_status(origin, read, nominal):
    if origin[:2] != read[:2]:
        return "different_clock_or_epoch"
    lower, upper = elapsed(origin, read)
    if nominal == ANCHORS[1]:
        if upper <= 0 or lower > EARLY_UPPER:
            return "outside_window"
        if lower <= 0 or upper > EARLY_UPPER:
            return "boundary_uncertain"
    else:
        if upper < TERMINAL_LOWER or lower > TERMINAL_UPPER:
            return "outside_window"
        if lower < TERMINAL_LOWER or upper > TERMINAL_UPPER:
            return "boundary_uncertain"
    return "within_window"


def anchor_statuses(slots):
    """One redacted state for each exact declared slot, never a model score."""
    origin_rows = slots.get(ANCHORS[0], ())
    origin = origin_rows[0] if len(origin_rows) == 1 else None
    for nominal in ANCHORS:
        rows = slots.get(nominal, ())
        if not rows:
            yield "missing_nominal_anchor"
        elif len(rows) > 1:
            yield "duplicate_nominal_anchor"
        elif rows[0] is None:
            yield "unknown_clock_or_time"
        elif nominal == ANCHORS[0]:
            yield "usable_timestamp"
        elif origin is None:
            yield "unusable_origin"
        else:
            yield timing_status(origin, rows[0], nominal)


def verdict(slots):
    # A duplicate scheduled anchor is ambiguous even when both rows agree.
    if any(len(slots.get(key, ())) > 1 for key in ANCHORS):
        return "duplicate_nominal_anchor"
    if any(key not in slots for key in ANCHORS):
        return "missing_nominal_anchor"
    origin, early, terminal = (slots[n][0] for n in ANCHORS)
    if any(stamp is None for stamp in (origin, early, terminal)):
        return "unknown_clock_or_time"
    if not origin[:2] == early[:2] == terminal[:2]:
        return "different_clock_or_epoch"
    for role, nominal, read in zip(("early", "terminal"), ANCHORS[1:], (early, terminal)):
        status = timing_status(origin, read, nominal)
        if status != "within_window":
            return f"{role}_{status}"
    return "eligible"


def audit(roster_path, capture_path, *, diagnose_anchors=False, roster_sha256=None):
    if roster_sha256 is not None and (
            not isinstance(roster_sha256, str) or not SHA256.fullmatch(roster_sha256)):
        raise ValueError("frozen roster SHA-256 must be 64 lowercase hexadecimal digits")
    expected = {}
    for key, _ in records(roster_path, IDENTITY, expected_sha256=roster_sha256):
        if key in expected:
            raise ValueError("duplicate frozen acquisition identity")
        expected[key] = {}
    if not expected:
        raise ValueError("frozen acquisition roster must not be empty")

    present, unexpected = set(), set()
    off_anchor_rows = 0
    for key, (nominal, clock, epoch, tick, error) in records(capture_path, IDENTITY + TIMING):
        nominal = integer(nominal)
        if any(value and not KEY.fullmatch(value) for value in (clock, epoch)):
            raise ValueError("invalid clock identity")
        tick = integer(tick) if tick else None
        error = integer(error) if error else None
        stamp = (clock, epoch, tick, error) if clock and epoch and tick is not None and error is not None else None
        if key not in expected:
            unexpected.add(key)
            continue
        present.add(key)
        if nominal not in ANCHORS:
            off_anchor_rows += 1
            continue
        expected[key].setdefault(nominal, []).append(stamp)

    classifications = Counter(
        verdict(slots) if key in present else "missing_capture"
        for key, slots in expected.items()
    )
    report = {
        "schema": "clotho.rram_retention_anchor_audit.v1",
        "frozen_attempted_episodes": len(expected),
        "frozen_attempted_dies": len({key[:3] for key in expected}),
        "eligible_episodes": classifications["eligible"],
        "unexpected_capture_episodes": len(unexpected),
        "nonscheduled_rows_in_frozen_episodes": off_anchor_rows,
        "frozen_episodes_by_primary_reason": dict(sorted(classifications.items())),
    }
    if roster_sha256 is not None:
        report["frozen_roster_byte_digest_verified"] = True
    if diagnose_anchors:
        by_anchor = {str(nominal): Counter() for nominal in ANCHORS}
        for slots in expected.values():
            for nominal, status in zip(ANCHORS, anchor_statuses(slots)):
                by_anchor[str(nominal)][status] += 1
        report["frozen_nominal_anchor_statuses_by_us"] = {
            nominal: dict(sorted(counts.items())) for nominal, counts in by_anchor.items()
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-roster", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--diagnose-anchors", action="store_true",
                        help="report separate frozen-roster status counts for 0, 1 s and 100 ks")
    parser.add_argument("--frozen-roster-sha256",
                        help="verify exact roster bytes against a previously trusted SHA-256")
    args = parser.parse_args()
    try:
        report = audit(args.frozen_roster, args.capture, diagnose_anchors=args.diagnose_anchors,
                       roster_sha256=args.frozen_roster_sha256)
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: input could not be read as UTF-8 CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["eligible_episodes"] != report["frozen_attempted_episodes"]
               or report["unexpected_capture_episodes"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
