"""Audit frozen RRAM retention slots without reading conductance or fitting."""
import argparse
from collections import Counter
import csv
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


def records(path, fields):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                         | getattr(os, "O_BINARY", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("inputs must be regular files")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input exceeds byte limit")
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
    lower, upper = elapsed(origin, early)
    if upper <= 0 or lower > EARLY_UPPER:
        return "early_outside_window"
    if lower <= 0 or upper > EARLY_UPPER:
        return "early_boundary_uncertain"
    lower, upper = elapsed(origin, terminal)
    if upper < TERMINAL_LOWER or lower > TERMINAL_UPPER:
        return "terminal_outside_window"
    if lower < TERMINAL_LOWER or upper > TERMINAL_UPPER:
        return "terminal_boundary_uncertain"
    return "eligible"


def audit(roster_path, capture_path):
    expected = {}
    for key, _ in records(roster_path, IDENTITY):
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
    return {
        "schema": "clotho.rram_retention_anchor_audit.v1",
        "frozen_attempted_episodes": len(expected),
        "frozen_attempted_dies": len({key[:3] for key in expected}),
        "eligible_episodes": classifications["eligible"],
        "unexpected_capture_episodes": len(unexpected),
        "nonscheduled_rows_in_frozen_episodes": off_anchor_rows,
        "frozen_episodes_by_primary_reason": dict(sorted(classifications.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-roster", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.frozen_roster, args.capture)
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: input could not be read as UTF-8 CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["eligible_episodes"] != report["frozen_attempted_episodes"]
               or report["unexpected_capture_episodes"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
