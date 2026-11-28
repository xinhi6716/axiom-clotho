"""Audit voltage/time-only bipolar sweep geometry without inspecting any current."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import re
import stat

FIELDS = ("trace_id", "time_s", "voltage_V")
ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", re.ASCII)
TOKEN = re.compile(r"[+-]?[0-9]{1,18}(?:\.[0-9]{1,30})?", re.ASCII)
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 100_000


def classify(points):
    if len(points) < 4:
        return "too_few_rows", None
    if any(t1 <= t0 for (t0, _), (t1, _) in zip(points, points[1:])):
        return "timestamp_order_not_strict", None
    volts = [v for _, v in points]
    if volts[0] != 0 or volts[-1] != 0:
        return "start_or_end_not_exactly_zero", None
    if min(volts) >= 0 or max(volts) <= 0:
        return "both_voltage_polarities_not_observed", None
    directions = []
    for a, b in zip(volts, volts[1:]):
        sign = (b > a) - (b < a)
        if sign and (not directions or directions[-1] != sign):
            directions.append(sign)
    if directions != [1, -1, 1]:
        return "unresolved_or_extra_voltage_reversal", None
    # Positive apex and negative apex delimit the common descending run.
    # No interpolated crossing is promoted to a measured zero-voltage row.
    high, low = max(volts), min(volts)
    last_high = len(volts) - 1 - volts[::-1].index(high)
    first_low = volts.index(low)
    zeros = volts[last_high + 1:first_low].count(0)
    status = ("bipolar_with_unique_measured_middle_zero" if zeros == 1 else
              "bipolar_with_middle_zero_plateau" if zeros else "bipolar_only_brackets_middle_zero")
    apex_repeat = volts.count(high) > 1 or volts.count(low) > 1
    return status, apex_repeat


def audit(path):
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("input must be a regular file")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input exceeds byte limit")
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    header = next(reader, [])
    if len(header) != len(FIELDS) or set(header) != set(FIELDS):
        raise ValueError("input requires exactly trace_id,time_s,voltage_V; omit all current labels")
    columns = [header.index(field) for field in FIELDS]
    traces = defaultdict(list)
    for n, row in enumerate(reader, 1):
        if n > MAX_ROWS or len(row) != len(FIELDS):
            raise ValueError("row limit exceeded or wrong field count")
        identity, time, voltage = (row[i] for i in columns)
        if not ID.fullmatch(identity) or not TOKEN.fullmatch(time) or not TOKEN.fullmatch(voltage):
            raise ValueError("expected opaque trace identity and exact fixed-point time/voltage")
        elapsed, bias = Decimal(time), Decimal(voltage)
        if elapsed < 0:
            raise ValueError("time cannot be negative")
        traces[identity].append((elapsed, bias))
    if not traces:
        raise ValueError("no voltage/time traces")
    outcomes = [classify(trace) for trace in traces.values()]
    return {
        "schema": "clotho.rram_bipolar_sweep_preflight.v1",
        "declared_traces": len(traces),
        "traces_by_geometry": dict(sorted(Counter(status for status, _ in outcomes).items())),
        "recognized_bipolar_traces_with_repeated_apex": sum(apex is True for _, apex in outcomes),
        "recognized_bipolar_traces_with_unique_apices": sum(apex is False for _, apex in outcomes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Trace names are not verified devices. Geometric reversals are not switching events; "
        "a zero bracket does not establish a measured zero anchor."
    ))
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.csv)
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: input could not be read as UTF-8 CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
