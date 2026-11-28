"""Audit voltage/time-only bipolar sweep geometry without inspecting any current."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
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


def classify_local_clocks(points):
    if points[0][0] != 0:
        return "first_local_clock_not_zero", None
    segments = [[points[0][1]]]
    for (previous, _), (clock, voltage) in zip(points, points[1:]):
        if clock == previous:
            return "tied_local_clock_values", None
        if clock < previous:
            if clock != 0:
                return "clock_rewind_not_to_exact_zero", None
            segments.append([])
        segments[-1].append(voltage)
    if len(segments) != 4:
        return "expected_exactly_four_local_clock_segments", None
    if any(len(leg) < 2 for leg in segments):
        return "insufficient_rows_in_local_clock_segment", None
    if any(left[-1] != right[0] for left, right in zip(segments, segments[1:])):
        return "voltage_mismatch_at_local_clock_reset", None
    up, back, down, return_leg = segments
    if any(v != 0 for v in (up[0], back[-1], down[0], return_leg[-1])):
        return "missing_exact_zero_at_sweep_endpoint_or_middle", None
    for leg, sign in zip(segments, (1, -1, -1, 1)):
        if any(((v1 > v0) - (v1 < v0)) != sign for v0, v1 in zip(leg, leg[1:])):
            return "local_segment_voltage_direction_not_strict", None
    # Strict leg slopes and exact zeros guarantee opposite-sign apices.
    # Each apex is observed in two separate segments, not merged.
    return "four_local_clock_segments_with_matched_voltage_edges", True


def csv_records(path, *, expected_sha256=None):
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("input must be a regular file")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input exceeds byte limit")
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("geometry input SHA-256 mismatch")
    return csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)


def declared_leg_agreement(path, input_ordinals, inferred):
    reader = csv_records(path)
    header = next(reader, [])
    if len(header) != 3 or set(header) != {"trace_id", "capture_record", "protocol_leg"}:
        raise ValueError("segment roster requires trace_id,capture_record,protocol_leg")
    indices = [header.index(name) for name in ("trace_id", "capture_record", "protocol_leg")]
    declarations = {}
    for count, row in enumerate(reader, 1):
        if count > MAX_ROWS or len(row) != 3:
            raise ValueError("segment roster row limit exceeded or wrong field count")
        identity, ordinal, leg = (row[i] for i in indices)
        if not re.fullmatch(r"[0-9]{1,6}", ordinal) or leg not in {"1", "2", "3", "4"}:
            raise ValueError("segment roster requires record numbers and leg 1–4")
        number = int(ordinal)
        if input_ordinals.get(number) != identity:
            raise ValueError("segment roster does not identify the exact geometry record")
        if number in declarations:
            raise ValueError("duplicate geometry record in segment roster")
        declarations[number] = int(leg)
    if len(declarations) != len(input_ordinals):
        raise ValueError("segment roster must cover every geometry record")
    counts = Counter()
    for number, leg in declarations.items():
        actual = inferred.get(number)
        status = ("local_geometry_unresolved" if actual is None else
                  "declared_leg_matches_local_geometry" if actual == leg else
                  "declared_leg_conflicts_with_local_geometry")
        counts[status] += 1
    return dict(sorted(counts.items()))


def audit(path, *, local_clocks=False, segment_roster=None, expected_geometry_sha256=None):
    if segment_roster is not None and (not local_clocks or expected_geometry_sha256 is None):
        raise ValueError("segment roster requires confirmed local clocks and a trusted geometry SHA-256")
    if expected_geometry_sha256 is not None and not re.fullmatch(r"[a-f0-9]{64}", expected_geometry_sha256):
        raise ValueError("expected geometry SHA-256 must be 64 lowercase hexadecimal digits")
    reader = csv_records(path, expected_sha256=expected_geometry_sha256)
    header = next(reader, [])
    if len(header) != len(FIELDS) or set(header) != set(FIELDS):
        raise ValueError("input requires exactly trace_id,time_s,voltage_V; omit all current labels")
    columns = [header.index(field) for field in FIELDS]
    traces = defaultdict(list)
    trace_records = defaultdict(list)
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
        if segment_roster is not None:
            trace_records[identity].append(n)
    if not traces:
        raise ValueError("no voltage/time traces")
    classify_trace = classify_local_clocks if local_clocks else classify
    outcomes = [classify_trace(trace) for trace in traces.values()]
    report = {
        "schema": "clotho.rram_bipolar_sweep_preflight.v1",
        "declared_traces": len(traces),
        "traces_by_geometry": dict(sorted(Counter(status for status, _ in outcomes).items())),
        "recognized_bipolar_traces_with_repeated_apex": sum(apex is True for _, apex in outcomes),
        "recognized_bipolar_traces_with_unique_apices": sum(apex is False for _, apex in outcomes),
    }
    if local_clocks:
        report["clock_interpretation"] = "four_declared_local_clocks"
        report["intersegment_elapsed_time_verified"] = False
        report["physical_time_unit_verified"] = False
    if segment_roster is not None:
        input_ordinals, inferred = {}, {}
        for (identity, points), (status, _) in zip(traces.items(), outcomes):
            ordinals = trace_records[identity]
            input_ordinals.update((ordinal, identity) for ordinal in ordinals)
            if status == "four_local_clock_segments_with_matched_voltage_edges":
                leg, previous = 1, points[0][0]
                for ordinal, (clock, _) in zip(ordinals, points):
                    if clock < previous:
                        leg += 1
                    inferred[ordinal] = leg
                    previous = clock
        report["declared_segment_roster_support_by_record"] = declared_leg_agreement(
            segment_roster, input_ordinals, inferred
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Trace names are not verified devices. Geometric reversals are not switching events; "
        "a zero bracket does not establish a measured zero anchor."
    ))
    parser.add_argument("csv", type=Path)
    parser.add_argument("--confirmed-four-local-segment-clocks", action="store_true",
                        help="confirm a source protocol with three documented rewinds to local zero")
    parser.add_argument("--segment-roster", type=Path,
                        help="compare a complete independently declared protocol leg per input data record")
    parser.add_argument("--expected-geometry-sha256",
                        help="bind segment roster positions to previously trusted exact geometry bytes")
    args = parser.parse_args()
    try:
        report = audit(args.csv, local_clocks=args.confirmed_four_local_segment_clocks,
                       segment_roster=args.segment_roster, expected_geometry_sha256=args.expected_geometry_sha256)
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: input could not be read as UTF-8 CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
