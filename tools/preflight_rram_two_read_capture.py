"""Check a PBA-like nominal-one-second ADC-code capture, not retention eligibility."""
import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

FIELDS = ("addr", "i", "timept", "time", "after_write", "after_relax")
INTEGER = re.compile(r"[0-9]{1,9}", re.ASCII)
DECIMAL = re.compile(r"[0-9]{1,2}(?:\.[0-9]{1,30})?", re.ASCII)
CAPTURE_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", re.ASCII)
LANES = frozenset(range(16))
MAX_BYTES = 8 * 1024 * 1024


def checked_integer(value):
    if not INTEGER.fullmatch(value):
        raise ValueError("address, lane, timepoint and ADC codes must be unsigned decimal integers")
    return int(value)


def observable_fingerprint(groups):
    """Digest the six validated native fields, not metadata or user labels."""
    digest = hashlib.sha256(b"clotho.two_read.required_observables.v1\n")
    for address, (elapsed, pairs) in sorted(groups.items()):
        whole, _, fraction = format(elapsed, "f").partition(".")
        # Strip representational zeros without Decimal.normalize() rounding
        # ambient-context precision for near-identical elapsed values.
        timestamp = whole + "." + fraction.rstrip("0")
        for lane, (initial, final) in sorted(pairs.items()):
            digest.update(f"{address},{lane},1,{timestamp},{initial},{final}\n".encode("ascii"))
    return digest.digest()


def planned_addresses(specs):
    plan = {}
    for capture, chip, expression in specs or ():
        if not CAPTURE_ID.fullmatch(capture) or not CAPTURE_ID.fullmatch(chip):
            raise ValueError("invalid planned capture or chip identity")
        key = capture, chip
        if key in plan:
            raise ValueError("duplicate sparse address plan for capture/chip")
        tokens = expression.split(",")
        if not all(INTEGER.fullmatch(token) for token in tokens):
            raise ValueError("sparse address plan requires comma-separated unsigned addresses")
        addresses = {int(token) for token in tokens}
        if len(addresses) != len(tokens):
            raise ValueError("duplicate address in sparse plan")
        plan[key] = addresses
    return plan


def audit(path, seen_files, seen_payloads, seen_observables, expected_address_range=None, expected_addresses=None):
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                           | getattr(os, "O_BINARY", 0)), "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("capture must be a regular file")
        physical_file = info.st_dev, info.st_ino
        if physical_file in seen_files:
            raise ValueError("same input file assigned to multiple capture/chip identities")
        seen_files.add(physical_file)
        raw = file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("capture exceeds byte limit")
    digest = hashlib.sha256(raw).digest()
    if digest in seen_payloads:
        raise ValueError("byte-identical capture payloads cannot support separate capture/chip identities")
    seen_payloads.add(digest)
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    header = next(reader, [])
    if len(set(header)) != len(header) or not set(FIELDS).issubset(header):
        raise ValueError("capture requires distinct header fields including all six native columns")
    indices = tuple(header.index(name) for name in FIELDS)
    groups = {}
    for row in reader:
        if len(row) != len(header):
            raise ValueError("capture row column count differs from header")
        address, lane, nominal, elapsed, initial, final = (row[i] for i in indices)
        addr, lane, timepoint, initial, final = map(checked_integer, (address, lane, nominal, initial, final))
        if lane not in LANES or timepoint != 1 or not (2 <= initial <= 64 and 2 <= final <= 64):
            raise ValueError("capture requires lanes 0–15, nominal timepoint 1 and ADC codes 2–64")
        if not DECIMAL.fullmatch(elapsed) or not Decimal("0.98") < Decimal(elapsed) < Decimal("1.02"):
            raise ValueError("shared elapsed time must be a decimal strictly between 0.98 and 1.02 seconds")
        elapsed = Decimal(elapsed)
        shared, pairs = groups.setdefault(addr, (elapsed, {}))
        if lane in pairs:
            raise ValueError("duplicate native address/lane assignment")
        if shared != elapsed:
            raise ValueError("lanes at an address do not share one elapsed time")
        pairs[lane] = (initial, final)
    if not groups or any(pairs.keys() != LANES for _, pairs in groups.values()):
        raise ValueError("capture requires nonempty complete 16-lane address groups")
    if expected_address_range is not None:
        low, high = expected_address_range
        # Distinct dictionary keys and exact inclusive span establish support
        # without materializing the whole expected range.
        if len(groups) != high - low + 1 or min(groups) != low or max(groups) != high:
            raise ValueError("capture does not exactly match the frozen address range")
    if expected_addresses is not None and groups.keys() != expected_addresses:
        raise ValueError("capture does not match its frozen sparse address plan")
    semantics = observable_fingerprint(groups)
    if semantics in seen_observables:
        raise ValueError("identical required capture observables cannot support separate capture/chip identities")
    seen_observables.add(semantics)
    return {
        "complete_address_groups": len(groups),
        "native_paired_rows": len(groups) * len(LANES),
        "same_row_persistence_abs_error_sum_adc_codes": sum(
            abs(initial - final) for _, pairs in groups.values() for initial, final in pairs.values()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="append", nargs=3, required=True, metavar=("CAPTURE_ID", "CHIP_ID", "CSV"))
    expected_group = parser.add_mutually_exclusive_group()
    expected_group.add_argument("--expected-address-range", nargs=2, metavar=("FIRST", "LAST"),
                        help="require exactly this predeclared inclusive contiguous range in every capture")
    expected_group.add_argument("--expected-addresses", action="append", nargs=3,
                                metavar=("CAPTURE_ID", "CHIP_ID", "ADDR,ADDR,..."),
                                help="exact independently declared sparse addresses, one option per capture")
    args = parser.parse_args()
    expected = None
    if args.expected_address_range is not None:
        first, last = args.expected_address_range
        if not INTEGER.fullmatch(first) or not INTEGER.fullmatch(last) or int(first) > int(last):
            parser.error("expected address range requires two ordered unsigned native addresses")
        expected = int(first), int(last)
    try:
        plans = planned_addresses(args.expected_addresses)
        if plans and plans.keys() != {(capture, chip) for capture, chip, _ in args.capture}:
            raise ValueError("sparse address plans must cover exactly the requested capture/chip identities")
        seen = set()
        files = set()
        payloads = set()
        observables = set()
        reports = []
        for capture_id, chip_id, filename in args.capture:
            if not CAPTURE_ID.fullmatch(capture_id) or not CAPTURE_ID.fullmatch(chip_id):
                raise ValueError("invalid capture or chip identity")
            key = capture_id, chip_id
            if key in seen:
                raise ValueError("duplicate capture/chip identity")
            seen.add(key)
            reports.append({"capture_id": capture_id, "chip_id": chip_id,
                            **audit(Path(filename), files, payloads, observables, expected, plans.get(key))})
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: capture could not be read as UTF-8 CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    report = {
        "schema": "clotho.rram_two_read_capture_preflight.v1",
        "captures": reports,
        "scope": "paired ADC codes at a nominal one-second timepoint; elapsed time shared by address",
        "physical_capture_independence_verified": False,
        "verified_nominal_zero_timestamp": False,
        "native_100ks_endpoint": False,
        "retention_gate_eligible": False,
    }
    if expected is not None:
        report["matched_declared_address_range"] = list(expected)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
