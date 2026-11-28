"""Check a declared 16-lane test pattern across two already exported capture paths."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import stat

HEADER = ("capture_lane", "signature")
LANES = frozenset(range(16))
MAX_BYTES = 16_384


def decode_route(path):
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("signature export must be a regular file")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("signature export exceeds byte limit")
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    header = next(reader, ())
    if len(header) != 2 or set(header) != set(HEADER):
        raise ValueError("signature export needs exactly capture_lane,signature; no outcome columns")
    columns = [header.index(name) for name in HEADER]
    lookup, covered = {}, set()
    for row in reader:
        if len(row) != 2:
            raise ValueError("malformed signature record")
        label, token = (row[i] for i in columns)
        if not re.fullmatch(r"[0-9]{1,2}", label) or int(label) not in LANES:
            raise ValueError("capture lane must be an unsigned integer from 0–15")
        lane = int(label)
        if lane in lookup:
            raise ValueError("duplicate capture lane")
        if not re.fullmatch(r"[01]{8}", token):
            raise ValueError("signature must contain exactly eight binary digits")
        source, complement = int(token[:4], 2), int(token[4:], 2)
        if source ^ complement != 15:
            raise ValueError("signature complement does not match four-bit source lane")
        if source in covered:
            raise ValueError("duplicate decoded reference lane")
        lookup[lane] = source
        covered.add(source)
    if lookup.keys() != LANES:
        raise ValueError("signature export must cover all sixteen capture lanes exactly once")
    return (info.st_dev, info.st_ino), [lookup[lane] for lane in range(16)]


def compare(early, late):
    late_slot = {source: slot for slot, source in enumerate(late)}
    relative = [late_slot[source] for source in early]
    return {
        "schema": "clotho.rram_lane_signature_check.v1",
        "early_capture_lane_to_declared_reference_lane": early,
        "late_capture_lane_to_declared_reference_lane": late,
        "early_to_late_slot_for_same_declared_reference_lane": relative,
        "early_nonidentity_lanes": sum(slot != source for slot, source in enumerate(early)),
        "late_nonidentity_lanes": sum(slot != source for slot, source in enumerate(late)),
        "lanes_reassigned_between_routes": sum(slot != target for slot, target in enumerate(relative)),
        "device_outcomes_read": False,
        "reference_physical_identity_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "The caller must establish an independent lane reference. Bit order is MSB-first lane, "
        "then its four-bit complement. Matching codes alone do not verify the reference wiring; "
        "coordinated changes to both nibbles can form another valid codeword."
    ))
    parser.add_argument("--early-signatures", type=Path, required=True)
    parser.add_argument("--late-signatures", type=Path, required=True)
    args = parser.parse_args()
    try:
        early_file, early = decode_route(args.early_signatures)
        late_file, late = decode_route(args.late_signatures)
        if early_file == late_file:
            raise ValueError("require two distinct capture exports, not the same file")
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: could not read signature export as UTF-8 CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    report = compare(early, late)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(report["early_nonidentity_lanes"] or report["late_nonidentity_lanes"]))


if __name__ == "__main__":
    raise SystemExit(main())
