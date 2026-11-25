"""Audit explicitly mapped RRAM events. Read-only; emits aggregate counts."""
import argparse
import csv
from collections import Counter, defaultdict
from decimal import Decimal
import io
import json
import math
from pathlib import Path
import re
import stat

FIELDS = ("chip", "cell", "episode", "transition", "event_index", "kind",
          "voltage_V", "width_s", "conductance_S", "protocol")
ROLES = ("pre_read", "write", "post_read")
IDENTIFIER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", re.ASCII)
INDEX = re.compile(r"[0-9]{1,18}", re.ASCII)
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", re.ASCII)
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 100_000


def number(raw, positive=False):
    if len(raw) > 64 or not NUMBER.fullmatch(raw):
        return None
    val = float(raw)
    if not math.isfinite(val) or not 1e-150 <= abs(val) <= 1e150:
        return None
    return val if not positive or val > 0 else None


def coverage(entries):
    """Summarize shared declared conditions without publishing their labels."""
    by_condition = defaultdict(Counter)
    for signature, chip in entries:
        by_condition[signature][chip] += 1
    return {
        "declared_strata": len(by_condition),
        "multi_chip_strata": sum(len(chips) >= 2 for chips in by_condition.values()),
        "eligible_transitions_in_multi_chip_strata": sum(
            sum(chips.values()) for chips in by_condition.values() if len(chips) >= 2
        ),
    }


def audit(path):
    """Return redacted counts, rejecting invalid identities and local timelines."""
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("input must be a regular file")
    with path.open("rb") as source:
        raw = source.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input exceeds byte limit")
    if b"\x00" in raw:
        raise ValueError("input contains NUL")
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    header = next(reader, [])
    if len(header) != len(FIELDS) or set(header) != set(FIELDS):
        raise ValueError("input must have the declared columns exactly once")
    transitions = defaultdict(list)
    previous = {}
    seen = set()
    invalid = set()
    count = 0
    for count, row in enumerate(reader, 1):
        if count > MAX_ROWS or len(row) != len(FIELDS):
            raise ValueError("row limit exceeded or CSV record has wrong field count")
        d = dict(zip(header, row))
        key = tuple(d[k] for k in FIELDS[:4])
        if not all(IDENTIFIER.fullmatch(s) for s in key):
            raise ValueError("invalid physical identity or transition label")
        if not INDEX.fullmatch(d["event_index"]):
            raise ValueError("invalid event index")
        episode = key[:3]
        idx = int(d["event_index"])
        old = previous.get(episode)
        if old and idx <= old[0]:
            raise ValueError("event indices must increase within each chip/cell/episode")
        if old and idx > old[0] + 1 and old[1] == key:
            invalid.add(key)
        if key in seen and old and old[1] != key:
            invalid.add(key)
        seen.add(key)
        previous[episode] = (idx, key)
        role = d["kind"]
        v = number(d["voltage_V"])
        if role == "write":
            width = number(d["width_s"], True)
            valid = v is not None and width is not None and d["conductance_S"] == ""
            if valid:
                # Preserve distinct declared write settings even if they
                # share a binary64 representation. Admission is unchanged.
                v = Decimal(d["voltage_V"])
                width = Decimal(d["width_s"])
        else:
            width = None
            valid = role in ("pre_read", "post_read") and v is not None and (
                d["width_s"] == "" and number(d["conductance_S"], True) is not None
            )
        protocol = d["protocol"]
        transitions[key].append((role, v, valid and bool(IDENTIFIER.fullmatch(protocol)), protocol, width))
    reasons = Counter()
    eligible = []
    read_coverage, action_coverage = [], []
    for key, events in transitions.items():
        if key in invalid:
            reasons["noncontiguous_episode_events"] += 1
        elif tuple(e[0] for e in events) != ROLES:
            reasons["expected_exactly_pre_write_post"] += 1
        elif not all(e[2] for e in events):
            reasons["invalid_event_fields"] += 1
        elif len({e[3] for e in events}) != 1:
            reasons["protocol_mismatch"] += 1
        elif events[0][1] != events[2][1]:
            reasons["read_voltage_mismatch"] += 1
        else:
            eligible.append(key)
            read_signature = (events[0][3], events[0][1])
            action_signature = read_signature + (events[1][1], events[1][4])
            read_coverage.append((read_signature, key[0]))
            action_coverage.append((action_signature, key[0]))
    return {
        "schema": "clotho.rram_write_pair_audit.v1",
        "events": count,
        "declared_transitions": len(transitions),
        "eligible_transitions": len(eligible),
        "rejected_transitions_by_primary_reason": dict(sorted(reasons.items())),
        "eligible_declared_cells": len({k[:2] for k in eligible}),
        "eligible_declared_chips": len({k[0] for k in eligible}),
        "declared_read_condition_coverage": coverage(read_coverage),
        "declared_exact_action_coverage": coverage(action_coverage),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.csv)
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: input could not be read as UTF-8 CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if (not report["eligible_transitions"] or
                 report["eligible_transitions"] != report["declared_transitions"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
