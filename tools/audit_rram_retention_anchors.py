"""Audit frozen RRAM retention slots without reading conductance or fitting."""
import argparse
from collections import Counter, defaultdict
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
STRESS_INTERVALS = ("bias_0_to_1s", "bias_1s_to_100ks")
STRESS_STATES = ("off_schedule_bias_observed", "no_off_schedule_bias_documented", "unknown")
ANCHORS = (0, 1_000_000, 100_000_000_000)
EARLY_UPPER = 1_020_000
TERMINAL_LOWER, TERMINAL_UPPER = 80_000_000_000, 120_000_000_000
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 100_000
KEY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", re.ASCII)
INTEGER = re.compile(r"[0-9]{1,18}", re.ASCII)
SIGNED_INTEGER = re.compile(r"[+-]?[0-9]{1,18}", re.ASCII)
SHA256 = re.compile(r"[a-f0-9]{64}", re.ASCII)


def records(path, fields, *, expected_sha256=None, digest_label="frozen roster"):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                         | getattr(os, "O_BINARY", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("inputs must be regular files")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input exceeds byte limit")
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{digest_label} SHA-256 mismatch")
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


def duplicate_anchor_status(rows):
    """Describe repeated metadata without deciding which row to keep."""
    first = None
    unknown = False
    for stamp in rows:
        if stamp is None:
            unknown = True
        elif first is None:
            first = stamp
        elif stamp != first:
            return "conflicting_usable_stamps"
    if first is None:
        return "all_stamps_unresolved"
    if unknown:
        return "usable_and_unresolved_stamps"
    return "identical_usable_stamps"


def terminal_alternative_status(slots, off_anchor_stamps):
    if not off_anchor_stamps:
        return "no_nonscheduled_rows"
    origins = slots.get(ANCHORS[0], ())
    if len(origins) != 1 or origins[0] is None:
        return "unusable_origin"
    origin = origins[0]
    uncertain = unresolved = False
    for stamp in off_anchor_stamps:
        status = timing_status(origin, stamp, ANCHORS[-1]) if stamp is not None else None
        if status == "within_window":
            return "at_least_one_in_window"
        if status == "boundary_uncertain":
            uncertain = True
        if status in (None, "different_clock_or_epoch"):
            unresolved = True
    if uncertain:
        return "boundary_uncertain_without_in_window"
    if unresolved:
        return "unresolved_without_proven_in_window"
    return "all_outside_window"


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


def acquisition_blocks(path, expected, *, expected_sha256=None):
    """Require a total, one-to-one assignment onto the exact frozen roster."""
    blocks = {}
    for key, (block,) in records(path, IDENTITY + ("acquisition_block",),
                                 expected_sha256=expected_sha256, digest_label="acquisition block map"):
        if not KEY.fullmatch(block):
            raise ValueError("invalid acquisition block label")
        if key not in expected:
            raise ValueError("acquisition block map contains non-roster episode")
        if key in blocks:
            raise ValueError("duplicate episode in acquisition block map")
        blocks[key] = block
    if len(blocks) != len(expected):
        raise ValueError("acquisition block map does not cover the frozen roster")
    return blocks


def stress_history(path, expected):
    """Validate optional redacted stress observations, not treatment assignments."""
    result = {}
    for key, history in records(path, IDENTITY + STRESS_INTERVALS):
        if key not in expected:
            raise ValueError("stress history contains a non-roster episode")
        if key in result:
            raise ValueError("duplicate episode in stress history")
        if any(state not in STRESS_STATES for state in history):
            raise ValueError("invalid stress history status")
        result[key] = history
    return result


def whole_episode_stress_status(history):
    if history is None:
        return "missing_declaration"
    if "off_schedule_bias_observed" in history:
        return "off_schedule_bias_observed_in_any_interval"
    if history == ("no_off_schedule_bias_documented",) * len(STRESS_INTERVALS):
        return "no_off_schedule_bias_documented_in_both_intervals"
    return "unknown_without_observed_off_schedule_bias"


def readout_paths(path, expected):
    paths = defaultdict(dict)
    for key, (raw_nominal, route) in records(path, IDENTITY + ("nominal_us", "readout_path")):
        nominal = integer(raw_nominal)
        if nominal not in ANCHORS:
            raise ValueError("readout path table includes an off-anchor slot")
        if key not in expected or len(expected[key].get(nominal, ())) != 1:
            raise ValueError("readout path does not identify one captured frozen anchor")
        if route and not KEY.fullmatch(route):
            raise ValueError("invalid observed readout path label")
        if nominal in paths[key]:
            raise ValueError("duplicate readout path for the same captured anchor")
        paths[key][nominal] = route
    return paths


def readout_path_status(slots, metadata):
    if any(len(slots.get(nominal, ())) != 1 for nominal in ANCHORS):
        return "missing_or_ambiguous_capture"
    known = [path for path in metadata.values() if path]
    if len(set(known)) > 1:
        return "observed_path_change"
    if len(known) == len(ANCHORS):
        return "same_declared_path_at_all_anchors"
    return "missing_or_unknown_path_without_observed_change"


def read_bias_intervals(path, expected):
    biases = defaultdict(dict)
    fields = IDENTITY + ("nominal_us", "observed_read_bias_uV", "read_bias_error_uV")
    for key, (raw_nominal, value, error) in records(path, fields):
        nominal = integer(raw_nominal)
        if nominal not in ANCHORS:
            raise ValueError("read bias table includes an off-anchor slot")
        if key not in expected or len(expected[key].get(nominal, ())) != 1:
            raise ValueError("read bias does not identify one captured frozen anchor")
        if nominal in biases[key]:
            raise ValueError("duplicate read bias for the same captured anchor")
        if (value and not SIGNED_INTEGER.fullmatch(value)) or (error and not INTEGER.fullmatch(error)):
            raise ValueError("read bias and error must be integral microvolts; error cannot be negative")
        if value and error:
            tick, bound = int(value), int(error)
            biases[key][nominal] = (tick - bound, tick + bound)
        else:
            biases[key][nominal] = None
    return biases


def read_bias_status(slots, metadata):
    if any(len(slots.get(nominal, ())) != 1 for nominal in ANCHORS):
        return "missing_or_ambiguous_capture"
    known = [interval for interval in metadata.values() if interval is not None]
    if known and max(lo for lo, _ in known) > min(hi for _, hi in known):
        return "common_reported_bias_ruled_out"
    if len(known) == len(ANCHORS):
        return "all_known_common_bias_not_ruled_out"
    return "missing_or_unknown_bias_without_known_conflict"


def validate_digest(digest, label):
    if digest is not None and (not isinstance(digest, str) or not SHA256.fullmatch(digest)):
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hexadecimal digits")


def audit(roster_path, capture_path, *, diagnose_anchors=False, roster_sha256=None,
          diagnose_dies=False, block_map_path=None, block_map_sha256=None, stress_history_path=None,
          diagnose_die_stress=False, diagnose_duplicate_anchors=False,
          diagnose_terminal_alternatives=False, capture_sha256=None, readout_path_table=None,
          read_bias_table=None):
    validate_digest(roster_sha256, "frozen roster")
    validate_digest(capture_sha256, "timestamp capture")
    validate_digest(block_map_sha256, "acquisition block map")
    if block_map_sha256 is not None and block_map_path is None:
        raise ValueError("acquisition block map SHA-256 requires an acquisition block map")
    if diagnose_die_stress and stress_history_path is None:
        raise ValueError("die stress diagnosis requires a stress history table")
    expected = {}
    for key, _ in records(roster_path, IDENTITY, expected_sha256=roster_sha256):
        if key in expected:
            raise ValueError("duplicate frozen acquisition identity")
        expected[key] = {}
    if not expected:
        raise ValueError("frozen acquisition roster must not be empty")

    blocks = (acquisition_blocks(block_map_path, expected, expected_sha256=block_map_sha256)
              if block_map_path is not None else None)
    stress = stress_history(stress_history_path, expected) if stress_history_path is not None else None
    present, unexpected = set(), set()
    off_anchor_rows = 0
    off_anchor_stamps = defaultdict(list) if diagnose_terminal_alternatives else None
    observations = records(capture_path, IDENTITY + TIMING, expected_sha256=capture_sha256,
                           digest_label="timestamp capture")
    for key, (nominal, clock, epoch, tick, error) in observations:
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
            if off_anchor_stamps is not None:
                off_anchor_stamps[key].append(stamp)
            continue
        expected[key].setdefault(nominal, []).append(stamp)

    paths = readout_paths(readout_path_table, expected) if readout_path_table is not None else None
    attempted_paths, eligible_paths = Counter(), Counter()
    biases = read_bias_intervals(read_bias_table, expected) if read_bias_table is not None else None
    attempted_bias, eligible_bias = Counter(), Counter()
    classifications = Counter()
    die_counts = defaultdict(lambda: [0, 0]) if diagnose_dies else None
    eligible_block_dies = defaultdict(set)
    eligible_die_blocks = defaultdict(set)
    stress_attempted = [Counter(), Counter()]
    stress_eligible = [Counter(), Counter()]
    whole_stress_attempted, whole_stress_eligible = Counter(), Counter()
    die_stress_attempted = defaultdict(set) if diagnose_die_stress else None
    die_stress_eligible = defaultdict(set) if diagnose_die_stress else None
    eligible_stress_dies = set() if diagnose_die_stress else None
    for key, slots in expected.items():
        reason = verdict(slots) if key in present else "missing_capture"
        classifications[reason] += 1
        if paths is not None:
            path_status = readout_path_status(slots, paths.get(key, {}))
            attempted_paths[path_status] += 1
            if reason == "eligible":
                eligible_paths[path_status] += 1
        if biases is not None:
            status = read_bias_status(slots, biases.get(key, {}))
            attempted_bias[status] += 1
            if reason == "eligible":
                eligible_bias[status] += 1
        if die_counts is not None:
            counts = die_counts[key[:3]]
            counts[0] += 1
            counts[1] += int(reason == "eligible")
        if blocks is not None and reason == "eligible":
            block, die = blocks[key], key[:3]
            eligible_block_dies[block].add(die)
            eligible_die_blocks[die].add(block)
        if stress is not None:
            history = stress.get(key)
            overall = whole_episode_stress_status(history)
            whole_stress_attempted[overall] += 1
            if reason == "eligible":
                whole_stress_eligible[overall] += 1
            if die_stress_attempted is not None:
                die = key[:3]
                die_stress_attempted[overall].add(die)
                if reason == "eligible":
                    die_stress_eligible[overall].add(die)
                    eligible_stress_dies.add(die)
            for i, state in enumerate(history or ("missing_declaration",) * len(STRESS_INTERVALS)):
                stress_attempted[i][state] += 1
                if reason == "eligible":
                    stress_eligible[i][state] += 1
    report = {
        "schema": "clotho.rram_retention_anchor_audit.v1",
        "frozen_attempted_episodes": len(expected),
        "frozen_attempted_dies": len({key[:3] for key in expected}),
        "eligible_episodes": classifications["eligible"],
        "unexpected_capture_episodes": len(unexpected),
        "nonscheduled_rows_in_frozen_episodes": off_anchor_rows,
        "frozen_episodes_by_primary_reason": dict(sorted(classifications.items())),
    }
    if capture_sha256 is not None:
        report["timestamp_capture_byte_digest_verified"] = True
    if roster_sha256 is not None:
        report["frozen_roster_byte_digest_verified"] = True
    if block_map_sha256 is not None:
        report["acquisition_block_map_byte_digest_verified"] = True
    if diagnose_anchors:
        by_anchor = {str(nominal): Counter() for nominal in ANCHORS}
        for slots in expected.values():
            for nominal, status in zip(ANCHORS, anchor_statuses(slots)):
                by_anchor[str(nominal)][status] += 1
        report["frozen_nominal_anchor_statuses_by_us"] = {
            nominal: dict(sorted(counts.items())) for nominal, counts in by_anchor.items()
        }
    if diagnose_duplicate_anchors:
        duplicates = {str(nominal): Counter() for nominal in ANCHORS}
        for slots in expected.values():
            for nominal in ANCHORS:
                rows = slots.get(nominal, ())
                if len(rows) > 1:
                    duplicates[str(nominal)][duplicate_anchor_status(rows)] += 1
        report["duplicate_frozen_anchor_metadata_by_us"] = {
            nominal: dict(sorted(counts.items())) for nominal, counts in duplicates.items()
        }
    if off_anchor_stamps is not None:
        alternatives = Counter(
            terminal_alternative_status(slots, off_anchor_stamps.get(key, ()))
            for key, slots in expected.items() if ANCHORS[-1] not in slots
        )
        report["nonscheduled_read_timing_when_nominal_terminal_absent"] = {
            "frozen_episodes_without_nominal_terminal": sum(alternatives.values()),
            "episodes_by_status": dict(sorted(alternatives.items())),
        }
    if die_counts is not None:
        complete = partial = absent = maximum = 0
        for attempted, eligible in die_counts.values():
            complete += int(eligible == attempted)
            partial += int(0 < eligible < attempted)
            absent += int(eligible == 0)
            maximum = max(maximum, eligible)
        report["declared_die_timing_support"] = {
            "dies_with_all_frozen_episodes_eligible": complete,
            "dies_with_some_but_not_all_frozen_episodes_eligible": partial,
            "dies_with_no_frozen_episodes_eligible": absent,
            "max_eligible_episodes_on_one_die": maximum,
        }
    if blocks is not None:
        report["declared_acquisition_block_timing_support"] = {
            "frozen_declared_blocks": len(set(blocks.values())),
            "blocks_with_eligible_episodes": len(eligible_block_dies),
            "blocks_with_eligible_episodes_from_multiple_dies": sum(
                len(dies) > 1 for dies in eligible_block_dies.values()
            ),
            "dies_with_eligible_episodes_in_multiple_blocks": sum(
                len(die_blocks) > 1 for die_blocks in eligible_die_blocks.values()
            ),
        }
    if stress is not None:
        report["declared_stress_history_support"] = {
            field: {
                "frozen_attempted_by_status": dict(sorted(stress_attempted[i].items())),
                "timing_eligible_by_status": dict(sorted(stress_eligible[i].items())),
            }
            for i, field in enumerate(STRESS_INTERVALS)
        }
        report["declared_whole_episode_stress_history_support"] = {
            "frozen_attempted_by_status": dict(sorted(whole_stress_attempted.items())),
            "timing_eligible_by_status": dict(sorted(whole_stress_eligible.items())),
        }
    if die_stress_attempted is not None:
        report["declared_die_stress_history_support"] = {
            "frozen_die_denominator": report["frozen_attempted_dies"],
            "timing_eligible_die_denominator": len(eligible_stress_dies),
            "frozen_dies_with_status": {
                status: len(dies) for status, dies in sorted(die_stress_attempted.items())
            },
            "timing_eligible_dies_with_status": {
                status: len(dies) for status, dies in sorted(die_stress_eligible.items())
            },
        }
    if paths is not None:
        report["declared_readout_path_support"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_paths.items())),
            "timing_eligible_by_status": dict(sorted(eligible_paths.items())),
        }
    if biases is not None:
        report["reported_read_bias_support"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_bias.items())),
            "timing_eligible_by_status": dict(sorted(eligible_bias.items())),
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
    parser.add_argument("--diagnose-dies", action="store_true",
                        help="summarize frozen timing support by declared physical die")
    parser.add_argument("--acquisition-block-map", type=Path,
                        help="exact frozen episode-to-block metadata map for aggregate support")
    parser.add_argument("--acquisition-block-map-sha256",
                        help="verify exact block map bytes against a previously trusted SHA-256")
    parser.add_argument("--stress-history", type=Path,
                        help="optional episode-keyed off-schedule electrical stress declarations")
    parser.add_argument("--diagnose-die-stress", action="store_true",
                        help="count declared dice by whole-episode stress status; requires --stress-history")
    parser.add_argument("--diagnose-duplicate-anchors", action="store_true",
                        help="describe repeated scheduled timestamp metadata without repairing it")
    parser.add_argument("--diagnose-terminal-alternatives", action="store_true",
                        help="report off-slot timing for absent terminal rows without substituting")
    parser.add_argument("--capture-sha256",
                        help="verify exact outcome-free timestamp capture against a trusted SHA-256")
    parser.add_argument("--readout-paths", type=Path,
                        help="independent captured anchor to readout path metadata")
    parser.add_argument("--read-bias", type=Path,
                        help="independent captured anchor to observed read-bias bounds")
    args = parser.parse_args()
    try:
        report = audit(args.frozen_roster, args.capture, diagnose_anchors=args.diagnose_anchors,
                       roster_sha256=args.frozen_roster_sha256, diagnose_dies=args.diagnose_dies,
                       block_map_path=args.acquisition_block_map,
                       block_map_sha256=args.acquisition_block_map_sha256,
                       stress_history_path=args.stress_history, diagnose_die_stress=args.diagnose_die_stress,
                       diagnose_duplicate_anchors=args.diagnose_duplicate_anchors,
                       diagnose_terminal_alternatives=args.diagnose_terminal_alternatives,
                       capture_sha256=args.capture_sha256, readout_path_table=args.readout_paths,
                       read_bias_table=args.read_bias)
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: input could not be read as UTF-8 CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["eligible_episodes"] != report["frozen_attempted_episodes"]
               or report["unexpected_capture_episodes"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
