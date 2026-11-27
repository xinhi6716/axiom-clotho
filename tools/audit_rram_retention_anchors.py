"""Audit frozen RRAM retention slots without reading conductance or fitting."""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
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
DECIMAL_TIME = re.compile(r"[0-9]{1,18}(?:\.[0-9]{1,30})?", re.ASCII)
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


def repeat_request_support(path, capture_records, *, diagnose_cross_anchor=False):
    keys = {}
    fields = IDENTITY + ("nominal_us", "capture_record", "source_request_key")
    for identity, (nominal, ordinal, request_key) in records(path, fields):
        anchor = identity, integer(nominal)
        number = integer(ordinal)
        if number not in capture_records or capture_records[number][0] != anchor:
            raise ValueError("repeat request entry does not identify the declared capture record and anchor")
        if number in keys:
            raise ValueError("duplicate repeat request declaration for a capture record")
        if not KEY.fullmatch(request_key):
            raise ValueError("invalid source request key")
        keys[number] = request_key
    groups = defaultdict(list)
    for number, (anchor, metadata) in capture_records.items():
        groups[anchor].append((keys.get(number), metadata))
    counts = Counter()
    for group in groups.values():
        if len(group) < 2:
            continue
        by_key = defaultdict(set)
        for key, metadata in group:
            if key is not None:
                by_key[key].add(metadata)
        if any(len(content) > 1 for content in by_key.values()):
            status = "same_request_key_conflicting_timing_metadata"
        elif any(key is None for key, _ in group):
            status = "missing_request_key_evidence"
        elif any(not clock or not epoch or tick is None or error is None
                 for _, (clock, epoch, tick, error) in group):
            status = "incomplete_timing_metadata"
        elif len(by_key) == 1:
            status = "same_request_key_same_timing_metadata"
        elif len(set().union(*by_key.values())) == 1:
            status = "distinct_request_keys_equal_timing_metadata"
        else:
            status = "multiple_request_keys_nonidentical_timing_metadata"
        counts[status] += 1
    report = {"repeated_frozen_anchor_groups": sum(counts.values()),
              "groups_by_status": dict(sorted(counts.items()))}
    if diagnose_cross_anchor:
        anchors_by_key = defaultdict(set)
        for number, key in keys.items():
            anchors_by_key[key].add(capture_records[number][0])
        cross = Counter()
        for anchors in anchors_by_key.values():
            if len(anchors) == 1:
                cross["one_frozen_anchor"] += 1
            else:
                identities = {episode for episode, _ in anchors}
                nominals = {nominal for _, nominal in anchors}
                status = ("multiple_anchors_within_one_episode" if len(identities) == 1 else
                          "multiple_episodes_same_nominal_anchor" if len(nominals) == 1 else
                          "multiple_episodes_multiple_nominal_anchors")
                cross[status] += 1
        report["conditional_single_anchor_request_key_support"] = {
            "keys_with_at_least_one_declared_frozen_anchor": len(anchors_by_key),
            "frozen_anchor_groups_without_full_key_evidence": sum(
                any(key is None for key, _ in group) for group in groups.values()
            ),
            "keys_by_assignment_status": dict(sorted(cross.items())),
        }
    return report


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


def planned_readout_paths(path, roster):
    plans = defaultdict(dict)
    fields = IDENTITY + ("nominal_us", "planned_readout_path")
    for key, (raw_nominal, route) in records(path, fields):
        nominal = integer(raw_nominal)
        if key not in roster or nominal not in ANCHORS:
            raise ValueError("planned readout path does not identify a frozen anchor")
        if not KEY.fullmatch(route):
            raise ValueError("invalid planned readout path label")
        if nominal in plans[key]:
            raise ValueError("duplicate planned readout path for a frozen anchor")
        plans[key][nominal] = route
    if len(plans) != len(roster) or any(len(plan) != len(ANCHORS) for plan in plans.values()):
        raise ValueError("planned readout paths must cover all anchors of the frozen roster")
    return plans


def planned_path_status(slots, observed, plan, nominals=ANCHORS):
    unresolved = False
    for nominal in nominals:
        path = observed.get(nominal)
        if len(slots.get(nominal, ())) != 1 or not path:
            unresolved = True
        elif path != plan[nominal]:
            return "logged_route_differs_from_plan"
    return "unresolved_route" if unresolved else "all_logged_routes_match_plan"


def acquisition_event_roles(path, roster, *, planned):
    roles = defaultdict(dict)
    field = "planned_event_role" if planned else "logged_event_role"
    vocabulary = {"state_write_end", "measurement_read", "stress_start", "stress_end", "scheduler_marker"}
    for key, (raw_nominal, role) in records(path, IDENTITY + ("nominal_us", field)):
        nominal = integer(raw_nominal)
        if key not in roster or nominal not in ANCHORS:
            raise ValueError("event role does not identify a frozen anchor")
        if not planned and len(roster[key].get(nominal, ())) != 1:
            raise ValueError("logged event role does not identify one captured frozen anchor")
        if role not in vocabulary and (planned or role != "unknown"):
            raise ValueError("invalid event role")
        if nominal in roles[key]:
            raise ValueError("duplicate event role for a frozen anchor")
        roles[key][nominal] = role
    if planned and (len(roles) != len(roster)
                    or any(len(plan) != len(ANCHORS) for plan in roles.values())):
        raise ValueError("planned event roles must cover all frozen anchors")
    return roles


def event_role_status(plan, evidence):
    unresolved = False
    for nominal, expected in plan.items():
        logged = evidence.get(nominal, "unknown")
        if logged == "unknown":
            unresolved = True
        elif logged != expected:
            return "logged_event_role_differs_from_plan"
    return "unresolved_event_role" if unresolved else "all_logged_event_roles_match_plan"


def native_read_identity_statuses(path, roster):
    reads_by_anchor, anchors_by_read = {}, defaultdict(set)
    fields = IDENTITY + ("nominal_us", "native_read_namespace", "native_read_id")
    for key, (raw_nominal, namespace, read_id) in records(path, fields):
        nominal = integer(raw_nominal)
        anchor = (key, nominal)
        if key not in roster or nominal not in ANCHORS or len(roster[key].get(nominal, ())) != 1:
            raise ValueError("native read identity does not identify one captured frozen anchor")
        if anchor in reads_by_anchor:
            raise ValueError("duplicate native read identity declaration for a captured anchor")
        if not KEY.fullmatch(namespace) or not KEY.fullmatch(read_id):
            raise ValueError("invalid native read namespace or identity")
        identity = (namespace, read_id)
        reads_by_anchor[anchor] = identity
        anchors_by_read[identity].add(anchor)

    conflicting = {key for anchors in anchors_by_read.values() if len(anchors) > 1
                   for key, _ in anchors}
    statuses = {}
    for key in roster:
        if key in conflicting:
            statuses[key] = "native_read_reused_across_anchor_assignments"
        elif all((key, nominal) in reads_by_anchor for nominal in ANCHORS):
            statuses[key] = "all_anchors_have_distinct_logged_native_reads"
        else:
            statuses[key] = "missing_or_ambiguous_native_read_assignment"
    return statuses, reads_by_anchor


def cross_episode_event_identity_support(orders, reads):
    """Compare counters only for independently identified distinct individual reads."""
    assignments = defaultdict(list)
    for key, ordered_anchors in orders.items():
        for nominal, event in ordered_anchors.items():
            if nominal in ANCHORS[1:]:
                assignments[event].append((key, reads.get((key, nominal))))

    support = Counter()
    for group in assignments.values():
        if len({key for key, _ in group}) < 2:
            continue
        by_namespace = defaultdict(lambda: defaultdict(set))
        for key, identity in group:
            if identity is not None:
                namespace, read_id = identity
                by_namespace[namespace][key].add(read_id)
        # Different IDs are comparable only inside a native read namespace;
        # different episodes must participate in the conflicting evidence.
        if any(len(episodes) > 1 and len(set().union(*episodes.values())) > 1
               for episodes in by_namespace.values()):
            support["same_counter_for_distinct_logged_reads"] += 1
        else:
            labels = {identity for _, identity in group}
            if len(labels) == 1 and None not in labels:
                support["same_counter_for_same_logged_read"] += 1
            else:
                support["same_counter_with_unresolved_read_identity"] += 1
    return {
        "cross_episode_shared_stream_counter_groups": sum(support.values()),
        "groups_by_status": dict(sorted(support.items())),
    }


def decimal_time_tokens(path, roster):
    result = defaultdict(dict)
    fields = IDENTITY + ("nominal_us", "token_s", "declared_resolution_s")
    for key, (raw_nominal, token, resolution) in records(path, fields):
        nominal = integer(raw_nominal)
        if key not in roster or nominal not in ANCHORS or len(roster[key].get(nominal, ())) != 1:
            raise ValueError("decimal time token does not identify one captured frozen anchor")
        if nominal in result[key]:
            raise ValueError("duplicate decimal time token for a frozen anchor")
        if not DECIMAL_TIME.fullmatch(token) or not DECIMAL_TIME.fullmatch(resolution):
            raise ValueError("decimal token and resolution require nonnegative fixed-point seconds")
        center = Fraction(token)
        half = Fraction(resolution) / 2
        result[key][nominal] = (center - half, center + half)
    return result


def decimal_time_order(evidence):
    rightmost_lower = previous_upper = None
    separated = True
    for nominal in ANCHORS:
        if nominal not in evidence:
            continue
        lo, hi = evidence[nominal]
        if rightmost_lower is not None and hi <= rightmost_lower:
            return "strict_order_ruled_out_by_declared_token_bounds"
        rightmost_lower = max(rightmost_lower, lo) if rightmost_lower is not None else lo
        if previous_upper is not None and previous_upper >= lo:
            separated = False
        previous_upper = hi
    if len(evidence) != len(ANCHORS):
        return "incomplete_decimal_time_tokens"
    return ("strict_order_certified_by_disjoint_token_bounds" if separated
            else "strict_order_unresolved_by_declared_token_bounds")


def decimal_elapsed_status(evidence, nominal):
    if ANCHORS[0] not in evidence or nominal not in evidence:
        return "missing_origin_or_read_token"
    origin_lo, origin_hi = evidence[ANCHORS[0]]
    read_lo, read_hi = evidence[nominal]
    lo, hi = (read_lo - origin_hi) * 1_000_000, (read_hi - origin_lo) * 1_000_000
    early = nominal == ANCHORS[1]
    lower, upper = (0, EARLY_UPPER) if early else (TERMINAL_LOWER, TERMINAL_UPPER)
    if (hi <= lower if early else hi < lower) or lo > upper:
        return "declared_elapsed_bounds_outside_window"
    if (lo > lower if early else lo >= lower) and hi <= upper:
        return "declared_elapsed_bounds_within_window"
    return "declared_elapsed_bounds_partially_overlap_window"


def timing_envelope_windows(path, roster):
    windows = defaultdict(dict)
    fields = IDENTITY + ("clock", "epoch", "valid_from_tick_us", "valid_through_tick_us")
    for key, (clock, epoch, start, end) in records(path, fields):
        if key not in roster:
            raise ValueError("timing envelope window contains a non-roster episode")
        if not KEY.fullmatch(clock) or not KEY.fullmatch(epoch):
            raise ValueError("timing envelope window requires exact clock and epoch labels")
        identity = clock, epoch
        if identity in windows[key]:
            raise ValueError("duplicate timing envelope window for episode clock/epoch")
        lo, hi = integer(start), integer(end)
        if lo > hi:
            raise ValueError("timing envelope window bounds must be ordered")
        windows[key][identity] = lo, hi
    return windows


def timing_envelope_status(rows, windows):
    if len(rows) != 1 or rows[0] is None:
        return "unusable_anchor_timestamp"
    clock, epoch, tick, error = rows[0]
    window = windows.get((clock, epoch))
    if window is None:
        return "no_declared_window_for_anchor_clock_epoch"
    lo, hi = window
    if tick + error < lo or tick - error > hi:
        return "timestamp_interval_outside_declared_window"
    if tick - error < lo or tick + error > hi:
        return "timestamp_interval_partially_overlaps_declared_window"
    return "timestamp_interval_within_declared_window"


def native_timer_segments(path, roster):
    segments = defaultdict(dict)
    fields = IDENTITY + ("nominal_us", "native_timer_segment")
    for key, (raw_nominal, segment) in records(path, fields):
        nominal = integer(raw_nominal)
        if key not in roster or nominal not in ANCHORS or len(roster[key].get(nominal, ())) != 1:
            raise ValueError("native timer segment does not identify one captured frozen anchor")
        if nominal in segments[key]:
            raise ValueError("duplicate native timer segment declaration at one anchor")
        if segment and not KEY.fullmatch(segment):
            raise ValueError("invalid native timer segment identity")
        segments[key][nominal] = segment
    return segments


def native_timer_segment_status(evidence):
    known = [segment for segment in evidence.values() if segment]
    if len(set(known)) > 1:
        return "multiple_logged_native_timer_segments"
    if len(known) == len(ANCHORS):
        return "all_anchors_share_logged_native_timer_segment"
    return "native_timer_segment_continuity_not_established"


def logged_event_order(path, roster):
    records_by_episode = defaultdict(dict)
    fields = IDENTITY + ("nominal_us", "ordered_stream", "event_counter")
    for key, (raw_nominal, stream, raw_counter) in records(path, fields):
        nominal = integer(raw_nominal)
        if key not in roster or nominal not in ANCHORS or len(roster[key].get(nominal, ())) != 1:
            raise ValueError("event order does not identify one captured frozen anchor")
        if nominal in records_by_episode[key]:
            raise ValueError("duplicate event order declaration at one anchor")
        if not KEY.fullmatch(stream) or not INTEGER.fullmatch(raw_counter):
            raise ValueError("event order requires a stream identity and unsigned integral counter")
        records_by_episode[key][nominal] = stream, int(raw_counter)
    return records_by_episode


def event_order_status(evidence):
    seen = {}
    for nominal in ANCHORS:
        if nominal not in evidence:
            continue
        stream, counter = evidence[nominal]
        if stream in seen and counter <= seen[stream]:
            return "logged_order_conflicts_with_anchor_sequence"
        seen[stream] = counter
    if len(evidence) == len(ANCHORS) and len(seen) == 1:
        return "all_anchor_events_in_increasing_logged_order"
    return "logged_order_not_established"


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


def planned_read_bias(path, roster):
    plans = defaultdict(dict)
    fields = IDENTITY + ("nominal_us", "planned_bias_min_uV", "planned_bias_max_uV")
    for key, (raw_nominal, lower, upper) in records(path, fields):
        nominal = integer(raw_nominal)
        if key not in roster or nominal not in ANCHORS:
            raise ValueError("planned bias does not identify a frozen anchor")
        if nominal in plans[key]:
            raise ValueError("duplicate planned bias for a frozen anchor")
        if not SIGNED_INTEGER.fullmatch(lower) or not SIGNED_INTEGER.fullmatch(upper):
            raise ValueError("planned bias bounds must be signed integral microvolts")
        lo, hi = int(lower), int(upper)
        if lo > hi:
            raise ValueError("planned bias bounds must be ordered")
        plans[key][nominal] = lo, hi
    if len(plans) != len(roster) or any(len(plan) != len(ANCHORS) for plan in plans.values()):
        raise ValueError("planned bias must cover all anchors of the frozen roster")
    return plans


def planned_bias_status(observed, plan):
    unresolved = False
    for nominal, (lo, hi) in plan.items():
        interval = observed.get(nominal)
        if interval is None:
            unresolved = True
            continue
        a, b = interval
        if b < lo or a > hi:
            return "logged_bias_interval_disjoint_from_plan"
        if a < lo or b > hi:
            unresolved = True
    return "unresolved_bias" if unresolved else "all_logged_bias_intervals_contained_in_plan"


def read_bias_status(slots, metadata):
    if any(len(slots.get(nominal, ())) != 1 for nominal in ANCHORS):
        return "missing_or_ambiguous_capture"
    known = [interval for interval in metadata.values() if interval is not None]
    if known and max(lo for lo, _ in known) > min(hi for _, hi in known):
        return "common_reported_bias_ruled_out"
    if len(known) == len(ANCHORS):
        return "all_known_common_bias_not_ruled_out"
    return "missing_or_unknown_bias_without_known_conflict"


def captured_apertures(path, expected):
    apertures = defaultdict(dict)
    fields = IDENTITY + ("nominal_us", "aperture_open_offset_us", "aperture_close_offset_us")
    for key, (raw_nominal, opened, closed) in records(path, fields):
        nominal = integer(raw_nominal)
        if nominal not in ANCHORS[1:]:
            raise ValueError("aperture table accepts only the early and terminal anchor reads")
        if key not in expected or len(expected[key].get(nominal, ())) != 1:
            raise ValueError("aperture does not identify one captured frozen anchor")
        if nominal in apertures[key]:
            raise ValueError("duplicate aperture for the same captured anchor")
        if any(value and not SIGNED_INTEGER.fullmatch(value) for value in (opened, closed)):
            raise ValueError("aperture offsets must be signed integral microseconds")
        if opened and closed:
            lower, upper = int(opened), int(closed)
            if lower > upper:
                raise ValueError("aperture opens after it closes")
            apertures[key][nominal] = lower, upper
        else:
            apertures[key][nominal] = None
    return apertures


def planned_aperture_widths(path, roster):
    plans = defaultdict(dict)
    fields = IDENTITY + ("nominal_us", "min_aperture_width_us", "max_aperture_width_us")
    for key, (raw_nominal, lower, upper) in records(path, fields):
        nominal = integer(raw_nominal)
        if key not in roster or nominal not in ANCHORS[1:]:
            raise ValueError("planned aperture width does not identify a frozen read anchor")
        if nominal in plans[key]:
            raise ValueError("duplicate planned aperture width for a frozen anchor")
        lo, hi = integer(lower), integer(upper)
        if lo > hi:
            raise ValueError("planned aperture width bounds must be ordered")
        plans[key][nominal] = lo, hi
    if len(plans) != len(roster) or any(len(plan) != len(ANCHORS[1:]) for plan in plans.values()):
        raise ValueError("planned aperture widths must cover all frozen read anchors")
    return plans


def aperture_width_status(offsets, bounds):
    if offsets is None:
        return "missing_or_unknown_aperture_width"
    width = offsets[1] - offsets[0]
    if width < bounds[0]:
        return "reported_width_below_plan"
    if width > bounds[1]:
        return "reported_width_above_plan"
    return "reported_width_within_plan"


def aperture_status(slots, nominal, offsets):
    rows = slots.get(nominal, ())
    if len(rows) != 1:
        return "missing_or_ambiguous_capture"
    if offsets is None:
        return "missing_or_unknown_aperture"
    origins = slots.get(ANCHORS[0], ())
    if len(origins) != 1 or origins[0] is None or rows[0] is None:
        return "unusable_origin_or_read_timestamp"
    origin, observation = origins[0], rows[0]
    if origin[:2] != observation[:2]:
        return "different_clock_or_epoch"

    delta = observation[2] - origin[2]
    error = observation[3] + origin[3]
    early = nominal == ANCHORS[1]
    lower, upper = (0, EARLY_UPPER) if early else (TERMINAL_LOWER, TERMINAL_UPPER)
    # Both aperture ends shift by the same timestamp error. The full
    # aperture fits only for shifts between the endpoint constraints.
    need_lower = lower - (delta + offsets[0])
    need_upper = upper - (delta + offsets[1])
    min_fit, max_fit = max(-error, need_lower), min(error, need_upper)
    if min_fit > max_fit or (early and max_fit == need_lower):
        return "full_aperture_cannot_fit"
    if ((-error > need_lower if early else -error >= need_lower)
            and error <= need_upper):
        return "full_aperture_within_window"
    return "aperture_fit_depends_on_timestamp_error"


def common_read_error_aperture_status(origin, early, terminal, metadata):
    # A common read correction must satisfy both read bounds. Subtracting
    # the shared origin correction produces one contiguous shift interval.
    span = min(early[3], terminal[3]) + origin[3]
    lower, upper, lower_open = -span, span, False
    for nominal, read in zip(ANCHORS[1:], (early, terminal)):
        strict = nominal == ANCHORS[1]
        win_lo, win_hi = (0, EARLY_UPPER) if strict else (TERMINAL_LOWER, TERMINAL_UPPER)
        delta = read[2] - origin[2]
        opened, closed = (delta + offset for offset in metadata[nominal])
        candidate_lower = win_lo - opened
        if candidate_lower > lower:
            lower, lower_open = candidate_lower, strict
        elif candidate_lower == lower:
            lower_open |= strict
        upper = min(upper, win_hi - closed)
        if lower > upper or (lower == upper and lower_open):
            return "no_joint_fit_under_common_read_error"
    if lower == -span and upper == span and not lower_open:
        return "joint_fit_for_all_errors_under_common_read_error"
    return "joint_fit_possible_under_common_read_error"


def joint_aperture_status(slots, metadata, *, common_read_error=False):
    """Exact shared-origin test conditional on a box of separate read residuals."""
    if any(len(slots.get(nominal, ())) != 1 for nominal in ANCHORS):
        return "missing_or_ambiguous_capture"
    if any(metadata.get(nominal) is None for nominal in ANCHORS[1:]):
        return "missing_or_unknown_aperture"
    origin, early, terminal = (slots[nominal][0] for nominal in ANCHORS)
    if any(stamp is None for stamp in (origin, early, terminal)):
        return "unusable_origin_or_read_timestamp"
    if not origin[:2] == early[:2] == terminal[:2]:
        return "different_clock_or_epoch"

    if common_read_error:
        return common_read_error_aperture_status(origin, early, terminal, metadata)
    lower, upper, upper_open = -origin[3], origin[3], False
    for nominal, read in zip(ANCHORS[1:], (early, terminal)):
        strict = nominal == ANCHORS[1]
        win_lo, win_hi = (0, EARLY_UPPER) if strict else (TERMINAL_LOWER, TERMINAL_UPPER)
        delta = read[2] - origin[2]
        opened, closed = (delta + offset for offset in metadata[nominal])
        if closed - opened > win_hi - win_lo or (strict and closed - opened == win_hi - win_lo):
            return "no_joint_fit_under_read_residual_box"
        # For actual elapsed endpoints [opened + x - o, closed + x - o],
        # project the read residual x in [-read_error, read_error] onto
        # allowed shared origin corrections o. The upper condition for an
        # early anchor is strict because elapsed start must be positive.
        lower = max(lower, closed - win_hi - read[3])
        candidate_upper = opened - win_lo + read[3]
        if candidate_upper < upper:
            upper, upper_open = candidate_upper, strict
        elif candidate_upper == upper:
            upper_open |= strict
        if lower > upper or (lower == upper and upper_open):
            return "no_joint_fit_under_read_residual_box"

    if all(aperture_status(slots, nominal, metadata[nominal]) == "full_aperture_within_window"
           for nominal in ANCHORS[1:]):
        return "joint_fit_for_all_errors_in_read_residual_box"
    return "joint_fit_possible_under_read_residual_box"


def commanded_slots(path, roster, *, expected_sha256=None):
    commands = defaultdict(set)
    for key, (raw_nominal,) in records(path, IDENTITY + ("nominal_us",),
                                       expected_sha256=expected_sha256, digest_label="commanded schedule"):
        nominal = integer(raw_nominal)
        if key not in roster:
            raise ValueError("commanded schedule contains a non-roster episode")
        if nominal in commands[key]:
            raise ValueError("duplicate nominal slot in commanded schedule")
        commands[key].add(nominal)
    if len(commands) != len(roster):
        raise ValueError("commanded schedule does not cover the frozen roster")
    return commands


def empty_schedule_coverage():
    return dict.fromkeys((
        "commanded_slots", "observed_rows_in_frozen_episodes",
        "missing_commanded_slots", "commanded_slots_observed_once",
        "commanded_slots_observed_more_than_once", "extra_rows_at_commanded_slots",
        "uncommanded_slots_observed", "uncommanded_rows",
        "episodes_with_exact_slot_coverage", "episodes_with_equal_row_count_but_slot_mismatch",
    ), 0)


def schedule_coverage(commands, observed, terminations=None):
    totals = Counter(empty_schedule_coverage())
    groups = defaultdict(lambda: Counter(empty_schedule_coverage()))
    for key, scheduled in commands.items():
        counts = observed.get(key, {})
        rows = sum(counts.values())
        exact_once = repeated = missing = extra = 0
        for nominal in scheduled:
            count = counts.get(nominal, 0)
            if not count:
                missing += 1
            elif count == 1:
                exact_once += 1
            else:
                repeated += 1
                extra += count - 1
        uncommanded = counts.keys() - scheduled
        exact = not (missing or repeated or uncommanded)
        episode = {
            "commanded_slots": len(scheduled),
            "observed_rows_in_frozen_episodes": rows,
            "missing_commanded_slots": missing,
            "commanded_slots_observed_once": exact_once,
            "commanded_slots_observed_more_than_once": repeated,
            "extra_rows_at_commanded_slots": extra,
            "uncommanded_slots_observed": len(uncommanded),
            "uncommanded_rows": sum(counts[nominal] for nominal in uncommanded),
            "episodes_with_exact_slot_coverage": int(exact),
            "episodes_with_equal_row_count_but_slot_mismatch": int(not exact and rows == len(scheduled)),
        }
        totals.update(episode)
        if terminations is not None:
            group = groups[terminations.get(key, "missing_declaration")]
            group.update(episode)
            group["frozen_attempted_episodes"] += 1
    return dict(totals), {status: dict(counts) for status, counts in sorted(groups.items())}


def termination_dispositions(path, roster, *, expected_sha256=None):
    allowed = {"completed", "value_threshold_stop", "instrument_abort", "operator_abort", "unknown"}
    dispositions = {}
    for key, (reason,) in records(path, IDENTITY + ("termination_disposition",),
                                  expected_sha256=expected_sha256,
                                  digest_label="termination dispositions"):
        if key not in roster:
            raise ValueError("termination table contains a non-roster episode")
        if key in dispositions:
            raise ValueError("duplicate episode in termination table")
        if reason not in allowed:
            raise ValueError("invalid termination disposition")
        dispositions[key] = reason
    return dispositions


def validate_digest(digest, label):
    if digest is not None and (not isinstance(digest, str) or not SHA256.fullmatch(digest)):
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hexadecimal digits")


def audit(roster_path, capture_path, *, diagnose_anchors=False, roster_sha256=None,
          diagnose_dies=False, block_map_path=None, block_map_sha256=None, stress_history_path=None,
          diagnose_die_stress=False, diagnose_duplicate_anchors=False,
          diagnose_terminal_alternatives=False, capture_sha256=None, readout_path_table=None,
          read_bias_table=None, aperture_table=None, diagnose_joint_aperture=False,
          diagnose_common_read_error_aperture=False, commanded_schedule_path=None,
          termination_table_path=None, commanded_schedule_sha256=None,
          termination_dispositions_sha256=None, planned_path_table=None, native_read_table=None,
          planned_bias_table=None, planned_role_table=None, logged_role_table=None,
          planned_aperture_width_table=None, event_order_table=None,
          diagnose_individual_read_counters=False, native_timer_segment_table=None, timing_envelope_table=None, decimal_time_token_table=None,
          diagnose_decimal_elapsed=False, diagnose_early_route_terminal=False, repeat_request_table=None,
          diagnose_cross_anchor_requests=False):
    if diagnose_cross_anchor_requests and repeat_request_table is None:
        raise ValueError("cross-anchor request diagnosis requires repeat request keys")
    if repeat_request_table is not None and capture_sha256 is None:
        raise ValueError("repeat request reconciliation requires an expected capture SHA-256")
    if diagnose_early_route_terminal and (planned_path_table is None or readout_path_table is None):
        raise ValueError("early route/terminal diagnosis requires both planned and observed readout paths")
    if diagnose_decimal_elapsed and decimal_time_token_table is None:
        raise ValueError("decimal elapsed diagnosis requires decimal time tokens")
    if diagnose_individual_read_counters and (event_order_table is None or native_read_table is None):
        raise ValueError("individual-read counter diagnosis requires both event order and native read identities")
    if planned_aperture_width_table is not None and aperture_table is None:
        raise ValueError("planned aperture widths require a read aperture table")
    if (planned_role_table is None) != (logged_role_table is None):
        raise ValueError("event role audit requires both planned and logged event role tables")
    if planned_bias_table is not None and read_bias_table is None:
        raise ValueError("planned bias requires observed read bias")
    if planned_path_table is not None and readout_path_table is None:
        raise ValueError("planned readout paths require observed readout paths")
    validate_digest(roster_sha256, "frozen roster")
    validate_digest(capture_sha256, "timestamp capture")
    validate_digest(block_map_sha256, "acquisition block map")
    validate_digest(commanded_schedule_sha256, "commanded schedule")
    validate_digest(termination_dispositions_sha256, "termination dispositions")
    if termination_dispositions_sha256 is not None and termination_table_path is None:
        raise ValueError("termination dispositions SHA-256 requires a termination dispositions table")
    if commanded_schedule_sha256 is not None and commanded_schedule_path is None:
        raise ValueError("commanded schedule SHA-256 requires a commanded schedule")
    if block_map_sha256 is not None and block_map_path is None:
        raise ValueError("acquisition block map SHA-256 requires an acquisition block map")
    if diagnose_die_stress and stress_history_path is None:
        raise ValueError("die stress diagnosis requires a stress history table")
    if (diagnose_joint_aperture or diagnose_common_read_error_aperture) and aperture_table is None:
        raise ValueError("joint aperture diagnosis requires a read aperture table")
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
    commands = (commanded_slots(commanded_schedule_path, expected,
                                expected_sha256=commanded_schedule_sha256)
                if commanded_schedule_path is not None else None)
    observed_slots = defaultdict(Counter) if commands is not None else None
    terminations = (termination_dispositions(termination_table_path, expected,
                                             expected_sha256=termination_dispositions_sha256)
                    if termination_table_path is not None else None)
    termination_counts = defaultdict(Counter)
    present, unexpected = set(), set()
    off_anchor_rows = 0
    off_anchor_stamps = defaultdict(list) if diagnose_terminal_alternatives else None
    observations = records(capture_path, IDENTITY + TIMING, expected_sha256=capture_sha256,
                           digest_label="timestamp capture")
    indexed_capture_records = {} if repeat_request_table is not None else None
    for ordinal, (key, (nominal, clock, epoch, tick, error)) in enumerate(observations, 1):
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
        if observed_slots is not None:
            observed_slots[key][nominal] += 1
        if nominal not in ANCHORS:
            off_anchor_rows += 1
            if off_anchor_stamps is not None:
                off_anchor_stamps[key].append(stamp)
            continue
        expected[key].setdefault(nominal, []).append(stamp)
        if indexed_capture_records is not None:
            indexed_capture_records[ordinal] = ((key, nominal), (clock, epoch, tick, error))

    plans = planned_readout_paths(planned_path_table, expected) if planned_path_table is not None else None
    attempted_plan_paths, eligible_plan_paths = Counter(), Counter()
    terminal_by_early_route = defaultdict(Counter)
    paths = readout_paths(readout_path_table, expected) if readout_path_table is not None else None
    attempted_paths, eligible_paths = Counter(), Counter()
    native_reads, native_read_assignments = (native_read_identity_statuses(native_read_table, expected)
                                               if native_read_table is not None else (None, None))
    attempted_native_reads, eligible_native_reads = Counter(), Counter()
    event_orders = logged_event_order(event_order_table, expected) if event_order_table is not None else None
    attempted_event_orders, eligible_event_orders = Counter(), Counter()
    timer_segments = (native_timer_segments(native_timer_segment_table, expected)
                      if native_timer_segment_table is not None else None)
    attempted_segments, eligible_segments = Counter(), Counter()
    decimal_tokens = (decimal_time_tokens(decimal_time_token_table, expected)
                      if decimal_time_token_table is not None else None)
    attempted_token_order, eligible_token_order = Counter(), Counter()
    attempted_decimal_elapsed = {nominal: Counter() for nominal in ANCHORS[1:]}
    eligible_decimal_elapsed = {nominal: Counter() for nominal in ANCHORS[1:]}
    envelope_windows = (timing_envelope_windows(timing_envelope_table, expected)
                        if timing_envelope_table is not None else None)
    attempted_envelopes = {nominal: Counter() for nominal in ANCHORS}
    eligible_envelopes = {nominal: Counter() for nominal in ANCHORS}
    role_plans = (acquisition_event_roles(planned_role_table, expected, planned=True)
                  if planned_role_table is not None else None)
    logged_roles = (acquisition_event_roles(logged_role_table, expected, planned=False)
                    if logged_role_table is not None else None)
    attempted_roles, eligible_roles = Counter(), Counter()
    biases = read_bias_intervals(read_bias_table, expected) if read_bias_table is not None else None
    attempted_bias, eligible_bias = Counter(), Counter()
    bias_plans = planned_read_bias(planned_bias_table, expected) if planned_bias_table is not None else None
    attempted_bias_plan, eligible_bias_plan = Counter(), Counter()
    apertures = captured_apertures(aperture_table, expected) if aperture_table is not None else None
    width_plans = (planned_aperture_widths(planned_aperture_width_table, expected)
                   if planned_aperture_width_table is not None else None)
    attempted_widths = {nominal: Counter() for nominal in ANCHORS[1:]}
    eligible_widths = {nominal: Counter() for nominal in ANCHORS[1:]}
    attempted_apertures = {nominal: Counter() for nominal in ANCHORS[1:]}
    eligible_apertures = {nominal: Counter() for nominal in ANCHORS[1:]}
    attempted_joint, eligible_joint = Counter(), Counter()
    attempted_common_error, eligible_common_error = Counter(), Counter()
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
        if terminations is not None:
            stats = termination_counts[terminations.get(key, "missing_declaration")]
            stats["frozen_attempted_episodes"] += 1
            stats["episodes_with_capture"] += int(key in present)
            stats["episodes_with_nominal_terminal_row"] += int(ANCHORS[-1] in slots)
            stats["episodes_within_terminal_time_window"] += int(
                tuple(anchor_statuses(slots))[-1] == "within_window"
            )
            stats["timing_eligible_episodes"] += int(reason == "eligible")
        if paths is not None:
            path_status = readout_path_status(slots, paths.get(key, {}))
            attempted_paths[path_status] += 1
            if reason == "eligible":
                eligible_paths[path_status] += 1
        if plans is not None:
            status = planned_path_status(slots, paths.get(key, {}), plans[key])
            attempted_plan_paths[status] += 1
            if reason == "eligible":
                eligible_plan_paths[status] += 1
        if diagnose_early_route_terminal:
            early_status = planned_path_status(slots, paths.get(key, {}), plans[key], ANCHORS[1:2])
            terminal_status = tuple(anchor_statuses(slots))[-1]
            terminal_by_early_route[early_status][terminal_status] += 1
        if native_reads is not None:
            status = native_reads[key]
            attempted_native_reads[status] += 1
            if reason == "eligible":
                eligible_native_reads[status] += 1
        if event_orders is not None:
            status = event_order_status(event_orders.get(key, {}))
            attempted_event_orders[status] += 1
            if reason == "eligible":
                eligible_event_orders[status] += 1
        if timer_segments is not None:
            status = native_timer_segment_status(timer_segments.get(key, {}))
            attempted_segments[status] += 1
            if reason == "eligible":
                eligible_segments[status] += 1
        if envelope_windows is not None:
            for nominal in ANCHORS:
                status = timing_envelope_status(slots.get(nominal, ()), envelope_windows.get(key, {}))
                attempted_envelopes[nominal][status] += 1
                if reason == "eligible":
                    eligible_envelopes[nominal][status] += 1
        if decimal_tokens is not None:
            status = decimal_time_order(decimal_tokens.get(key, {}))
            attempted_token_order[status] += 1
            if reason == "eligible":
                eligible_token_order[status] += 1
        if diagnose_decimal_elapsed:
            evidence = decimal_tokens.get(key, {})
            for nominal in ANCHORS[1:]:
                status = decimal_elapsed_status(evidence, nominal)
                attempted_decimal_elapsed[nominal][status] += 1
                if reason == "eligible":
                    eligible_decimal_elapsed[nominal][status] += 1
        if role_plans is not None:
            status = event_role_status(role_plans[key], logged_roles.get(key, {}))
            attempted_roles[status] += 1
            if reason == "eligible":
                eligible_roles[status] += 1
        if biases is not None:
            status = read_bias_status(slots, biases.get(key, {}))
            attempted_bias[status] += 1
            if reason == "eligible":
                eligible_bias[status] += 1
        if bias_plans is not None:
            status = planned_bias_status(biases.get(key, {}), bias_plans[key])
            attempted_bias_plan[status] += 1
            if reason == "eligible":
                eligible_bias_plan[status] += 1
        if width_plans is not None:
            for nominal, limits in width_plans[key].items():
                status = aperture_width_status(apertures.get(key, {}).get(nominal), limits)
                attempted_widths[nominal][status] += 1
                if reason == "eligible":
                    eligible_widths[nominal][status] += 1
        if apertures is not None:
            for nominal in ANCHORS[1:]:
                status = aperture_status(slots, nominal, apertures.get(key, {}).get(nominal))
                attempted_apertures[nominal][status] += 1
                if reason == "eligible":
                    eligible_apertures[nominal][status] += 1
            if diagnose_joint_aperture:
                status = joint_aperture_status(slots, apertures.get(key, {}))
                attempted_joint[status] += 1
                if reason == "eligible":
                    eligible_joint[status] += 1
            if diagnose_common_read_error_aperture:
                status = joint_aperture_status(slots, apertures.get(key, {}), common_read_error=True)
                attempted_common_error[status] += 1
                if reason == "eligible":
                    eligible_common_error[status] += 1
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
    if repeat_request_table is not None:
        report["declared_repeat_request_support"] = repeat_request_support(
            repeat_request_table, indexed_capture_records, diagnose_cross_anchor=diagnose_cross_anchor_requests
        )
    if capture_sha256 is not None:
        report["timestamp_capture_byte_digest_verified"] = True
    if roster_sha256 is not None:
        report["frozen_roster_byte_digest_verified"] = True
    if block_map_sha256 is not None:
        report["acquisition_block_map_byte_digest_verified"] = True
    if commanded_schedule_sha256 is not None:
        report["commanded_schedule_byte_digest_verified"] = True
    if termination_dispositions_sha256 is not None:
        report["termination_dispositions_byte_digest_verified"] = True
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
    if plans is not None:
        report["planned_readout_path_agreement"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_plan_paths.items())),
            "timing_eligible_by_status": dict(sorted(eligible_plan_paths.items())),
        }
    if diagnose_early_route_terminal:
        report["terminal_anchor_support_by_early_route_agreement"] = {
            status: {"frozen_attempted_episodes": sum(counts.values()),
                     "frozen_attempted_by_terminal_status": dict(sorted(counts.items()))}
            for status, counts in sorted(terminal_by_early_route.items())
        }
    if native_reads is not None:
        report["native_read_identity_support"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_native_reads.items())),
            "timing_eligible_by_status": dict(sorted(eligible_native_reads.items())),
        }
    if event_orders is not None:
        report["logged_event_order_support"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_event_orders.items())),
            "timing_eligible_by_status": dict(sorted(eligible_event_orders.items())),
        }
    if diagnose_individual_read_counters:
        report["conditional_individual_read_counter_support"] = cross_episode_event_identity_support(
            event_orders, native_read_assignments
        )
    if timer_segments is not None:
        report["declared_native_timer_segment_support"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_segments.items())),
            "timing_eligible_by_status": dict(sorted(eligible_segments.items())),
        }
    if envelope_windows is not None:
        report["declared_timing_envelope_support_by_us"] = {
            str(nominal): {
                "frozen_attempted_by_status": dict(sorted(attempted_envelopes[nominal].items())),
                "timing_eligible_by_status": dict(sorted(eligible_envelopes[nominal].items())),
            }
            for nominal in ANCHORS
        }
    if decimal_tokens is not None:
        report["conditional_decimal_time_token_order_support"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_token_order.items())),
            "timing_eligible_by_status": dict(sorted(eligible_token_order.items())),
        }
    if diagnose_decimal_elapsed:
        report["conditional_decimal_elapsed_window_support_by_us"] = {
            str(nominal): {
                "frozen_attempted_by_status": dict(sorted(attempted_decimal_elapsed[nominal].items())),
                "timing_eligible_by_status": dict(sorted(eligible_decimal_elapsed[nominal].items())),
            }
            for nominal in ANCHORS[1:]
        }
    if role_plans is not None:
        report["acquisition_event_role_agreement"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_roles.items())),
            "timing_eligible_by_status": dict(sorted(eligible_roles.items())),
        }
    if biases is not None:
        report["reported_read_bias_support"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_bias.items())),
            "timing_eligible_by_status": dict(sorted(eligible_bias.items())),
        }
    if bias_plans is not None:
        report["planned_read_bias_agreement"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_bias_plan.items())),
            "timing_eligible_by_status": dict(sorted(eligible_bias_plan.items())),
        }
    if apertures is not None:
        report["reported_read_aperture_support_by_us"] = {
            str(nominal): {
                "frozen_attempted_by_status": dict(sorted(attempted_apertures[nominal].items())),
                "timing_eligible_by_status": dict(sorted(eligible_apertures[nominal].items())),
            }
            for nominal in ANCHORS[1:]
        }
    if width_plans is not None:
        report["planned_read_aperture_width_support_by_us"] = {
            str(nominal): {
                "frozen_attempted_by_status": dict(sorted(attempted_widths[nominal].items())),
                "timing_eligible_by_status": dict(sorted(eligible_widths[nominal].items())),
            }
            for nominal in ANCHORS[1:]
        }
    if diagnose_joint_aperture:
        report["conditional_shared_origin_aperture_support"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_joint.items())),
            "timing_eligible_by_status": dict(sorted(eligible_joint.items())),
        }
    if diagnose_common_read_error_aperture:
        report["conditional_common_read_error_aperture_support"] = {
            "frozen_attempted_by_status": dict(sorted(attempted_common_error.items())),
            "timing_eligible_by_status": dict(sorted(eligible_common_error.items())),
        }
    if commands is not None:
        coverage, by_termination = schedule_coverage(commands, observed_slots, terminations)
        report["commanded_nominal_slot_coverage"] = coverage
        if terminations is not None:
            report["commanded_nominal_slot_coverage_by_declared_termination"] = by_termination
    if terminations is not None:
        report["declared_termination_disposition_support"] = {
            reason: dict(counts) for reason, counts in sorted(termination_counts.items())
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
    parser.add_argument("--read-apertures", type=Path,
                        help="optional observed early/terminal integration aperture offsets")
    parser.add_argument("--diagnose-joint-aperture", action="store_true",
                        help="conditional shared-origin aperture test under separate bounded residuals")
    parser.add_argument("--diagnose-common-read-error-aperture", action="store_true",
                        help="conditional joint fit if both read corrections must be equal")
    parser.add_argument("--commanded-schedule", type=Path,
                        help="independent exact nominal-slot declaration per frozen episode")
    parser.add_argument("--commanded-schedule-sha256",
                        help="verify exact plan bytes against a previously trusted SHA-256")
    parser.add_argument("--termination-dispositions", type=Path,
                        help="optional episode-keyed controller/log termination dispositions")
    parser.add_argument("--termination-dispositions-sha256",
                        help="verify exact logged disposition bytes against a trusted archive SHA-256")
    parser.add_argument("--planned-readout-paths", type=Path,
                        help="independently declared expected route for each frozen anchor")
    parser.add_argument("--native-read-identities", type=Path,
                        help="optional instrument-native individual read identity per captured anchor")
    parser.add_argument("--planned-read-bias", type=Path,
                        help="independent allowed signed bias window at each frozen anchor")
    parser.add_argument("--planned-event-roles", type=Path,
                        help="independent frozen recipe of anchor event roles")
    parser.add_argument("--logged-event-roles", type=Path,
                        help="independent acquisition log of roles at existing unique anchors")
    parser.add_argument("--planned-aperture-widths", type=Path,
                        help="independent allowed aperture durations at frozen read anchors")
    parser.add_argument("--logged-event-order", type=Path,
                        help="independent event sequence for captured anchors on monotone streams")
    parser.add_argument("--diagnose-individual-read-counters", action="store_true",
                        help="assume one stream counter per individual read, not a fan-out trigger")
    parser.add_argument("--native-timer-segments", type=Path,
                        help="independent reset-qualified physical timer identities at captured anchors")
    parser.add_argument("--timing-envelope-windows", type=Path,
                        help="independent validity windows for declared timestamp-error envelopes")
    parser.add_argument("--decimal-time-tokens", type=Path,
                        help="independent common-clock rounded time tokens and declared resolutions")
    parser.add_argument("--diagnose-decimal-elapsed", action="store_true",
                        help="conditional origin-to-read elapsed bounds under declared token intervals")
    parser.add_argument("--diagnose-early-route-terminal", action="store_true",
                        help="describe terminal support stratified by route agreement at the early anchor only")
    parser.add_argument("--repeat-request-keys", type=Path,
                        help="independent source request keys for exact capture CSV record ordinals")
    parser.add_argument("--diagnose-cross-anchor-requests", action="store_true",
                        help="assume source request keys globally identify a single frozen anchor")
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
                       read_bias_table=args.read_bias, aperture_table=args.read_apertures,
                       diagnose_joint_aperture=args.diagnose_joint_aperture,
                       diagnose_common_read_error_aperture=args.diagnose_common_read_error_aperture,
                       commanded_schedule_path=args.commanded_schedule,
                       commanded_schedule_sha256=args.commanded_schedule_sha256,
                       termination_table_path=args.termination_dispositions,
                       termination_dispositions_sha256=args.termination_dispositions_sha256,
                       planned_path_table=args.planned_readout_paths,
                       native_read_table=args.native_read_identities,
                       planned_bias_table=args.planned_read_bias,
                       planned_role_table=args.planned_event_roles, logged_role_table=args.logged_event_roles,
                       planned_aperture_width_table=args.planned_aperture_widths,
                       event_order_table=args.logged_event_order,
                       diagnose_individual_read_counters=args.diagnose_individual_read_counters,
                       native_timer_segment_table=args.native_timer_segments,
                       timing_envelope_table=args.timing_envelope_windows,
                       decimal_time_token_table=args.decimal_time_tokens,
                       diagnose_decimal_elapsed=args.diagnose_decimal_elapsed,
                       diagnose_early_route_terminal=args.diagnose_early_route_terminal,
                       repeat_request_table=args.repeat_request_keys,
                       diagnose_cross_anchor_requests=args.diagnose_cross_anchor_requests)
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: input could not be read as UTF-8 CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["eligible_episodes"] != report["frozen_attempted_episodes"]
               or report["unexpected_capture_episodes"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
