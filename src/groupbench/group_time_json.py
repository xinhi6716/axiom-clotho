"""Strict, bounded JSON transport for metadata-only whole-group time plans.

No features, targets, model execution or file writing. The pure JSON API is
separate from evaluated/predicted bundle schemas; see GROUP_FORWARD_WINDOW_PLAN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .group_time_plan import (
    MAX_TICK, MIN_TICK, GroupSpan, TimeWindow, plan_group_forward_windows,
)

METADATA_SCHEMA_ID = "mlcalc.group_forward_window_metadata"
PLAN_SCHEMA_ID = "mlcalc.group_forward_window_plan"
SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 16
MAX_ROWS = 20_000
MAX_GROUPS = 20_000
MAX_WINDOWS = 1_000
MAX_WORK = 250_000
_ROOT_KEYS = frozenset({"schema_id", "schema_version", "time_unit", "clock_id", "gap",
                        "row_groups", "group_spans", "windows"})
_SPAN_KEYS = frozenset({"group", "start", "end", "label_available_at"})
_WINDOW_KEYS = frozenset({"lower", "upper"})
_LIMITATIONS = (
    "Metadata-only membership planning: zero fits and no scoring, selection or temporal-ML guarantee.",
    "Clock/unit labels, spans, label availability, source truth and relationships among separate group identities are declarations, not independently verified facts.",
    "Partial coverage is intentional: warmup, crossing and late groups can remain untested; later training participation does not count as test coverage.",
    "The planner does not inspect features or enforce downstream feature exclusion or train-only preprocessing. A separately reviewed modeling workflow must do that work.",
    "Output retains supplied group identities and timing metadata; review it before sharing. Row positions are meaningful only against the caller-retained row mapping.",
)


class GroupTimeJSONError(ValueError):
    """A categorical transport/planning rejection without source values or paths."""


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":"))


def _integer(token):
    # Check before conversion, including on Python versions without int limits.
    if len(token.lstrip("-")) > 19:
        raise GroupTimeJSONError("Integer tokens must fit signed-64-bit integers.")
    value = int(token)
    if not MIN_TICK <= value <= MAX_TICK:
        raise GroupTimeJSONError("Integer tokens must fit signed-64-bit integers.")
    return value


def _not_integer(_token):
    raise GroupTimeJSONError("Decimal, exponent and nonfinite numeric tokens are not accepted.")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GroupTimeJSONError("Duplicate JSON object keys are not accepted.")
        result[key] = value
    return result


def _check_depth(text):
    depth, in_string, escaped = 0, False, False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise GroupTimeJSONError("Metadata exceeds the container nesting limit.")
        elif char in "]}":
            depth -= 1
    # All remaining syntax, including unmatched containers, is checked by loads.


def _keys(value, expected):
    if type(value) is not dict or value.keys() != expected:
        raise GroupTimeJSONError("Metadata objects require exactly the declared fields.")


def _array(value, maximum):
    if type(value) is not list or not 1 <= len(value) <= maximum:
        raise GroupTimeJSONError("Metadata arrays must be nonempty and within their size limits.")


def _tick(value):
    if type(value) is not int or not MIN_TICK <= value <= MAX_TICK:
        raise GroupTimeJSONError("Ticks and gap must be signed-64-bit integer values, not Booleans.")


def _identity(value):
    if type(value) is str and value.strip():
        return
    if type(value) is int and MIN_TICK <= value <= MAX_TICK:
        return
    raise GroupTimeJSONError("Groups must be nonblank strings or signed-64-bit integer identities.")


def _metadata(text):
    if type(text) is not str:
        raise GroupTimeJSONError("Metadata must be a UTF-8 JSON text document.")
    try:
        if len(text) > MAX_INPUT_BYTES or len(text.encode("utf-8")) > MAX_INPUT_BYTES:
            raise GroupTimeJSONError("Metadata exceeds the input byte limit.")
    except UnicodeError:
        raise GroupTimeJSONError("Metadata must be a UTF-8 JSON text document.") from None
    if text.startswith("\ufeff"):
        raise GroupTimeJSONError("Metadata must not have a byte-order mark.")
    _check_depth(text)
    try:
        value = json.loads(text, object_pairs_hook=_object, parse_int=_integer,
                           parse_float=_not_integer, parse_constant=_not_integer)
    except (ValueError, RecursionError) as error:
        if isinstance(error, GroupTimeJSONError):
            raise
        raise GroupTimeJSONError("Metadata must be one complete valid JSON document.") from None
    _keys(value, _ROOT_KEYS)
    if (type(value["schema_id"]) is not str or value["schema_id"] != METADATA_SCHEMA_ID
            or type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION):
        raise GroupTimeJSONError("Metadata schema identity or version is unsupported.")
    for name in ("time_unit", "clock_id"):
        label = value[name]
        if type(label) is not str or not label or label.strip() != label:
            raise GroupTimeJSONError("Clock and unit declarations must be nonblank unpadded strings.")
    _tick(value["gap"])
    _array(value["row_groups"], MAX_ROWS)
    _array(value["group_spans"], MAX_GROUPS)
    _array(value["windows"], MAX_WINDOWS)
    for group in value["row_groups"]:
        _identity(group)
    for span in value["group_spans"]:
        _keys(span, _SPAN_KEYS)
        _identity(span["group"])
        for key in ("start", "end", "label_available_at"):
            _tick(span[key])
    for window in value["windows"]:
        _keys(window, _WINDOW_KEYS)
        _tick(window["lower"])
        _tick(window["upper"])
    return value


def _output_bound(metadata):
    """Conservative ASCII bytes for the exact serializer below, incl. newline.

    For valid plans train/test/unused partition rows and groups: training implies
    end <= label_available_at + gap < lower, whereas testing implies start >=
    lower. Thus each group appears once in membership arrays and once in
    exclusions per fold. Outside folds, group identity appears once in spans,
    once in total and once across tested/untested (4 copies allowed below).
    Row indexes need <=5 digits; signed ticks <=20 characters. 256 bytes/group
    covers span keys/ticks or exclusion keys/reason strings; 512 bytes/fold
    covers fixed keys, window ticks and membership counts. 8192 covers fixed
    schema/limitation text and top-level memberships. Array separators are
    included in per-row/group allowances. Strings use JSON escaping, not raw
    UTF-8 length. Keep this proof and bound tests in sync with the serializer.
    """
    rows, spans, windows = metadata["row_groups"], metadata["group_spans"], metadata["windows"]
    rbytes = sum(len(_json(group)) for group in rows)
    gbytes = sum(len(_json(span["group"])) for span in spans)
    dbytes = len(_json(metadata["time_unit"])) + len(_json(metadata["clock_id"]))
    r, g, w = len(rows), len(spans), len(windows)
    return (8192 + 32 * r + 256 * g + rbytes + 4 * gbytes + dbytes
            + w * (512 + 16 * r + 256 * g + 2 * gbytes))


def _preflight(metadata):
    if len(metadata["windows"]) * (len(metadata["row_groups"]) + len(metadata["group_spans"])) > MAX_WORK:
        raise GroupTimeJSONError("Metadata exceeds the window-by-row/group work limit.")
    if _output_bound(metadata) > MAX_OUTPUT_BYTES:
        raise GroupTimeJSONError("Metadata exceeds the conservative serialized-output bound.")


def _membership(value):
    return {"rows": list(value.rows), "groups": list(value.groups),
            "n_rows": value.n_rows, "n_groups": value.n_groups}


def _document(plan):
    # Explicit fields: future core dataclass changes must not widen this schema.
    return {
        "schema_id": PLAN_SCHEMA_ID, "schema_version": SCHEMA_VERSION,
        "protocol": plan.protocol, "fits_performed": plan.fits_performed,
        "coverage_mode": plan.coverage_mode, "time_unit": plan.time_unit,
        "clock_id": plan.clock_id, "gap": plan.gap, "row_groups": list(plan.row_groups),
        "group_spans": [{"group": span.group, "start": span.start, "end": span.end,
                         "label_available_at": span.label_available_at} for span in plan.group_spans],
        "folds": [{"index": fold.index,
                   "window": {"lower": fold.window.lower, "upper": fold.window.upper},
                   "train": _membership(fold.train), "test": _membership(fold.test),
                   "unused": _membership(fold.unused),
                   "exclusions": [{"group": reason.group, "train": reason.train, "test": reason.test}
                                  for reason in fold.exclusions]} for fold in plan.folds],
        "total": _membership(plan.total), "tested": _membership(plan.tested),
        "untested": _membership(plan.untested), "limitations": list(_LIMITATIONS),
    }


def plan_group_forward_windows_json(text: str) -> str:
    """Return one complete bounded JSON plan plus newline, or raise ValueError.

    Pure text-to-text, no I/O or fitting. Integers and decoded string identities
    are preserved exactly. Conservative preflight may reject outputs that would
    actually be smaller. Consumers need integer-preserving JSON decoding.
    """
    metadata = _metadata(text)
    _preflight(metadata)
    try:
        plan = plan_group_forward_windows(
            metadata["row_groups"],
            group_spans=[GroupSpan(**span) for span in metadata["group_spans"]],
            windows=[TimeWindow(**window) for window in metadata["windows"]],
            time_unit=metadata["time_unit"], clock_id=metadata["clock_id"], gap=metadata["gap"],
        )
        output = _json(_document(plan)) + "\n"
        if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise GroupTimeJSONError("Complete plan exceeds the serialized-output byte limit.")
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise GroupTimeJSONError("Metadata does not define a supported complete plan within output limits.") from None
    return output


def _read_metadata(path):
    # Reuse only this bounded regular-file byte reader, never its CSV semantics.
    # It follows explicit local symlinks and checks retained bytes/signatures.
    from .saved_scoring import _read_outcomes
    try:
        raw, _digest = _read_outcomes(path)
        return raw.decode("utf-8")
    except (OSError, ValueError):
        raise GroupTimeJSONError("Metadata must be an available bounded UTF-8 regular local file.") from None


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        self.print_usage(sys.stderr)
        self.exit(2, "group forward-window plan rejected: expected one metadata file; output and model options are not supported.\n")


def main(argv=None):
    parser = _Parser(
        prog="mlcalc plan-group-forward-windows", allow_abbrev=False,
        description="Plan whole-group forward windows from strict declared metadata; stdout JSON only, zero fits.",
        epilog="No output/overwrite options. Local file symlinks are allowed. Shell redirection can overwrite files before this command starts. Exit 0 planned, 2 rejected.",
    )
    parser.add_argument("metadata", type=Path, help="Strict v1 metadata JSON in a bounded regular local file")
    args = parser.parse_args(argv)
    try:
        output = plan_group_forward_windows_json(_read_metadata(args.metadata))
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        print("group forward-window plan rejected: invalid, unavailable or excessive metadata; no plan was written.", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(output)
    except OSError:
        print("group forward-window plan failed: stdout could not be written completely.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
