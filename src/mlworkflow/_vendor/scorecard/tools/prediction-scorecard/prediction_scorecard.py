#!/usr/bin/env python3
"""Audit regression predictions from CSV, including optional equal-group summaries."""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


class InputError(ValueError):
    """Invalid observations or an unsupported numerical range."""


def finite_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{label}: expected a number") from exc
    if not math.isfinite(number):
        raise InputError(f"{label}: NaN and infinity are not allowed")
    return number


def load_rows(path, id_column="id", true_column="y_true", pred_column="y_pred", group_column=None, *, numeric_parser=None):
    if group_column is not None and (not isinstance(group_column, str) or not group_column.strip()):
        raise InputError("group column name must be nonblank; omit --group-column for an ungrouped scorecard")
    # Optional internal hook for companions; legacy parsing remains unchanged.
    parse_number = finite_number if numeric_parser is None else numeric_parser
    selected = [id_column, true_column, pred_column] + ([group_column] if group_column else [])
    if len(set(selected)) != len(selected):
        raise InputError("ID, target, prediction and group columns must be distinct")
    rows, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames
        if not headers or any(not name or not name.strip() for name in headers):
            raise InputError("CSV must have nonblank column names")
        if len(headers) != len(set(headers)):
            raise InputError("CSV column names must be unique")
        missing = set(selected) - set(headers)
        if missing:
            raise InputError("missing columns: " + ", ".join(sorted(missing)))
        for raw in reader:
            line = reader.line_num
            if None in raw or any(value is None for value in raw.values()):
                raise InputError(f"line {line}: row width does not match header")
            identifier = raw[id_column].strip()
            if not identifier:
                raise InputError(f"line {line}: blank ID")
            if identifier in seen:
                raise InputError(f"line {line}: duplicate ID {identifier!r}")
            seen.add(identifier)
            group = raw[group_column].strip() if group_column else None
            if group_column and not group:
                raise InputError(f"line {line}: blank group")
            rows.append({
                "id": identifier,
                "y_true": parse_number(raw[true_column], f"line {line}, {true_column}"),
                "y_pred": parse_number(raw[pred_column], f"line {line}, {pred_column}"),
                "group": group,
            })
    if not rows:
        raise InputError("CSV contains no observations")
    return rows


def stable_mean(values):
    # Sum first to retain representable subnormal means; scale only on overflow.
    try:
        return math.fsum(values) / len(values)
    except OverflowError:
        scale = max(abs(value) for value in values)
        return scale * (math.fsum(value / scale for value in values) / len(values))


def stable_rms(values):
    scale = max(abs(value) for value in values)
    return 0.0 if scale == 0 else scale * math.sqrt(math.fsum((value / scale) ** 2 for value in values) / len(values))


def metrics(rows):
    if not rows:
        raise InputError("cannot score an empty sample")
    truth = [row["y_true"] for row in rows]
    errors = [row["y_pred"] - row["y_true"] for row in rows]
    if not all(math.isfinite(value) for value in errors + truth):
        raise InputError("numeric range too large for finite prediction errors; rescale values")
    try:
        rmse = stable_rms(errors)
        result = {"n": len(rows), "mae": stable_mean([abs(error) for error in errors]),
                  "rmse": rmse, "mean_error": stable_mean(errors), "r2": None}
        if len(rows) < 2:
            result["r2_status"] = "undefined_fewer_than_two_rows"
        elif all(value == truth[0] for value in truth):
            result["r2_status"] = "undefined_constant_target"
        else:
            # Center relative to an observed value before taking the mean. This
            # preserves small differences on top of a large common offset. Scale
            # before averaging to retain subnormal target spans as well.
            offsets = [value - truth[0] for value in truth]
            if not all(math.isfinite(value) for value in offsets):
                raise InputError("numeric range too large for target deviations; rescale values")
            target_scale = max(abs(value) for value in offsets)
            normalized = [value / target_scale for value in offsets]
            center = stable_mean(normalized)
            target_variance = stable_mean([(value - center) ** 2 for value in normalized])
            error_scale = max(abs(value) for value in errors)
            if error_scale == 0:
                result["r2"] = 1.0
            else:
                error_variance = stable_mean([(value / error_scale) ** 2 for value in errors])
                result["r2"] = 1.0 - (error_scale / target_scale) ** 2 * (error_variance / target_variance)
            result["r2_status"] = "defined"
        if not all(value is None or not isinstance(value, float) or math.isfinite(value) for value in result.values()):
            raise InputError("metric overflows; rescale values")
        return result
    except (OverflowError, ZeroDivisionError) as exc:
        raise InputError("metric overflows; rescale values") from exc


def score(rows, grouped=False):
    result = {
        "schema_version": 1,
        "overall": metrics(rows),
        "notes": [
            "Descriptive scores only: no uncertainty, independence or causal claims.",
            "Overall metrics weight rows; equal-group means weight each group equally.",
            "R2 is null for fewer than two rows or a constant target, even for perfect predictions.",
        ],
    }
    if grouped:
        buckets = defaultdict(list)
        for row in rows:
            buckets[row["group"]].append(row)
        groups = [{"group": group, **metrics(buckets[group])} for group in sorted(buckets)]
        defined = [group["r2"] for group in groups if group["r2"] is not None]
        result["groups"] = groups
        result["equal_group_summary"] = {
            "group_count": len(groups),
            "mae_mean": stable_mean([group["mae"] for group in groups]),
            "rmse_mean": stable_mean([group["rmse"] for group in groups]),
            "mean_error_mean": stable_mean([group["mean_error"] for group in groups]),
            "r2_mean": stable_mean(defined) if len(defined) == len(groups) else None,
            "r2_defined_groups": len(defined),
            "r2_status": "defined" if len(defined) == len(groups) else "undefined_for_one_or_more_groups",
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="UTF-8 CSV with unique IDs and finite numeric values")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--true-column", default="y_true")
    parser.add_argument("--pred-column", default="y_pred")
    parser.add_argument("--group-column", help="optional subject/site/batch column for descriptive group metrics")
    args = parser.parse_args(argv)
    try:
        rows = load_rows(args.csv_file, args.id_column, args.true_column, args.pred_column, args.group_column)
        result = score(rows, grouped=args.group_column is not None)
    except (InputError, OSError, UnicodeError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
