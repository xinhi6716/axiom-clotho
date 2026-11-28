"""Audit declared chamber-monitor intervals, not device temperature or retention."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path


def read_csv(path, fields):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != list(fields):
            raise ValueError(f"expected columns: {','.join(fields)}")
        for row in reader:
            if None in row or any(value is None or not value.strip() for value in row.values()):
                raise ValueError(f"incomplete row at line {reader.line_num}")
            yield row


def number(token):
    value = Decimal(token)
    if not value.is_finite():
        raise ValueError("numeric fields must be finite")
    return value


def audit(plan_path, monitor_path):
    plan = {}
    for row in read_csv(plan_path, ("episode", "start_s", "end_s", "min_c", "max_c")):
        name = row["episode"]
        start, end, low, high = map(number, (row["start_s"], row["end_s"], row["min_c"], row["max_c"]))
        if name in plan or start < 0 or not start < end or low > high:
            raise ValueError("invalid or duplicate planned episode")
        plan[name] = (start, end, low, high)
    if not plan:
        raise ValueError("plan must name at least one episode")
    windows = {name: [] for name in plan}
    for row in read_csv(monitor_path, ("episode", "start_s", "end_s", "observed_min_c", "observed_max_c")):
        name = row["episode"]
        start, end, low, high = map(number, (
            row["start_s"], row["end_s"], row["observed_min_c"], row["observed_max_c"]
        ))
        if name not in plan or start < 0 or not start < end or low > high:
            raise ValueError("invalid monitor interval or unplanned episode")
        windows[name].append((start, end, low, high))
    counts = {"complete_in_band": 0, "unobserved_gap": 0, "out_of_band": 0, "gap_and_out_of_band": 0}
    for name, (start, end, low, high) in plan.items():
        cursor = start
        gap = outside_band = False
        for left, right, seen_low, seen_high in sorted(windows[name]):
            if right <= start or left >= end:
                continue
            # An extrema window straddling a plan boundary cannot be clipped:
            # its out-of-band observation might lie outside the planned window.
            if left < start or right > end:
                raise ValueError("monitor interval straddles planned window boundary")
            if left < cursor:
                raise ValueError("overlapping monitor intervals within planned window")
            gap |= left > cursor
            outside_band |= seen_low < low or seen_high > high
            cursor = right
        gap |= cursor < end
        key = ("gap_and_out_of_band" if gap and outside_band else
               "unobserved_gap" if gap else "out_of_band" if outside_band else "complete_in_band")
        counts[key] += 1
    return {"planned_episodes": len(plan), "by_monitor_status": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="frozen episode windows and allowed Celsius bands")
    parser.add_argument("--monitor", required=True, type=Path,
                        help="certified contiguous time-block extrema on the declared episode-relative clock")
    args = parser.parse_args()
    try:
        result = audit(args.plan, args.monitor)
    except (OSError, ValueError, InvalidOperation, csv.Error):
        parser.exit(2, "invalid thermal metadata: check input files and declared intervals\n")
    print(json.dumps(result, sort_keys=True))
    return int(result["by_monitor_status"]["complete_in_band"] != result["planned_episodes"])


if __name__ == "__main__":
    raise SystemExit(main())
