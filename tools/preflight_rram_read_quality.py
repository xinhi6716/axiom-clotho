"""Check status and range metadata for declared early/late read pairs."""
import argparse
import csv
import json
from pathlib import Path


STATUSES = {"settled", "compliance", "overrange", "settling", "unknown"}


def records(path, fields):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != list(fields):
            raise ValueError
        for row in reader:
            if None in row or any(value is None or not value.strip() for value in row.values()):
                raise ValueError
            yield row


def audit(plan_path, quality_path):
    plan = {}
    for row in records(plan_path, ("pair", "expected_range")):
        key = row["pair"]
        if key in plan or row["expected_range"] == "unknown":
            raise ValueError
        plan[key] = row["expected_range"]
    if not plan:
        raise ValueError
    reads = {}
    for row in records(quality_path, ("pair", "role", "status", "range")):
        pair, role = row["pair"], row["role"]
        if (pair not in plan or role not in {"early", "late"}
                or row["status"] not in STATUSES or (pair, role) in reads):
            raise ValueError
        reads[pair, role] = (row["status"], row["range"])

    reasons = dict.fromkeys(("missing_read", "unknown_status", "compliance", "overrange",
                             "settling", "unknown_range", "unexpected_range", "range_changed"), 0)
    complete = 0
    for pair, expected_range in plan.items():
        flags = set()
        known_ranges = set()
        for role in ("early", "late"):
            if (pair, role) not in reads:
                flags.add("missing_read")
                continue
            status, read_range = reads[pair, role]
            if status == "unknown":
                flags.add("unknown_status")
            elif status != "settled":
                flags.add(status)
            if read_range == "unknown":
                flags.add("unknown_range")
            else:
                known_ranges.add(read_range)
                if read_range != expected_range:
                    flags.add("unexpected_range")
        if len(known_ranges) > 1:
            flags.add("range_changed")
        for flag in flags:
            reasons[flag] += 1
        complete += not flags
    return {"planned_pairs": len(plan), "pairs_matching_status_and_range": complete,
            "pairs_by_reason_nonexclusive": reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Inputs must be normalized from documented native status codes. "
        "Range labels must denote the same instrument mapping in plan and logs. "
        "Matching labels do not rule out unlogged transient range changes. "
        "This check does not establish physical eligibility."
    ))
    parser.add_argument("--plan", required=True, type=Path,
                        help="CSV: pair,expected_range")
    parser.add_argument("--quality", required=True, type=Path,
                        help="CSV: pair,role,status,range; role early/late; status " +
                             "/".join(sorted(STATUSES)) + "; use unknown for missing range")
    args = parser.parse_args()
    try:
        result = audit(args.plan, args.quality)
    except (ValueError, OSError, csv.Error):
        parser.exit(2, "invalid read-quality metadata: check input schemas and pair declarations\n")
    print(json.dumps(result, sort_keys=True))
    return int(result["planned_pairs"] != result["pairs_matching_status_and_range"])


if __name__ == "__main__":
    raise SystemExit(main())
