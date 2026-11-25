"""Replay invented programming-attempt metadata; no device interface."""

import json


def busy_time(intervals):
    """Measure the union of half-open intervals on one declared clock."""
    total = 0
    right = None
    for start, end in sorted(intervals):
        if type(start) is not int or type(end) is not int or start >= end:
            raise ValueError("An operation requires increasing integer ticks.")
        total += end - (start if right is None else max(start, min(right, end)))
        right = end if right is None else max(right, end)
    return total


def describe(attempt):
    start, stop, call, reference, pulses = attempt
    if type(start) is not int or any(x is not None and type(x) is not int for x in (stop, call, reference)):
        raise ValueError("Attempt timestamps must share the integer clock.")
    if (stop is not None and stop < start
            or call is not None and (call < start or stop is not None and call > stop)
            or reference is not None and (reference < start or call is not None and reference < call
                                          or stop is not None and reference < stop)):
        raise ValueError("Attempt timestamps are out of order.")
    intervals, writes = [], []
    for kind, lo, hi in pulses:
        if kind not in ("write", "read"):
            raise ValueError("Unsupported operation kind.")
        intervals.append((lo, hi))
        if kind == "write":
            writes.append(hi)
    occupied = busy_time(intervals)
    if any(lo < start or stop is not None and hi > stop or reference is not None and hi > reference
           for lo, hi in intervals):
        raise ValueError("Operation is outside the recorded attempt.")
    if writes and call is not None and max(writes) > call:
        raise ValueError("A programming write followed the declared decision.")
    return {
        "writes": len(writes),
        "reads": len(intervals) - len(writes),
        "waveform_work_ticks": sum(hi - lo for lo, hi in intervals),
        "busy_ticks_on_declared_clock": occupied,
        "start_to_terminal_ticks": None if stop is None else stop - start,
        "write_age_at_reference": None if not writes or reference is None else reference - max(writes),
        "declaration_age_at_reference": None if call is None or reference is None else reference - call,
    }


def fixture():
    return {
        "A": (0, 60, 60, 1060, [("write", 10, 40), ("read", 35, 45)]),
        "B": (0, 60, 60, 1060, [("write", 0, 10), ("read", 40, 50)]),
        "C": (0, None, None, None, [("read", 10, 15)]),
    }


def main():
    attempts = fixture()
    reports = {key: describe(row) for key, row in attempts.items()}
    print(json.dumps({"initiated_attempts": len(attempts), "attempts": reports}, indent=2))


if __name__ == "__main__":
    main()
