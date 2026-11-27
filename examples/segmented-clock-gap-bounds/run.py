"""Propagate certified elapsed-time gaps without inventing nominal identities."""
import json


def evaluate(origin, bridges, local_read, band, nominal_tag_verified):
    """Bounds are closed; each bridge is (previous local ending, gap interval)."""
    if local_read < 0 or band[0] > band[1]:
        raise ValueError("invalid local read or target band")
    lo, hi = origin if origin is not None else (None, None)
    if origin is not None and not 0 <= lo <= hi:
        raise ValueError("invalid program-origin interval")
    for ending, gap in bridges:
        if ending < 0 or (gap is not None and not 0 <= gap[0] <= gap[1]):
            raise ValueError("invalid segment ending or gap")
        if lo is not None:
            lo += ending + (gap[0] if gap is not None else 0)
        if hi is not None:
            hi = hi + ending + gap[1] if gap is not None else None
    lower = None if lo is None else lo + local_read
    upper = None if hi is None else hi + local_read
    left, right = band
    if (lower is not None and lower > right) or (upper is not None and upper < left):
        state = "outside"
    elif lower is not None and upper is not None and left <= lower <= upper <= right:
        state = "inside"
    else:
        state = "unresolved"
    return {"elapsed_bounds": [lower, upper], "elapsed_gate": state,
            "nominal_gate": nominal_tag_verified,
            "both_gates": state == "inside" and nominal_tag_verified}


def fixture():
    def run(gap, nominal=False, origin=(0, 0)):
        return evaluate(origin, [(8, gap)], 3, (13, 15), nominal)
    return {
        "unknown_gap": run(None),
        "bounded_but_crosses": run((1, 4), True),
        "bounded_elapsed_no_nominal": run((2, 4)),
        "both_verified": run((2, 4), True),
        "outside_proven_even_without_upper": evaluate((0, 0), [(14, None)], 3, (13, 15), True),
        "unverified_program_origin": run((2, 4), True, None),
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
