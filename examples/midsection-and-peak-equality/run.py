"""Distinguish midpoint, mean of local peaks, and peak of the mean."""
from fractions import Fraction
import json


def endpoints(profiles, planned, midpoint):
    if not planned or len(set(planned)) != len(planned) or midpoint not in planned:
        raise ValueError("invalid section roster")
    if not profiles.keys() <= set(planned):
        raise ValueError("unexpected section")
    width = {len(y) for y in profiles.values()}
    if len(width) > 1 or 0 in width:
        raise ValueError("profiles need a common registered grid")
    missing = [key for key in planned if key not in profiles]
    out = dict(
        missing=missing, midpoint_max=max(profiles[midpoint]) if midpoint in profiles else None,
        mean_section_max=None, max_mean_profile=None, common_argmax=None,
    )
    if missing:
        return out

    grid_size, = width
    maxima = [max(profiles[k]) for k in planned]
    averages = [
        Fraction(sum(profiles[k][j] for k in planned), len(planned))
        for j in range(grid_size)
    ]
    both = [
        j for j in range(grid_size)
        if all(profiles[k][j] == m for k, m in zip(planned, maxima))
    ]
    out.update(mean_section_max=str(Fraction(sum(maxima), len(planned))),
               max_mean_profile=str(max(averages)), common_argmax=both)
    return out


def fixture():
    planned = ("left", "middle", "right")
    shared = {"left": (4, 4, 0), "middle": (0, 2, 0), "right": (0, 4, 4)}
    separate = {**shared, "right": (0, 0, 4)}
    return {
        "shared_tied_peak": endpoints(shared, planned, "middle"),
        "disjoint_peaks": endpoints(separate, planned, "middle"),
        "missing_right_section": endpoints({k: shared[k] for k in planned[:2]},
                                           planned, "middle"),
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
