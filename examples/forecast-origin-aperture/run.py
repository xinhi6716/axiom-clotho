"""A center timestamp cannot certify a forecast-origin feature."""
from fractions import Fraction as Q
import json


def usable(window, ready, origin):
    start, end = window
    return start < end <= ready <= origin


def fixture():
    # Equal-duration bins on [-2,-1), [-1,0), [0,1), [1,2).
    # Forecast origin is 0. The first two bins are identical in both worlds.
    worlds = ((0, 0, 0, 0), (0, 0, 2, 2))
    narrow = (-2, 0)
    centered = (-1, 1)
    origin = 0
    rows = []
    for bins in worlds:
        y = bins[-1] // 2
        # Two prespecified rules; predictions are neither fitted to nor
        # recalibrated on the synthetic external roster.
        bad = Q(bins[1] + bins[2], 2)
        guarded = Q(sum(bins[:2]), 2) if usable(narrow, 0, origin) else Q(1, 2)
        # Without any known predictive relation the neutral rule is 1/2.
        # The past-only rule and neutral comparator remain distinct.
        rows.append((y, bad, guarded, Q(1, 2)))

    return {
        "center_passes_timestamp_filter": Q(sum(centered), 2) <= origin,
        "crossing_aperture_usable": usable(centered, 1, origin),
        "closed_past_aperture_usable": usable(narrow, 0, origin),
        "same_past_aperture_late_ready_usable": usable(narrow, 1, origin),
        "held_outcomes": [row[0] for row in rows],
        "held_mean_brier": {
            name: str(sum((r[column] - r[0]) ** 2 for r in rows) / len(rows))
            for column, name in ((1, "invalid_centered"), (2, "guarded_past"),
                                 (3, "fixed_neutral"))
        },
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
