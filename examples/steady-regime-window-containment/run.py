"""Exact section containment under synthetic command-response and registration uncertainty."""
from itertools import product
import json


def contains(outer, inner):
    return outer[0] <= inner[0] <= inner[1] <= outer[1]


def fixture():
    # Synthetic path s(t)=2t. Delays are independently bounded on each edge.
    regimes = sorted({
        (2 * (2 + on_delay), 2 * (10 + off_delay))
        for on_delay, off_delay in product((1, 3), (0, 2))
    })
    common = [max(lo for lo, _ in regimes), min(hi for _, hi in regimes)]
    sections = {"near_start": (10, 12), "on_boundary": (11, 12),
                "interior": (13, 17), "near_end": (19, 20)}
    checks = {}
    for name, (lo, hi) in sections.items():
        envelope = [lo - 1, hi + 1]
        worlds = [(r, [lo + shift, hi + shift])
                  for r, shift in product(regimes, (-1, 1))]
        checks[name] = {
            "nominal_center_passes_nominal_regime": 16 <= lo + hi <= 44,
            "registered_envelope": envelope,
            "certified_whole_section": contains(common, envelope),
            "compatible_failure_worlds": sum(not contains(r, w) for r, w in worlds),
        }
    return {"certified_common_regime": common, "section_checks": checks}


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
