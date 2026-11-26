"""Outcome-free crossed-roster feasibility witness on invented identities."""
import json
from itertools import combinations

from groupbench.splitting import plan_crossed_holdout


def summarize(edges):
    labels = range(4)
    rows = [(f"{d}:{b}", str(d), str(b)) for d, b in edges]
    # A roster can have valid singleton quadrants while still requiring dropped
    # rows to partition two held dies and two held acquisition blocks.
    singleton = [plan_crossed_holdout(rows, str(d), str(b)) for d, b in edges]
    assert all(tuple(map(len, (p.fit, p.new_recipe, p.new_device, p.both))) == (5, 1, 1, 1)
               for p in singleton)
    candidates = []
    for dies in combinations(labels, 2):
        for blocks in combinations(labels, 2):
            counts = [0] * 4
            for d, b in edges:
                counts[2 * (d in dies) + (b in blocks)] += 1
            if counts[0] and counts[3]:
                candidates.append((counts[1] + counts[2], *counts))
    best = min(candidates)
    return {"singleton_partitions": len(singleton),
            "minimum_excluded_side_rows": best[0], "one_optimal_quadrant_count": best[1:]}


def main():
    cycle = [(d, b) for d in range(4) for b in (d, (d + 1) % 4)]
    separate = [(d, b) for d in range(4) for b in range(4) if d // 2 == b // 2]
    for edges in (cycle, separate):
        for axis in (0, 1):
            assert [sum(e[axis] == label for e in edges) for label in range(4)] == [2] * 4
    result = {"connected_cycle": summarize(cycle), "two_components": summarize(separate)}
    assert result["connected_cycle"]["minimum_excluded_side_rows"] == 2
    assert result["two_components"]["minimum_excluded_side_rows"] == 0
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
