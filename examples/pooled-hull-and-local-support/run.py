"""Synthetic demonstrations; only input coordinates, no outcomes or fit."""

from groupbench.input_support import audit_numeric_pair_hull


def inside(points, query):
    return bool(audit_numeric_pair_hull(points, [query])["n_inside_or_on_hull"])


def summarize(groups, query, radius):
    def gather(skip=None):
        return [p for i, group in enumerate(groups) if i != skip for p in group]

    return (
        inside(gather(), query),
        sum(inside(group, query) for group in groups),
        sum(inside(gather(skip), query) for skip in range(len(groups))),
        sum(any(sum((x - y) ** 2 for x, y in zip(p, query)) <= radius**2 for p in group)
            for group in groups),
    )


def main():
    bridge = [[(0, 0), (0, 2)], [(0, 0), (0, 2)],
              [(2, 0), (2, 2)], [(2, 0), (2, 2)]]
    far = [[(-d, 0), (d, 0)] for d in [10, 20, 30]]
    near = [[(-d, 0), (d, 0)] for d in [0.125, 0.25, 0.5]]
    fixtures = (
        ("four groups across an empty middle", bridge, (1, 1), (True, 0, 4, 0)),
        ("two edge groups", bridge[::2], (1, 1), (True, 0, 0, 0)),
        ("twenty copies within one group", [bridge[0] * 20, *bridge[1:]], (1, 1), (True, 0, 4, 0)),
        ("far points on both sides", far, (0, 0), (True, 3, 3, 0)),
        ("near points on both sides", near, (0, 0), (True, 3, 3, 3)),
    )
    for name, groups, query, expected in fixtures:
        found = summarize(groups, query, radius=1)
        if found != expected:
            raise AssertionError((name, found))
        print(f"{name}: {found}")


if __name__ == "__main__":
    main()
