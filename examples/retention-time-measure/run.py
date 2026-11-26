"""Exact scoring of toy piecewise-constant losses under two target time laws."""
import json
from fractions import Fraction


def report():
    edges = (1, 10, 100)
    losses = {"A": (0, 2), "B": (3, 0)}
    clock_widths = tuple(b - a for a, b in zip(edges, edges[1:]))
    assert all(b == 10 * a for a, b in zip(edges, edges[1:]))
    log_widths = (1, 1)  # Exact widths in log10(seconds).
    return {
        name: {
            model: str(Fraction(sum(w * loss for w, loss in zip(widths, values)), sum(widths)))
            for model, values in losses.items()
        }
        for name, widths in (("log_uniform", log_widths), ("clock_uniform", clock_widths))
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
