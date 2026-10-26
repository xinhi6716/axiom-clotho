"""An invented grouped fixture; current units are illustrative, not device physics."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261026
ROWS = 384
TRAIN_GROUPS = tuple(range(8))
TEST_GROUPS = tuple(range(8, 12))


def generate() -> pd.DataFrame:
    """Keep the draw and arithmetic order fixed for this exposed example."""
    rng = np.random.default_rng(SEED)
    groups = np.repeat(np.arange(12), 32)
    u = rng.uniform(0, 1, ROWS)
    offsets = rng.normal(0, 0.10, 12)
    epsilon = rng.normal(0, 0.05, ROWS)
    current_A = 1e-9 * (1 + 9 * u)
    current_nA = current_A * 1e9
    response = 2 + 3 * u + 0.7 * np.sin(6 * np.pi * u) + offsets[groups] + epsilon
    return pd.DataFrame({
        "row_id": np.arange(ROWS), "group": groups,
        "split": np.where(groups < 8, "train", "test"),
        "current_A": current_A, "current_nA": current_nA, "y": response,
    })


def write_fixture(frame: pd.DataFrame, path: Path) -> None:
    """Create, never replace, the requested CSV."""
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, float_format="%.17g")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New fixture CSV; no overwrite")
    args = parser.parse_args()
    try:
        # Validate output before generating even this small synthetic fixture.
        with args.output.open("x", encoding="utf-8", newline="") as stream:
            generate().to_csv(stream, index=False, float_format="%.17g")
    except OSError:
        parser.exit(2, "Cannot create fixture: choose a new path with an existing writable parent.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
