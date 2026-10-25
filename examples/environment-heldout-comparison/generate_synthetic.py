#!/usr/bin/env python3
"""Original invented proxy-reversal fixture, not measured semiconductor data."""
import csv
import io
import json
from pathlib import Path


def csv_bytes(columns, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode()


def fixtures():
    train, future, outcomes = [], [], []
    for environment in ("invented_process_A", "invented_process_B"):
        for k in range(8):
            x = k + 1
            y = 3 + 0.2 * x
            train.append({"device_id": environment + "_D" + str(k), "dependency_block": environment + "_block" + str(k // 2),
                          "environment": environment, "target": format(y, ".12g"), "preprocess_setting": str(x),
                          "proxy_readout": format(y + (0.01 if k % 2 else -0.01), ".12g")})
    for k in range(4):
        x = 1.5 + 2 * k
        y = 3 + 0.2 * x
        identity = "invented_process_C_D" + str(k)
        future.append({"device_id": identity, "dependency_block": "invented_process_C_block" + str(k // 2),
                       "environment": "invented_process_C", "preprocess_setting": format(x, ".12g"),
                       "proxy_readout": format(7.8 - y, ".12g")})
        outcomes.append({"device_id": identity, "target": format(y, ".12g")})
    spec = {"schema_version": "environment-feature-probe-v1", "heldout_environment": "invented_process_C",
            "features_a": ["preprocess_setting", "proxy_readout"], "features_b": ["preprocess_setting"],
            "ridge_alpha": 1.0, "shuffle_seed": 7, "shuffle_scope": "within_training_environment", "target_unit": "V"}
    return {"training.csv": csv_bytes(["device_id", "dependency_block", "environment", "target", "preprocess_setting", "proxy_readout"], train),
            "future.csv": csv_bytes(["device_id", "dependency_block", "environment", "preprocess_setting", "proxy_readout"], future),
            "outcomes.csv": csv_bytes(["device_id", "target"], outcomes), "spec.json": (json.dumps(spec, indent=2) + "\n").encode()}


if __name__ == "__main__":
    # Regeneration is deliberately explicit and exclusive, not a user data reader.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(mode=0o700)
    for name, blob in fixtures().items():
        with (args.output_directory / name).open("xb") as stream:
            stream.write(blob)
