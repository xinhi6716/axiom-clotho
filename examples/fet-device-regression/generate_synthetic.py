#!/usr/bin/env python3
"""Generate wholly invented one-branch/device FET regression interface fixtures.

This arithmetic is not a physical simulator or a calibrated measurement model.
The endpoint constants are supplied separately, never calculated from features.
No randomness, timestamps, downloads, target-based selection, or model fitting.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

TRANSFER_COLUMNS = (
    "lot", "wafer", "device", "condition", "replicate", "branch",
    "temperature_K", "drain_voltage", "body_voltage", "gate_voltage", "drain_current",
)
PHYSICAL_COLUMNS = ("lot", "wafer", "device")
# Original invented endpoint constants, declared independently of the transfer
# arithmetic below. Not measurements, extracted quantities, or accuracy claims.
TRAINING_ENDPOINTS = (
    "11.2", "7.4", "16.8", "6.1", "13.7", "9.5",
    "17.3", "8.2", "12.6", "10.9", "15.1", "5.7",
)
LATER_OUTCOMES = ("14.1", "5.8", "10.7")


def csv_bytes(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build():
    """Return deterministic fixture payloads; no filesystem or ML operations."""
    transfer, partitions, training, later = [], [], [], []
    for lot_index in range(5):
        lot = "invented_lot_" + str(lot_index + 1)
        for device_index in range(3):
            physical = {"lot": lot, "wafer": "01", "device": f"{device_index + 1:03d}"}
            partition = "train" if lot_index < 4 else "predict"
            partitions.append({**physical, "partition": partition})
            # One complete seven-point branch at a declared fixed bias. Local
            # wafer/device labels intentionally repeat in separate lot namespaces.
            base = 2 + (lot_index + device_index) % 3
            amplitude = 1 + (3 * lot_index + device_index) % 5
            for step in range(7):
                transfer.append({
                    **physical, "condition": "invented_fixed_bias_0p8",
                    "replicate": "recorded_01", "branch": "declared_forward",
                    "temperature_K": "300", "drain_voltage": "0.8",
                    "body_voltage": "0", "gate_voltage": f"0.{step}",
                    "drain_current": str(amplitude * base ** step),
                })
            if partition == "train":
                training.append({**physical, "target": TRAINING_ENDPOINTS[3 * lot_index + device_index]})
            else:
                later.append({**physical, "target": LATER_OUTCOMES[device_index]})
    config = {
        "schema_version": "measurement-regression-v2",
        "adapter": "fet-transfer-device-v1",
        "endpoint_name": "invented_later_response",
        "target_unit": "arbitrary_units",
        "group_by": "lot",
        "features": ["criterion_gate_source_voltage_V", "log_current_slope_decades_per_V"],
        "measurement": {
            "voltage_unit": "V", "current_unit": "nA",
            "criterion_current": "50", "criterion_current_unit": "nA",
            "current_polarity": "positive", "max_bracket_span": "0.1",
            "window_min": "0.1", "window_max": "0.4",
        },
        "outer_splits": 2, "inner_splits": 2,
        "baselines": ["dummy", "ridge"], "seed": 1729,
        "objective": "group_mae", "max_fits": 500,
        "declarations": {
            "global_physical_ids": True,
            "pre_outcome_features": True,
            "endpoint_not_derived_from_features": True,
            "measurement_assumptions_accepted": True,
        },
    }
    return {
        "raw/transfer.csv": csv_bytes(TRANSFER_COLUMNS, transfer),
        "partitions.csv": csv_bytes((*PHYSICAL_COLUMNS, "partition"), partitions),
        "training_targets.csv": csv_bytes((*PHYSICAL_COLUMNS, "target"), training),
        "heldout_outcomes.csv": csv_bytes((*PHYSICAL_COLUMNS, "target"), later),
        "config.json": (json.dumps(config, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    }


def write_new(output_dir, payloads):
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir) or not output_dir.parent.is_dir():
        raise ValueError("output must be a fresh directory with an existing parent")
    output_dir.mkdir(mode=0o700)
    for name, raw in sorted(payloads.items()):
        path = output_dir / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        write_new(args.output_dir, build())
    except (OSError, ValueError):
        parser.exit(2, "error: fixture output must be fresh and writable; incomplete output is not usable\n")
    print("Created wholly invented FET interface fixtures; not physical or ML validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
