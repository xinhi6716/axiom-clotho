"""Closed, versioned measurement adapters; never import caller-supplied code."""
from pathlib import Path
import re

from .. import contracts as c

FET_ADAPTER = "fet-transfer-device-v1"
FET_FEATURES = ("criterion_gate_source_voltage_V", "log_current_slope_decades_per_V")
FET_FILES = ("features.csv", "predictors.csv", "feature_schema.json", "vth_report.json",
             "log_slope_report.json", "transfer_input.csv", "workflow_status.json", "provenance.json")
# Inventories are independent of (and cannot be shortened by) a saved receipt.
FET_INVENTORIES = {
    "prepared": {"training.csv", "future.csv", "cohort.csv", "config.json", "provenance.json", "identity_map.csv"}
                | {"features/" + name for name in FET_FILES},
    "run": c.INVENTORIES["run"] | {"identity_map.csv"},
    "score": c.INVENTORIES["score"].copy(),
}
ADAPTERS = {
    FET_ADAPTER: {"config_version": "measurement-regression-v2", "raw_files": ("transfer.csv",),
                  "feature_files": FET_FILES, "features": FET_FEATURES,
                  "physical_key": ("lot", "wafer", "device"),
                  "acquisition_key": ("lot", "wafer", "device", "condition", "replicate", "branch"),
                  "group_by": ("lot", "wafer", "device")},
}


def is_v2_config(config):
    return isinstance(config, dict) and config.get("schema_version") == "measurement-regression-v2"


def is_v2_bundle(path):
    """Dispatch hint only; full validation must precede any use of payloads."""
    root = Path(path)
    receipt_path = root / "COMPLETE.json"
    if root.is_symlink() or receipt_path.is_symlink():
        raise c.WorkflowError("saved bundles cannot contain symbolic links")
    value = c.parse_json(c.read_regular(receipt_path))
    return isinstance(value, dict) and value.get("schema_version") == "measurement-workflow-bundle-v2"


def finish_bundle(path, kind):
    path = Path(path)
    payloads = {name: c.read_regular(path / name) for name in FET_INVENTORIES[kind]}
    receipt = {"schema_version": "measurement-workflow-bundle-v2", "adapter": FET_ADAPTER,
               "kind": kind, "files_sha256": {n: c.sha(b) for n, b in sorted(payloads.items())},
               "integrity_not_authenticity": True}
    c.write_new(path / "COMPLETE.json", c.json_bytes(receipt))
    (path / "INCOMPLETE.json").unlink()
    return receipt


def write_bundle(path, kind, payloads):
    if set(payloads) != FET_INVENTORIES[kind] or any(not isinstance(b, bytes) for b in payloads.values()):
        raise c.WorkflowError("internal adapter output inventory mismatch")
    path = c.begin_bundle(path)
    for name, data in sorted(payloads.items()):
        (path / name).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        c.write_new(path / name, data)
    return finish_bundle(path, kind)


def load_bundle(path, kind):
    path = Path(path)
    expected = FET_INVENTORIES[kind]
    try:
        if path.is_symlink() or not path.is_dir():
            raise c.WorkflowError("bundle must be a regular local directory")
        actual = set()
        for item in path.rglob("*"):
            if item.is_symlink():
                raise c.WorkflowError("saved bundles cannot contain symbolic links")
            if not item.is_dir():
                actual.add(item.relative_to(path).as_posix())
        if actual != expected | {"COMPLETE.json"}:
            raise c.WorkflowError("bundle is incomplete or has unexpected files; do not repair it silently")
        raw = c.read_regular(path / "COMPLETE.json")
        receipt = c.parse_json(raw)
        if (not isinstance(receipt, dict)
                or set(receipt) != {"schema_version", "adapter", "kind", "files_sha256", "integrity_not_authenticity"}
                or receipt["schema_version"] != "measurement-workflow-bundle-v2"
                or receipt["adapter"] != FET_ADAPTER or receipt["kind"] != kind
                or receipt["integrity_not_authenticity"] is not True
                or not isinstance(receipt["files_sha256"], dict)
                or set(receipt["files_sha256"]) != expected
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
                       for h in receipt["files_sha256"].values())):
            raise c.WorkflowError("saved adapter bundle receipt does not match its fixed inventory")
        payloads = {name: c.read_regular(path / name) for name in expected}
        if any(c.sha(data) != receipt["files_sha256"][name] for name, data in payloads.items()):
            raise c.WorkflowError("saved bundle bytes changed; checksum agreement is not authentication")
        return payloads, c.sha(raw)
    except OSError:
        raise c.WorkflowError("unable to read a complete saved bundle") from None
