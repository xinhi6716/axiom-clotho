"""Blank caller-input scaffolds, not invented data or prepared ML bundles.

This module uses only the standard library and existing serialization/writing
helpers. Scientific choices deliberately fail the unchanged prepare contracts.
"""
import os
import stat

from . import contracts as c
from .adapters import ADAPTERS, FET_ADAPTER, FET_FEATURES


RRAM_HEADERS = {
    "manifest.csv": ("batch", "wafer", "device", "cycle", "replicate", "branch", "state", "condition"),
    "conditions.csv": ("condition", "read_voltage_V", "temperature_K", "read_delay_ms", "protocol"),
    "iv_sweeps.csv": ("device", "cycle", "branch", "voltage", "current"),
}
FET_HEADERS = {
    "transfer.csv": (*ADAPTERS[FET_ADAPTER]["acquisition_key"], "temperature_K", "drain_voltage",
                     "body_voltage", "gate_voltage", "drain_current"),
}
STATUS_FILE = "SCAFFOLD_STATUS.json"


def _config(workflow):
    config = {
        "schema_version": "measurement-regression-v1",
        "endpoint_name": None, "target_unit": None, "group_by": None, "features": [],
        "max_bracket_span_V": None,
        "outer_splits": 3, "inner_splits": 2, "baselines": ["dummy", "ridge"],
        "seed": 7, "objective": "group_mae", "max_fits": 200,
        "declarations": {name: False for name in sorted(c.DECLARATIONS)},
    }
    if workflow == "fet":
        del config["max_bracket_span_V"]
        config.update(schema_version="measurement-regression-v2", adapter=FET_ADAPTER,
                      outer_splits=2, seed=1729, max_fits=500,
                      measurement={name: None for name in (
                          "voltage_unit", "current_unit", "criterion_current", "criterion_current_unit",
                          "current_polarity", "max_bracket_span", "window_min", "window_max")})
    return config


def _readme(workflow):
    if workflow == "rram":
        identity = "globally namespaced device IDs; choose group_by from batch, wafer, device"
        feature_names = c.FEATURES
        measurements = """### RRAM measurement choices

- `raw/manifest.csv`: explicitly record HRS/LRS states, paired cycles and identities.
- `raw/conditions.csv`: fixed read condition for each device; one feature row per device.
- `raw/iv_sweeps.csv`: voltages MUST be V; currents MUST be A. Conditions use V, K
  and ms as named in the headers. These are fixed input units, NOT guesses from
  your instrument; convert and document your source data explicitly if needed.
- Set `max_bracket_span_V` to a justified positive decimal string in volts.
  No bracket limit, state, unit conversion or endpoint is inferred.
- Repeated cycles are aggregated within the device, not independent training rows.
- A wafer ID cannot map to different batches. Local IDs must be globally namespaced.

Full contract: https://github.com/xinhi6716/axiom-clotho/blob/main/docs/CLOTHO_CSV_BETA.md
"""
        localized = "RRAM 固定使用 V、A、K、ms（依欄名）；請先核對儀器單位，程式不會猜測或轉換。"
    else:
        identity = "exact (lot, wafer, device) tuples; choose group_by from lot, wafer, device"
        feature_names = FET_FEATURES
        measurements = """### FET measurement choices

- `raw/transfer.csv`: exactly one complete acquisition per (lot, wafer, device).
  Condition, replicate and branch must be explicit; no averaging or automatic selection.
- Voltage columns are source-referenced VDS, VBS and VGS. Each branch requires
  fixed nonzero VDS, fixed VBS, positive fixed temperature_K, and strictly monotonic VGS.
- In `measurement`, choose `voltage_unit` from V, mV, uV. It applies to all voltage
  columns, `max_bracket_span`, `window_min` and `window_max`.
- Choose `current_unit` and `criterion_current_unit` independently from A, mA, uA,
  nA, pA. Supply signed terminal currents, not current density or normalized current.
- Set a signed nonzero `criterion_current`; `current_polarity` is positive or
  negative and must agree with its sign. This is NOT a declaration of transistor type.
- Set a positive `max_bracket_span` and ordered `window_min` < `window_max`.
  All four measurement numbers are JSON decimal strings, without embedded units.
- Criterion VGS and windowed log slope are descriptors, not certified threshold
  physics or proof of a subthreshold regime. No physical settings are chosen for you.

Full contract: https://github.com/xinhi6716/axiom-clotho/blob/main/docs/FET_DEVICE_REGRESSION.md
"""
        localized = "FET 請明確填寫電壓／電流單位、帶正負號的準則電流、極性、括距與視窗；程式不會推測。"
    return (f"""# Your {workflow.upper()} measurement inputs — blank scaffold

**Not ready to prepare. No measurement rows or invented results were generated.**
This is an editable input directory, NOT a prepared/run/score bundle. Nothing was
fitted, extracted, downloaded or uploaded. Existing invented demos are separate.
The init command is included in 0.3.0b4; the frozen 0.3.0b3 wheel lacks it.

## 1. Supply measurements and exact joins

Use UTF-8 comma-separated CSV, decimal points, and the supplied exact headers.
Keep IDs as exact text, including leading zeros and case. Do not pad, normalize,
silently drop or join rows by position. Identity: **{identity}**.

| File | Fill with |
| --- | --- |
| `raw/*.csv` | Your canonical measurements, using the units and acquisition rules below. |
| `partitions.csv` | Every physical device exactly once, with partition `train` or `predict`. |
| `training_targets.csv` | Exactly training devices, with independently supplied finite numeric target values. |
| `later_outcomes.csv` | Leave empty until later scoring; then exactly the complete saved prediction cohort. |
| `SCAFFOLD_STATUS.json` | Setup status only, not scientific validation; leave this file unchanged. |

Keep prediction outcomes out of training targets. `prepare`, `plan` and `run` do
not accept this later-outcome file; only `score` does. Do not fabricate an endpoint
from a feature or its algebraic proxy. Preserve raw sources and any conversion map.

{measurements}
## 2. Edit config.json before preparing

- Replace `endpoint_name` and `target_unit` with your actual separately measured
  endpoint and its unit. Do not substitute invented example names or units.
- Choose `group_by` deliberately. Train and predict groups must be disjoint;
  the program cannot infer physical independence, aliases or a valid study design.
- Set `features` to a nonempty, unique subset of:
{chr(10).join('  - `' + name + '`' for name in feature_names)}
- Complete the measurement choices above. JSON `null` and the empty feature list
  are deliberate blockers, not defaults that the program will infer.
- Review every `declarations` entry. Change false to true only if justified by
  your actual protocol: global physical IDs, pre-outcome features, an endpoint not
  derived from features, and accepted measurement assumptions. These remain caller
  assertions, NOT scientifically verified facts. Never tick them just to pass validation.
- Split counts, dummy/ridge families, seed, group_mae and max_fits are editable
  computational starting settings reused from public software examples. They are
  NOT scientific recommendations. Enough distinct training groups must remain for
  nested comparison. max_fits limits planned pipeline fits, not time or memory.

## 3. Prepare, inspect the plan, then decide whether to fit

After filling and reviewing the inputs, run from this directory. Choose fresh
sibling output names; the examples below must not already exist.

```sh
clotho prepare --input-dir raw --partitions partitions.csv \\
  --training-targets training_targets.csv --config config.json \\
  --output-dir ../prepared
clotho plan --prepared ../prepared
# Only after reviewing the plan; this command fits models:
clotho run --prepared ../prepared --output-dir ../model-run
# Only after predictions exist and later outcomes are actually available:
clotho score --run-dir ../model-run --outcomes later_outcomes.csv \\
  --output-dir ../evaluation
```

The unchanged prepare contracts reject this untouched scaffold before writing a
prepared bundle. Plan/run need the existing ML dependencies. Keep all completed
output bundles together; checksums do not establish scientific validity.

## Safety and privacy

Real measurements, IDs, targets and later outputs are private local artifacts.
Do not commit or share them without separately reviewing and authorizing that use.
Use trusted, idle local directories. Init rejects existing outputs and pre-existing
symlink path components at preflight. Exclusive writes preserve competing files;
observed root/status replacement or an unexpected inventory fails creation. These
checks do NOT promise a transaction or prevent all outside file creation under
hostile concurrent filesystem changes. Init never unlinks a status file: it updates
only its exclusively created, still-open status handle. A failure may leave partial
files. Missing, malformed or INCOMPLETE_DO_NOT_USE status means do not use the
scaffold; retry a fresh name. Never edit status to imply success. A successful
scaffold records READY_TO_EDIT_NOT_VALIDATED, not a validated analysis, and has no
COMPLETE.json. A failed command is a failure even if a status file appears ready.

## 繁體中文快速指南

這是空白輸入範本，尚不能執行 prepare；沒有產生虛構量測，也沒有訓練或上傳。
請填入自己的量測與獨立目標，保留 CSV 標頭及精確身分，分別指定 train／predict。
{localized}
請自行選定目標名稱、目標單位、實體分組、特徵與量測參數；null 不是自動推論。
科學聲明預設 false，只有真實流程支持時才能改成 true，不可只為通過檢查。
later_outcomes.csv 留到預測完成且後續結果可用時才填寫，僅供 score 使用。
先 prepare、再 plan 檢查，確認後才 run；輸出請使用全新相鄰目錄。
0.3.0b4 已包含 init；凍結的 0.3.0b3 wheel 不含此指令。
""").encode("utf-8")


def _check_created(destination, directory_stat, status_stat, payloads):
    """Detect observed replacement; not a transaction against hostile writers."""
    def same_identity(path, original, kind):
        current = path.lstat()
        return kind(current.st_mode) and (current.st_dev, current.st_ino) == (original.st_dev, original.st_ino)

    if (not same_identity(destination, directory_stat, stat.S_ISDIR)
            or not same_identity(destination / STATUS_FILE, status_stat, stat.S_ISREG)):
        raise c.WorkflowError("init output identity changed during creation; do not use partial output; retry a fresh directory")
    files, directories = set(), set()
    for path in destination.rglob("*"):
        mode = path.lstat().st_mode
        name = path.relative_to(destination).as_posix()
        if stat.S_ISREG(mode):
            files.add(name)
        elif stat.S_ISDIR(mode):
            directories.add(name)
        else:
            raise c.WorkflowError("init output contains unexpected entries; do not use partial output; retry a fresh directory")
    if files != set(payloads) | {STATUS_FILE} or directories != {"raw"}:
        raise c.WorkflowError("init output inventory changed; do not use partial output; retry a fresh directory")
    if any(c.read_regular(destination / name) != data for name, data in payloads.items()):
        raise c.WorkflowError("init output bytes changed; do not use partial output; retry a fresh directory")


def _write_status(handle, workflow, status):
    # The handle refers only to the file exclusively created by this invocation,
    # even if someone replaces its directory entry. Never reopen, unlink or replace.
    handle.seek(0)
    handle.write(c.json_bytes({"schema_version": "clotho-input-scaffold-v1", "workflow": workflow, "status": status}))
    handle.truncate()
    handle.flush()


def initialize(workflow, output):
    """Create blank inputs only; leave partial failures in place, never overwrite."""
    if workflow not in ("rram", "fet"):
        raise c.WorkflowError("choose an explicit rram or fet workflow")
    destination = c.check_fresh_output(output)
    # Check lexical ancestors too: resolve() would conceal a symlink from this test.
    if any(path.is_symlink() for path in (destination.absolute(), *destination.absolute().parents)):
        raise c.WorkflowError("init requires a fresh directory without symbolic-link path components")
    headers = RRAM_HEADERS if workflow == "rram" else FET_HEADERS
    identity = ("device",) if workflow == "rram" else ADAPTERS[FET_ADAPTER]["physical_key"]
    payloads = {"raw/" + name: c.csv_bytes(columns, []) for name, columns in headers.items()}
    payloads.update({
        "partitions.csv": c.csv_bytes((*identity, "partition"), []),
        "training_targets.csv": c.csv_bytes((*identity, "target"), []),
        "later_outcomes.csv": c.csv_bytes((*identity, "target"), []),
        "config.json": c.json_bytes(_config(workflow)),
        "README.md": _readme(workflow),
    })
    # A competing mkdir fails; no cleanup path deletes anyone else's files.
    destination.mkdir(mode=0o700)
    directory_stat = destination.lstat()
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise c.WorkflowError("init output identity changed during creation; retry a fresh directory")
    fd = os.open(destination / STATUS_FILE, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "r+b") as handle:
        status_stat = os.fstat(handle.fileno())
        _write_status(handle, workflow, "INCOMPLETE_DO_NOT_USE")
        (destination / "raw").mkdir(mode=0o700)
        for name, data in sorted(payloads.items()):
            c.write_new(destination / name, data)
        _check_created(destination, directory_stat, status_stat, payloads)
        _write_status(handle, workflow, "READY_TO_EDIT_NOT_VALIDATED")
        try:
            _check_created(destination, directory_stat, status_stat, payloads)
        except (c.WorkflowError, OSError):
            # Safe even after marker/root replacement: this is our original inode.
            try:
                _write_status(handle, workflow, "INCOMPLETE_DO_NOT_USE")
            except OSError:
                pass
            raise
    return destination
