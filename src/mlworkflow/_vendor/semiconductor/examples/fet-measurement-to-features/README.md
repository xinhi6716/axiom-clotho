# Shared FET measurements to separate branch features

An offline, original synthetic workflow that runs the existing
[constant-current Vth extractor](../../tools/vth-constant-current/README.md) and
[user-windowed transfer log-slope tool](../../tools/transfer-log-slope/README.md)
on **the same exact transfer-CSV bytes**. It exports one row per declared branch,
not one independent sample per row, and retains both complete tool reports.
It does not copy or change either extraction algorithm.

The two candidate features are:

- `criterion_gate_source_voltage_V`: signed VGS at the explicitly supplied signed
  terminal-current criterion, using an exact measured hit or adjacent **linear
  current** interpolation. It is an operational criterion value, not a universal
  physical threshold.
- `log_current_slope_decades_per_V`: signed descriptive least-squares slope of
  `log10(s * ID_A / 1 A)` versus signed VGS, using all measured samples inside an
  explicitly chosen inclusive VGS window. `s` follows the declared current sign.
  Neither this slope nor its reciprocal is a certified subthreshold swing.

No physical regime, device type, state, window, current criterion, geometry
normalization, target, or model is inferred. This is an arithmetic/data-lineage
example, not simulated transistor physics, experimental evidence, ML performance
or an endorsement of the fixture's criterion/window for real devices.

## Run from the repository root

Python 3.10+; only the standard library and these two existing tools. There is no
network, dependency installation, external API, instrument control, upload, or
model fit. The chosen output directory must not exist, and its parent must exist.

```sh
RUN_DIR="$(python3 -c 'import tempfile; print(tempfile.mkdtemp(prefix="fet-workflow-"))')"
export RUN_DIR
python3 examples/fet-measurement-to-features/generate_synthetic.py \
  --output-dir "$RUN_DIR/invented-input"

python3 examples/fet-measurement-to-features/build_features.py \
  "$RUN_DIR/invented-input/transfer.csv" \
  --voltage-unit V --current-unit nA \
  --criterion-current 50 --criterion-current-unit nA \
  --current-polarity positive --max-bracket-span 0.1 \
  --window-min 0.1 --window-max 0.4 \
  --accept-measurement-assumptions --output-dir "$RUN_DIR/branch-features"

python3 -m unittest discover -s examples/fet-measurement-to-features/tests -v
```

The checked-in `fixtures/transfer.csv` is an interchangeable input. The generator
reproduces it byte-for-byte, without randomness, timestamps or external sources.
It invents four namespaced physical-device labels across two lots, two technical
replicates and two branches per device: **16 branch rows from 112 observations**.
Local device names intentionally repeat across lots. Identity is the full
`(lot, wafer, device)`, not the device text alone.

The explicit arithmetic is `ID[nA] = L * R * B * base**step`, with `L` equal to
1 or 5 for the two lots, `R` equal to 1 or 3 for the two replicates, `B` equal to
1 or 2 for forward/reverse branches, and `base` equal to 10 or 4 for the two local
device labels. VGS is `step / 10 V`, for integer steps 0 through 6; reverse
branches are stored from step 6 to 0. This is invented arithmetic, not a fitted
or physically validated device model. The forward and reverse multipliers are
not evidence of a hysteresis mechanism.

At the documented settings the first forward/reverse pair gives criterion VGS
approximately `0.1444444444 V` and `0.1166666667 V`, respectively, with `0.1 V`
interpolation brackets. Both log slopes are `10 decades/V`. The base-4 devices
give approximately `6.020599913 decades/V`. Four points per branch are in the
window: 64 fitted rows and 48 outside-window rows retained in the audits.
These numbers test software, not real-device accuracy or scientific validity.

## Select one already-declared fixed-bias branch

Use this optional preparation step when your **already-declared 11-column
transfer table** contains several bias conditions or recorded repeats. It is not
an instrument/vendor-export importer and does not discover forward branches,
assign acquisition IDs or infer units. Before using it, correctly map your source
to the schema below and retain the original acquisition records. In particular,
the first increasing VGS run in a multi-bias export may be at **zero drain bias**,
not at the bias you intended to analyze. An exact `--branch forward` filter in
the standalone Vth tool also matches that label across conditions and repeats.

This companion selects **one entire exact six-ID group**, then checks its declared
drain bias and VGS direction. All six IDs, both units, expected nonzero VDS and
increasing/decreasing direction are required. There is no default, wildcard,
first-matching repeat or fallback to a branch whose feature extraction succeeds.
Names such as `forward` carry no automatic direction or physical interpretation.

Copy this complete example from the repository root:

```sh
SELECT_DIR="$(python3 -c 'import tempfile; print(tempfile.mkdtemp(prefix="fet-selection-"))')"
python3 examples/fet-measurement-to-features/select_transfer.py \
  examples/fet-measurement-to-features/fixtures/multibias_transfer.csv \
  --lot invented_lot --wafer invented_wafer --device one_device \
  --condition bias_8 --replicate acquisition_1 --branch forward \
  --voltage-unit V --current-unit nA --drain-bias 0.8 \
  --gate-direction increasing --output-dir "$SELECT_DIR/selected"

python3 examples/fet-measurement-to-features/build_features.py \
  "$SELECT_DIR/selected/transfer.csv" \
  --voltage-unit V --current-unit nA \
  --criterion-current 90 --criterion-current-unit nA \
  --current-polarity positive --max-bracket-span 0.1 \
  --window-min 0 --window-max 0.3 \
  --accept-measurement-assumptions --output-dir "$SELECT_DIR/features"
```

Both commands exit **0**. The invented fixture has 96 observations: eleven
declared drain biases from 0 to 1 V, each with four-point forward/reverse branches,
and a separately labeled second acquisition at 0.8 V. Its `condition=bias_8` and
`replicate=acquisition_1` labels were supplied by the fixture's author, **not
inferred by the selector**. Current values are arbitrary arithmetic, not simulated
transistor physics. The selected four records are original physical CSV lines
**66–69**; 92 unselected records remain in the retained source. The unchanged
joint workflow produces one row with criterion VGS **0.1 V** and descriptive log
slope **10 decades/V** at these synthetic settings. These are software examples,
not recommended measurement criteria or real-device validation. All branches and
both acquisitions still refer to one invented physical-device label.

The selection directory contains exactly:

- `source.csv`: complete, unchanged input bytes, including excluded zero-bias,
  reverse and repeated acquisitions.
- `transfer.csv`: the selected rows, in supplied order and the existing schema.
  All decoded CSV fields, including numeric tokens, are retained; output quoting
  and header order are canonicalized. **No numerical unit conversion occurs in
  this export**; pass the same declared units to the downstream workflow.
- `selection.json`: exact selection/units, input and output SHA-256 fingerprints,
  total/selected/excluded record and group counts, and each selected record's
  original data-record ordinal and inclusive physical start/end lines mapped to
  its new CSV position. The downstream reports correctly refer to `transfer.csv`
  lines 2–5 in this example; use this manifest to trace those lines back to 66–69.
  Keep both bundles together. Hashes identify bytes, not scientific authenticity.

The API `select_transfer(input_path, *, voltage_unit, current_unit, group,
expected_drain_bias, gate_direction)` returns these three byte payloads without
writing them; `group` is exactly the six ID fields below and bias is decimal text
in the declared voltage unit. `workflow.write_outputs` is the existing explicit
fresh-directory writer used by the CLI.

### Selection checks and limits

- The complete input first passes the existing bounded schema and numeric parser,
  including excluded rows. Malformed values are not hidden by selection. Existing
  limits remain 8 MiB, 10,000 records, 1,000 groups and 1,000 records per group;
  the re-encoded selected CSV must also fit the 8 MiB input limit.
- The selected key must occupy **one contiguous block of data records**. This
  preparation step is intentionally stricter than the joint extractor, which
  allows interleaving. A separated occurrence of the same key rejects rather
  than being merged or treated as a first match. Correctly distinguish recorded
  repeats in the input; software cannot recover acquisition identity from labels.
- The whole selected group must have at least three points, exact expected
  nonzero VDS, fixed VBS, positive fixed Kelvin temperature, and strictly
  increasing/decreasing VGS as declared. Exact decimal comparisons are used, not
  an implicit tolerance. Duplicate/flat coordinates and reversals reject; no
  sorting, trimming, interpolation or deduplication is performed. A wrong bias
  rejects without selecting another group. Zero bias in an excluded group is
  allowed and retained; selecting zero bias is rejected for this FET route.
- Signed/zero/nonmonotonic current does not change which rows are selected.
  Currents are parsed for existing numeric validity, but no criterion or fit is
  run here. The downstream extractors may still reject the selected branch.
  Selected IDs must satisfy the joint workflow's existing spreadsheet-safe ID
  rule; the complete source snapshot is not sanitized for spreadsheet use.
- BOM, quoted fields, multiline fields and blank physical records follow the
  existing parser. Record ordinals exclude the header and blank records; physical
  line ranges include them. Original quote syntax lives in `source.csv`, while
  `transfer.csv` preserves decoded field values. Neither locator is a device ID.
- Fixed recorded values and direction do not prove settled/calibrated physical
  conditions, commanded-versus-measured semantics, verified chronology, a device
  regime or independent samples. Correct units, terminal references and protocol
  declarations remain the caller's responsibility.

Selection exit **0** means the declared subset/mapping was produced, not that
feature extraction or physics was validated. Exit **2** is a fatal declaration,
input or output error; diagnostics omit raw IDs, values and paths. Validation
finishes before output creation. Existing files, directories and symlinks are
never overwritten; rerunning the same command with the same destination exits 2.
The three-file bundle is **not a filesystem-atomic transaction**: an I/O failure
may leave incomplete newly created output. Discard it and use a new destination.
The existing writer requests POSIX mode 0700 for the directory and 0600 for files;
other platforms may differ. No data are transmitted. Real source/subset bundles
remain private unless separately authorized for sharing.

## Exact input and parameter contract

Both tools accept the same UTF-8 CSV schema, with exactly these unique columns,
in any order. No manifest join or cross-device matching is invented:

```csv
lot,wafer,device,condition,replicate,branch,temperature_K,drain_voltage,body_voltage,gate_voltage,drain_current
```

- All six ID columns together identify one acquisition-ordered branch. Preserve
  branches, repeats, lots and wafers; do not concatenate acquisitions under one
  key. Interleaving groups is allowed, but within-group order is never sorted.
  Conditions may differ between groups; results are not pooled or made comparable
  just because they share a device. The condition ID must link to a separately
  retained acquisition protocol, including timing and preconditioning.
- Voltage columns mean source-referenced VDS, VBS and VGS. `--voltage-unit` is
  required (`V`, `mV`, `uV`) and applies to all three columns, the positive maximum
  VGS bracket span, and both inclusive window bounds. Kelvin temperature is
  fixed and positive; VDS is fixed and nonzero and VBS fixed within each branch.
- `--current-unit` and `--criterion-current-unit` are independently required
  (`A`, `mA`, `uA`, `nA`, `pA`). Currents are signed, unnormalized terminal currents,
  not density or W/L-normalized values. `--current-polarity positive|negative`
  declares the sign convention, not n-/p-channel type. Negative current requires
  a negative criterion, such as `--criterion-current=-50` with unit `nA`.
- There is no default criterion, window, branch filter or bracket width. A single
  criterion and window apply to every branch in a run. Different required
  criteria/windows need separately documented runs; do not tune them to a target
  or to obtain attractive results. The spacing bound is not uncertainty.
- IDs must satisfy the tools' nonempty, unpadded, control-free, 128-character
  limit. For CSV feature export, IDs beginning with `=`, `+`, `-`, or `@` are also
  rejected rather than escaped or renamed, to avoid spreadsheet formulas.
  The exact input snapshot and JSON audits are not spreadsheet-sanitized.
- All existing numeric/resource contracts apply: decimal/scientific text with
  at most 64 stripped characters, 32 coefficient digits and exponent magnitude
  at most 200; nonzero SI inputs from `1e-150` through `1e150`; maximum 8 MiB,
  10,000 rows, 1,000 groups and 1,000 rows per group. Malformed values anywhere,
  even outside the chosen window, fail. Only regular files are read.
- Blank physical lines are skipped, not malformed nonempty rows. UTF-8 BOM and
  quoted CSV are supported. Source rows are physical line numbers at the end of
  each record, including the header. The original bytes are retained unchanged.

The required `--accept-measurement-assumptions` declares acceptance of **both**
linked tools' assumption sets: source-referenced signed terminals; fixed device
state, temperature, bias and acquisition protocol; correctly declared polarity;
justified criterion, explicit window and local linear-current interpolation;
and independent review of settling, noise floor, leakage, compliance, contacts
and physical interpretation. Software cannot verify those declarations or infer
linear/saturation/subthreshold regimes. No current sign is rescued by absolute
value, no points trimmed, no branch merged, and no geometry correction performed.

The joint workflow is deliberately stricter than either tool alone. For example,
a wrong-sign sample outside the log window can be retained with a warning by the
log tool but rejected by the Vth tool. A valid zero log slope is exported as zero,
not missing; its undefined reciprocal remains null in the full log report.

## Outputs, failures and audit

Success creates eight files:

- `features.csv`: one row per six-part branch key, retaining device/condition/
  replicate/branch IDs, lot-based `split_group`, fixed conditions, criterion,
  window, bracket limit, counts, extraction method and the two candidate features.
- `predictors.csv`: **exactly the two allowlisted numeric columns**, in exactly
  the same data-row order as `features.csv`. It has no IDs, target or metadata.
  Do not shuffle or separately filter it: use the full table for identity joins
  and grouping, and `provenance.json` for its row-to-acquisition mapping.
- `feature_schema.json`: the exact seven-field version-1 calculator allowlist:
  `schema_version`, `row_key`, `model_feature_allowlist`, `metadata_columns`,
  `target_columns`, `feature_units`, `warnings`. Targets are deliberately empty.
- `vth_report.json` and `log_slope_report.json`: complete upstream report objects,
  with their original schema, parameters, observations, warnings, source rows,
  interpolation brackets, window decisions, residuals and signed reciprocals.
  No wrapper silently changes their top-level schema; JSON formatting may differ.
- `transfer_input.csv`: the exact bounded input snapshot used and hashed by both
  extractors, including original numeric text, header order and BOM if present.
- `workflow_status.json`: complete or audit-only status plus both per-branch
  statuses/rejection reasons. Successful extraction does not certify ML suitability.
- `provenance.json`: input, source-code and other-output SHA-256 fingerprints,
  runtime and exact invocation parameters, counts, physical-device key, and
  original-source/feature/predictor row lineage. Keep the whole output bundle.

If **any branch is rejected by either extractor**, exit is `1` and the requested
new directory contains only the five audit files: both full reports, input
snapshot, workflow status and provenance. No feature table, predictor CSV or
allowlist is emitted, even for other passing branches. Both statuses remain
visible; no incomplete branch is silently filtered or imputed. Investigate the
records and protocol rather than deleting failed groups until export succeeds.
This fail-closed policy is not a general missing-data or censoring methodology.

Exit `0` means a complete software export, not physical or ML validation.
Exit `2` means a fatal CLI, malformed-input, unsupported schema, join-integrity,
unsafe-CSV-ID or I/O error. Validation precedes output creation; a fatal parsing
error has no recoverable upstream report and creates no output directory. An
output I/O failure can leave an incomplete new directory: discard it and use a
new destination after investigating the cause. Existing files/directories and
live or dangling symlink destinations are never overwritten. Output directories
request POSIX mode `0700`, files `0600`; other platforms may differ. Diagnostics
omit private IDs/values/paths. Deliberately requested files contain raw data and
must not be committed with real measurements. Nothing is transmitted externally.

## Optional declared measurement annotations (Python companion)

**Question answered:** which analyzed measurement rows, reported Vth support
rows, and user-window rows carry an explicitly supplied current-floor relation
or compliance-status declaration? This is not an instrument validity check.
The companion does not refit, filter, delete, relabel, or change any existing
feature, predictor, window, criterion, extractor warning or status.

This first version accepts **only the existing Python `build_features.py`
bundle with provenance schema 1**. Rust/native provenance schema 2 and other
schemas are explicitly rejected, not guessed or advertised as supported.
No new dependencies, Rust command, vendor parser, instrument connection or GUI
are needed. The no-follow file-access implementation requires POSIX and Python
3.9 or later; Windows execution is not supported by this companion.

### Template → edit declarations → report

Start with a complete or audit-only Python bundle, here called `fet-run`.
Keep it unchanged and choose new destinations **outside** that directory:

```sh
python3 examples/fet-measurement-to-features/annotate_measurements.py template \
  --bundle fet-run --output declarations.json
```

Open `declarations.json` in a text editor. Edit **only** the entries under
`declarations`, preserving branch order, group IDs and every `source_row`.
The generated `context` already shows the six IDs, original decoded CSV fields,
input units, record indices, physical row numbers, reported Vth support and log
window/fit-use rows. Do not copy a hash from a different run or count rows by hand.

Each branch initially has an all-null `current_floor`. If you can explicitly
declare one magnitude boundary that applies to **every row of that branch**,
replace its four fields. For example, the following is an **invented declaration
for a synthetic fixture, not a recommended noise threshold**:

```json
"current_floor": {
  "value": "8",
  "unit": "pA",
  "definition": "Synthetic absolute terminal-current comparison boundary",
  "basis": "Original synthetic example only; not instrument characterization"
}
```

Leave all four fields null when unknown. If a branch has varying measurement
conditions and no single applicable boundary, leave it unknown; this version
does not invent a default or provide per-row floor overrides. Each generated row
also has `compliance_state: "unknown"` and `compliance_basis: null`. Change a
state to `"asserted"` or `"not_asserted"` only when supplying an explicit
corresponding declaration and a nonempty basis describing its status record.
Do not infer a state from a current value, plateau, programmed limit or floor.

```sh
python3 examples/fet-measurement-to-features/annotate_measurements.py report \
  --bundle fet-run --declarations declarations.json \
  --output-dir annotation-run
```

Read `annotation-run/summary.txt`, then inspect `annotations.json` for all rows
and their declaration/support intersections. `declarations_input.json` retains
the exact submitted metadata bytes. Keep these three files with the unchanged
original bundle; their hashes identify bytes, not authenticity.

### Meaning, units and missingness

- The floor is a caller-declared boundary for **absolute, unnormalized terminal
  drain current**. `value` is decimal/scientific **text**, at most 64 characters
  including whitespace, with the existing bounded numeric parser. `unit` must
  independently be `A`, `mA`, `uA`, `nA` or `pA`; no unit is inferred or inherited.
  Nonzero SI magnitude must be within `1e-150`–`1e150`, at most 32 coefficient
  digits and decimal exponent magnitude 200. Original value/unit text is retained.
- A supplied floor needs a nonempty `definition` (at most 256 characters) and
  `basis` (at most 2,048 characters). They must be control-free. Explain what
  the boundary represents and how it was obtained; the tool cannot establish
  applicability. It does not convert RMS, peak-to-peak, resolution or accuracy
  into one another, a detection limit, uncertainty or a confidence interval.
- Exact parsed current magnitudes are reported as `below`, `equal` or `above`
  that boundary; absent floor is `not_declared`. Negative floors and partial
  declarations are rejected. Zero is allowed, retained and explicitly warned:
  **a zero boundary does not establish noise-free measurement**. Signed current
  and its original numeric text remain unchanged.
- Compliance states are only `asserted`, `not_asserted` and `unknown`.
  `not_asserted` is a caller declaration about a status, **not proof that the
  instrument never entered compliance**. Non-unknown states require a nonempty
  control-free basis of at most 2,048 characters; an unknown state may also have
  a basis explaining what is missing. This version has no compliance-limit field.
  No status bit is decoded or fetched, and basis text/URLs are never followed.
- Missing declarations stay explicit. Keep every template row, including unknown
  rows; deleting, duplicating, reordering or adding rows/branches, changing the
  readonly context, or introducing unknown declaration fields is an error.
- An audit-only bundle is supported without creating feature outputs. If Vth
  extraction is null, `vth_support` and its row membership are **null/unavailable**,
  not an empty all-clear. Log-window membership remains visible even when the
  fit is rejected; `reported_used_in_log_fit` is then false and its aggregate
  is null. Only a consistent upstream `ok` status/non-null fit can report fit use.
  Unknown counts are reported for available scopes; no scope certifies validity.

### Analyzed-input binding, limits and failures

`source_row` means the physical line at the **end of a record in this bundle's
`transfer_input.csv`**, including the header. It is not an instrument-row claim
or the pre-selection `source.csv` row. Existing selector lineage is not recreated;
retain its `selection.json` separately if you need that earlier mapping.

Templates are generated from the actual retained bytes and the existing report
and provenance hashes. LF, CRLF, BOM and quoted records therefore get their own
correct binding without a hard-coded fixture SHA. Metadata JSON whitespace,
newline style and object-key order may change while editing; its submitted bytes
and hash are recorded. Changing analyzed input bytes, including line endings,
invalidates the old bundle/template. Recreate a consistent bundle and template;
the companion never silently normalizes or rebinds changed measurements.

The companion reads a fixed five-file audit allowlist, plus the three documented
feature files only for a complete bundle. It checks hashes, shared units,
reported observations, branch/row mappings, log-window membership, fit-use
markers, support references and existing lineage. It does **not** recompute
feature values or prove that a producer actually ran. A coherently rewritten
bundle can be internally consistent; hashes are not an authenticity mechanism.
No `source_sha256` path or arbitrary fingerprint-map path is opened. Audit-only
bundles containing stale `features.csv`, `predictors.csv` or `feature_schema.json`
are rejected rather than presented as an ordinary five-file export.

Inputs are bounded: transfer CSV 8 MiB; each core JSON report 32 MiB; provenance
16 MiB; workflow status, features CSV and predictors CSV each 8 MiB; feature
schema 1 MiB; required bundle files **96 MiB total**; declarations **64 MiB**.
The existing 10,000 rows, 1,000 branches and 1,000 rows/branch limits apply.
JSON duplicate keys/nonfinite numbers, missing files, incompatible schemas,
invalid units and any checked mapping/hash disagreement fail closed. The bundle
directory itself, its opened members and the declarations file must not be
symlinks; nonregular members, including FIFOs/devices, are rejected. Ancestor
directory resolution remains the operating system's normal path resolution.

Exit `0` means template/report creation only, even for unknown or adverse
declarations; it is **not a quality pass**. Exit `2` means a declaration,
integrity, schema, platform or I/O failure. Validation precedes output creation;
existing files/directories and output symlinks are never overwritten. Output
files request POSIX mode 0600 and report directories 0700. As in the existing
workflow, an I/O failure may leave incomplete newly created outputs; discard
them and use a new destination. Diagnostics omit private IDs/values/paths, but
deliberately requested reports contain raw private data. Nothing is transmitted.

Original synthetic annotation tests (including the hand-derived `20 pA` Vth
support and factor-four log slope) can be run independently:

```sh
python3 -m unittest discover -s examples/fet-measurement-to-features/tests \
  -p 'test_measurement_annotations.py' -v
```

## Calculator handoff without target leakage

The generated `feature_schema.json` is structurally compatible with the actual
ML Super Calculator / `groupbench` `--features-file` parser. That parser reads
column names, **not scientific validity, endpoint timing or causal eligibility**.
Its exact local parser was exercised without fitting a model; no calculator is
required to run this workflow. A calculator installation may have its own
numerical dependencies; this example neither installs nor imports them.

If you already have a local calculator checkout/environment, independently check
only the allowlist (substitute its actual `src` directory):

```sh
PYTHONPATH=/path/to/ml-super-calculator/src python3 - <<'PY'
import os
from pathlib import Path
from groupbench.feature_manifest import read_feature_manifest
path = Path(os.environ["RUN_DIR"]) / "branch-features/feature_schema.json"
assert read_feature_manifest(path) == [
    "criterion_gate_source_voltage_V", "log_current_slope_decades_per_V"]
print("Feature-name parser compatibility passed; no target or model was created.")
PY
```

For a real downstream task:

1. Define an independently sourced endpoint and when it becomes available, before
   creating model inputs. There is no fabricated target or executable model-run
   example here. Both features share one acquisition, so related-measurement or
   post-outcome leakage must be considered even without an algebraic identity.
2. **Never use the target itself or its algebraic proxy as a feature.** If Vth is
   the target, this same criterion VGS cannot be a predictor. If log slope is the
   target, its reciprocal or reciprocal magnitude leaks the answer. Both features
   are only candidates: revise the allowlist for the actual task. The schema
   excludes reciprocals and all diagnostic/metadata columns by default.
3. Preserve the actual dependency structure. The minimum physical-device key is
   `(lot, wafer, device)`; all branches, repeats and conditions of a device must
   stay in the same evaluation group. `split_group=lot` is a conservative example
   declaration, not proof of independent lots. Wafer, campaign or time dependencies
   may require a different, stricter grouping. Never randomly split branch rows.
4. Keep IDs and grouping metadata outside predictors. Use the full table for a
   reviewed one-to-one target join and grouping; the predictor-only CSV cannot
   reconstruct identities by itself. Fit preprocessing only on training groups.
5. Four invented devices, sixteen branches and two invented lots are **not** a
   meaningful sample for model evaluation or independent-device statistics.
   No score, fitted model, causal claim or real-device performance is reported.

## Verification scope

Run the focused command above and `python3 run_tests.py`. Regression tests cover
both actual tool calls, exact full-report retention, fixture regeneration,
branch/device identity, signs and units, exact decimal window selection, valid
zero slope, both one-tool-only rejection directions, fail-closed export,
source/output hashes, original-byte snapshot consistency, source-row lineage,
calculator schema shape, private diagnostics and no-overwrite outputs. Synthetic
software tests do not validate an instrument or a physical regime. These original
source files and fixtures use the repository's [MIT license](../../LICENSE).
