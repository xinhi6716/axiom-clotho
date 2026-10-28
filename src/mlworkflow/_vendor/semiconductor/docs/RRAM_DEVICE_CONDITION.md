# Explicit RRAM device-by-condition feature rows

The existing RRAM feature workflow defaults to one fixed declared condition per
globally unique physical-device ID. It still rejects a device changing conditions.
This opt-in mode instead produces one row per exact `(device, condition)` tuple,
without treating those rows as independent physical devices.

This is an offline preprocessing capability, not a model or evidence of device
performance. It uses the same signed I-V extraction, static `V/I` conversion and
explicit HRS/LRS pairing. It does not infer states, units, acquisitions, protocol
equivalence, physical independence, targets or prediction eligibility.

## Run an original synthetic example

From the repository root, choose a new output directory:

```sh
python3 examples/measurement-to-features/build_features.py \
  --input-dir examples/measurement-to-features/fixtures/multiple_conditions \
  --output-dir /tmp/invented-rram-condition-features \
  --max-bracket-span-V 0.01 --accept-measurement-assumptions \
  --feature-row-unit device-condition

# Native Rust, after the documented offline build or using a matching binary:
rust/target/release/smlt rram-features \
  --input-dir examples/measurement-to-features/fixtures/multiple_conditions \
  --output-dir /tmp/invented-native-rram-condition-features \
  --max-bracket-span-V 0.01 --accept-measurement-assumptions \
  --feature-row-unit device-condition
```

The invented fixture has one device in one batch/wafer, two complete cycles and
four state branches. Each branch is a single measured readpoint; all arithmetic
values are exactly representable in binary64. It is not a physical simulation.

- Condition A: 0.125 V, 300 K, 10 ms delay, protocol A. HRS/LRS resistances
  8192/128 ohm, paired ratio 64.
- Condition B: 0.25 V, 350 K, 20 ms delay, protocol B. HRS/LRS resistances
  1024/256 ohm, paired ratio 4.

Expected result: two condition rows, one physical-device group, ratios 64 and 4.
There is no pooled ratio median 34. Omitting the new option, or selecting
`--feature-row-unit device`, still rejects this bundle with exit 2 before output
creation. The ordinary default fixture continues to work unchanged.

## Identity, pairing and aggregation

- A `device` remains the unchanged, globally unique physical-device ID. Its
  batch and wafer cannot change. Do not append a condition to the device ID.
- Only conditions may differ between cycles of that device in the opt-in mode.
  A `(device, cycle)` still has exactly one declared HRS and one declared LRS,
  sharing one exact condition ID and one replicate ID. Cycle/replicate mapping
  remains one-to-one within the device. Reusing a cycle for another condition is
  not supported. The upstream acquisition key is not widened or remapped.
- Condition labels remain exact. Equal voltage, temperature, delay or protocol
  definitions do not merge differently named conditions. Matching labels are
  declarations, not verification of real laboratory conditions.
- Medians use only paired cycles inside the exact device/condition stratum. The
  ratio feature is the median of individual paired ratios, not the ratio of two
  resistance medians. No cross-condition or across-device pooled feature is
  emitted. The unchanged RRAM audit already leaves its overall per-device ratio
  summary null when it has multiple valid condition/voltage strata.
- Completeness, duplicate/missing acquisition rejection, signed nonzero same-sign
  `V/I`, no extrapolation, maximum interpolation gap, explicit physical assumptions,
  input bounds and all-or-nothing outputs remain unchanged.

These rows can support a separately justified condition-aware analysis. They do
not establish that different protocols are comparable or that a condition is an
appropriate model predictor. No condition variable is added to the predictor
allowlist automatically.

## Opt-in outputs and versioning

The same six output files are written to a new private directory. Opt-in
`features.csv` has these columns, in order:

```text
batch,wafer,device,condition,physical_device_group,split_group,read_voltage_V,temperature_K,read_delay_ms,protocol,cycle_count,ratio_below_one_cycle_count,hrs_resistance_median_ohm,lrs_resistance_median_ohm,resistance_ratio_median
```

`physical_device_group` equals the original device ID in every condition row.
`split_group` remains the original batch. The signed read voltage, temperature,
delay and protocol belong to that exact row's condition declaration. Counts,
identities, grouping keys and condition fields are metadata, not automatic model
inputs. The three existing resistance/ratio features remain the only predictors
in the allowlist. No target is generated.

The generic `feature_schema.json` remains the exact seven-field schema version 1
accepted by the current Calculator. Its explicit `row_key` is now
`["device", "condition"]`, and all non-feature columns are listed as metadata.
The envelope supports multi-column row keys already; it does not enforce any
downstream grouping choice.

Opt-in provenance is explicitly schema 3 in both Python and native Rust, with
`example_version: "0.2.0"`,
`workflow_contract: "rram-device-condition-features-v1"`,
`feature_row_unit: "device-condition"`, and the two-column row key. It retains
the exact source input hashes, engine-specific code/runtime provenance, full
condition declarations, all acquisition lineage and per-physical-device lineage.
New `feature_lineage` records, in feature-row order, name the exact device and
condition, original physical-device group, contributing cycles and replicates,
and the physical CSV source row of the condition declaration.

`observed_counts` separates:

- `physical_device_ids`: distinct declared original device strings, not a proof
  of statistical independence.
- `feature_rows`: exported device/condition rows.
- `condition_strata`: distinct exact `(device, condition)` tuples, intentionally
  equal to `feature_rows`; this is not the count of globally distinct condition
  labels. Two devices at one named condition form two strata.
- `paired_cycles`: complete accepted `(device, cycle)` pairs.

Default output columns and semantics are unchanged: example version 0.1.0,
feature schema 1 with row key `["device"]`, Python provenance schema 1 and native
provenance schema 2. New opt-in provenance is not a drop-in replacement for either
legacy provenance schema. Standalone RRAM and I-V report versions do not change.

## Required downstream grouping decision

The Calculator uses exactly one explicitly selected grouping axis. Merely adding
`physical_device_group` to a table or declaring it as metadata cannot prevent a
caller from selecting an inappropriate group. It does not infer nested/crossed
dependencies, choose a row key as the group, or automatically read group choices
from the feature declaration.

For a separately reviewed table with a separately measured endpoint, a workflow
template must therefore name the grouping column explicitly:

```sh
mlcalc run reviewed_labeled_device_conditions.csv \
  --target separately_measured_endpoint --group physical_device_group \
  --features-file feature_schema.json --out NEW_EVALUATION_DIRECTORY
```

Use the deliberately broader `--group split_group` instead when batch is the
justified dependence boundary. Never use condition or a composite device/condition
row key as a substitute for physical-device grouping. All conditions of a device
must stay on the same side of every inner and outer split. A device cannot move
between batches in this exporter, so its rows also stay together under the
retained batch grouping.

The one-device/two-condition example still has only one physical group and is
insufficient even for two group folds. Do not force cross-validation by relabeling
its rows. More rows do not create independent devices or batches.

The template is not permission to train on this synthetic fixture. A real
endpoint must not be manufactured from the same HRS/LRS measurements or their
derived ratio. Feature timing, endpoint ownership, independent unit, sample size,
crossed dependence, intended generalization and scientific comparability still
need separate justification.

## Verification

```sh
python3 -m unittest discover -s examples/measurement-to-features/tests -v
python3 run_tests.py
python3 rust/tests/rram_condition_cli_parity.py --help

# Optional consumer-contract smoke check. Use an existing trusted Calculator
# source and its installed dependencies; this command installs nothing.
python3 examples/measurement-to-features/check_calculator_groups.py \
  --calculator-source /path/to/ml-super-calculator
```

The optional consumer check loads the actual exported CSV through the Calculator's
current CSV reader, verifies the allowlist, and exercises grouped splitting for
six invented devices with two conditions each. It preserves IDs such as `001`
and `1`, checks inner/outer device disjointness for explicit device and batch
axes, and checks that the single-device example fails for insufficient groups.
It creates no target, fits no estimator and reports no prediction score. It is
an interface fixture check, not a scientific validation or a general compatibility
promise for other Calculator versions.

Output safety remains exclusive directory creation, owner-only requested POSIX
permissions, no overwrite, and redacted diagnostics. An output I/O failure can
leave a partial newly created directory. No network service, installation,
credential, private measurement or purchased resource is used by this workflow.
