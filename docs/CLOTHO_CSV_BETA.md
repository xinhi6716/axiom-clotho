[Axiom Clotho](../README.md) / RRAM CSV workflow

# Axiom Clotho — a complete, bounded CSV workflow

**Moirai family · local regression beta · RRAM v1 contract.**

Version: **0.3.0b5**. Blank-input setup is available through [clotho init](WORKFLOW_INIT.md); frozen b3 lacks init. The distribution remains `ml-super-calculator` so existing
imports and scripts have one owner. See the [version/upgrade boundary](CLOTHO_BETA_MIGRATION.md).

> **FET transfer data:** b5 retains [FET device regression](FET_DEVICE_REGRESSION.md)
> with the explicit v2 composite-device contract. The remainder of this page
> describes **RRAM v1**, not FET.

Clotho now does more than run a bundled synthetic demonstration: it accepts your
own declared measurement CSVs, joins your own training endpoints by exact device
ID, compares models with nested group-disjoint validation, predicts an explicitly
separate target-free cohort, then evaluates those saved predictions only when
you supply later outcomes. It never creates labels, chooses a scientific grouping
boundary for you, downloads a model, or sends measurements to a hosted service.

`clotho` is an additive entry point. `mlcalc`, `groupbench`, `mlcalc-workflow` and
the `groupbench` Python APIs are preserved. No frozen a4 asset or tag is changed.
From source, replace `clotho` with `PYTHONPATH=src python -m mlworkflow`.

## The four commands

Use a fresh **sibling** output directory at each stage. Never put a result inside
its input bundle: doing so would invalidate that bundle's exact file inventory.

```bash
clotho prepare --input-dir measurements \
  --partitions partitions.csv --training-targets training_targets.csv \
  --config analysis.json --output-dir prepared

clotho plan --prepared prepared

clotho run --prepared prepared --output-dir model-run

# Only after predictions exist and later observations are actually available:
clotho score --run-dir model-run --outcomes later_outcomes.csv --output-dir evaluation
```

Open `model-run/report.html` for the model comparison and device-keyed predictions;
open `evaluation/report.html` for exact-cohort saved evaluation. Both are standalone
offline HTML with no remote fonts, scripts or assets. Keep all bundle files together.

| Stage | Reads | Produces | Fitting |
| --- | --- | --- | --- |
| `prepare` | Raw measurement files, explicit partitions, training-only targets, configuration | Feature provenance, exact joined training/future tables, completion receipt | None; standard library only |
| `plan` | Complete prepared bundle | JSON to stdout with exact nested fit counts and input-support diagnostics | None; requires the existing ML stack for validation |
| `run` | Complete prepared bundle | Nested comparison, target-free predictions, linked report and completion receipt | Existing calculator algorithms only |
| `score` | Complete run bundle and later outcome CSV | Existing scorecard metrics, exact joined rows, report and completion receipt | None; standard library only |

The preparation and run commands accept **no future-outcome path**. They cannot
prove a human has never seen those outcomes, or that files were historically
frozen. No caller-supplied prediction-cohort outcome path is accepted or opened before
`score`. Component integrity checks may read the unrelated bundled legacy
invented fixture; they do not supply labels to this workflow. The older bundled
`demo` is a different fixed-example route, with its own documented preflight
hash/copy behavior.

## Your input contract

### 1. Measurement directory

This beta supports the existing strict RRAM measurement-to-features protocol,
not every format in the toolbox. The three files are:

| File | Required unique columns |
| --- | --- |
| `manifest.csv` | `batch,wafer,device,cycle,replicate,branch,state,condition` |
| `conditions.csv` | `condition,read_voltage_V,temperature_K,read_delay_ms,protocol` |
| `iv_sweeps.csv` | `device,cycle,branch,voltage,current` |

Column order may differ; unexpected or duplicate columns are rejected. Voltages
are volts, currents amperes. States must be explicitly HRS/LRS. Measurement keys,
paired states and repeat identities must agree. No state, unit conversion, missing
measurement, physical identity or endpoint is inferred.

Read the full [upstream measurement assumptions and format](../src/mlworkflow/_vendor/semiconductor/examples/measurement-to-features/README.md).
Here, one feature row represents one globally identified physical device at a
fixed condition, aggregating its repeated cycles. **Device-condition rows are not
supported by this supervised beta route**; the separate `features` command still
supports the existing opt-in export without training.

### 2. Explicit device partitions

`partitions.csv` contains exactly `device,partition`. Each extracted device must
occur once, with partition exactly `train` or `predict`. No missing, extra,
duplicate or silently discarded IDs are allowed. Rows are joined by exact text
identity, never row position, whitespace stripping or numeric coercion.

The configured `group_by` is one of `batch`, `wafer`, or `device`, taken from the
measurement manifest. Training and prediction groups must be disjoint. Wafer IDs
must not map to multiple batches. IDs must be globally namespaced by you: this
cannot discover aliases or tell you which physical level is scientifically right.
Broader shared environments, time periods, instruments or dependencies can remain.

### 3. Training-only endpoints

`training_targets.csv` contains exactly `device,target`, and exactly the **train**
device IDs. Targets must be supplied finite decimal measurements in your declared
unit. A prediction-device label in this file is an error, not a convenient join.
The tool does not derive a target from HRS/LRS features, impute missing targets,
choose endpoints, or decide whether their timing and semantics are meaningful.

`later_outcomes.csv` has the same two-column format, but must match exactly the
saved prediction cohort. Partial scoring, duplicate IDs, extra rows and omitted
devices are rejected. Keep a genuinely prospective or held-out endpoint file
separate until the scientific protocol permits scoring.

### 4. Explicit configuration

Copy [the annotated usage below](#configuration-example) or the
[invented fixture configuration](../examples/measurement-regression/fixtures/config.json),
then replace it with justified choices. Every field is required; unknown fields
are errors, not silently ignored options.

The endpoint name, unit, grouping and declarations are **caller assertions**, not
facts established by this program. Do not set a declaration to true merely to
make validation pass.

### Configuration example

```json
{
  "schema_version": "measurement-regression-v1",
  "endpoint_name": "REPLACE WITH YOUR ACTUAL MEASURED ENDPOINT",
  "target_unit": "REPLACE WITH YOUR ACTUAL UNIT",
  "group_by": "batch",
  "features": [
    "hrs_resistance_median_ohm",
    "lrs_resistance_median_ohm",
    "resistance_ratio_median"
  ],
  "max_bracket_span_V": "0.11",
  "outer_splits": 3,
  "inner_splits": 2,
  "baselines": ["dummy", "ridge"],
  "seed": 7,
  "objective": "group_mae",
  "max_fits": 200,
  "declarations": {
    "global_physical_ids": true,
    "pre_outcome_features": true,
    "endpoint_not_derived_from_features": true,
    "measurement_assumptions_accepted": true
  }
}
```

`0.11` is an example readpoint bracket limit, not a recommended physical protocol.
The three predictors are an explicit allowlist; select a nonempty subset. IDs,
group/condition labels, cycle counts and targets cannot become predictors through
this route. `dummy` and at least one of `ridge` or `extra_trees` are required.
`group_mae` gives each training group equal weight for inner selection. Outer
comparison estimates the selection procedure, not an extra model-selection stage.

The no-fit plan validates the actual grouped split schedule and checks that
comparison plus full-data-refit pipeline fits do not exceed `max_fits`. This is
a count of planned pipeline fits, **not** a wall-clock/memory reservation. Small
training-group counts can make nested comparison infeasible; do not repair that
by pretending repeated cycles are new independent devices.

## What to inspect

- `prepared/features/`: unchanged measurement feature/schema/lineage reports.
- `prepared/provenance.json`: hashes of captured input bytes and exact-join scope.
- `model-run/comparison/report.html`: nested grouped comparison of requested
  families and the inner-selected procedure, with per-group error diagnostics.
- `model-run/prediction_rows.csv`: stable device/group keys beside each prediction.
- `model-run/prediction/manifest.json`: full-data refit selection and input support.
- `evaluation/scores.json`: row metrics, equal-group metrics, undefined-R² status,
  exact source receipt/outcome hashes, and `fits_performed: 0`.
- Every `COMPLETE.json`: fixed output inventory and byte digests. An
  `INCOMPLETE.json`, missing file, unexpected file, altered bytes or symbolic link
  makes a saved bundle unusable. Never manually delete a failure marker to make
  the bundle appear successful; retry in a fresh directory.

Outputs are private-mode local files where POSIX permissions apply. I/O failure
can leave a deliberately incomplete directory. Input files are read into bounded
snapshots; the workflow does not modify them. Use trusted, idle local directories:
these checks are not filesystem transactions or authentication against a hostile
concurrent writer. Digests and recorded declarations do not establish scientific
validity, historical blinding, correct IDs or independence.

## Existing calculator capabilities remain the core

```bash
clotho tabular --help
clotho tabular plan run TRAIN.csv --target TARGET --group GROUP
clotho tabular binary --help
```

`tabular` delegates unchanged to the calculator CLI, including general regression,
binary classification, no-fit planning, saved-bundle verification and existing
saved analyses. Existing `mlcalc ...` commands work directly. Compatibility
messages may retain the calculator's original names and version; no frozen
release identity is rewritten.

## Beta boundaries, not hidden gaps

- The new measurement-supervised route is regression only, one row per device,
  with three declared measurement predictors. It does not cover every toolbox
  tool, chronology/Native400 data, automatic unit conversion, arbitrary endpoint
  joins, automatic feature discovery or arbitrary dependence graphs.
- Binary classification remains available through the calculator core, not this
  new measurement join schema.
- Predictions are not intervals, calibrated uncertainty, causal effects or
  evidence of applicability to another population. A single future group is
  scored descriptively, not turned into an independent-sample inference claim.
- The reusable scorecard is pinned byte-for-byte from research-workflow-utilities,
  with its MIT notice. Measurement tools likewise remain pinned; no algorithm is
  maintained as a divergent copy. `clotho components` shows exact source identities.
- No new runtime dependency, model download, paid service, executable model
  persistence, repository visibility change or external publication is performed.

## Reproducible software example

Follow [the complete invented example](../examples/measurement-regression/README.md).
It uses arbitrary literal endpoints and repeated numerical templates from the
existing walkthrough. Passing it is software evidence, **not real-data validation**.

[← Clotho overview](../README.md) · [返回 Clotho 總覽（繁體中文）](../README.zh-TW.md)
