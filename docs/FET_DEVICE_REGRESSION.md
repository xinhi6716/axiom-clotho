[Axiom Clotho](../README.md) / FET device regression

# FET device regression — Clotho 0.3.0b5

The b5 wheel retains the FET v2 adapter and pinned extractors. Use its matching
source archive for the invented input files below. Frozen b2 and earlier assets
do not include this route. [Installation and release scope](RELEASE_NOTES_0.3.0b5.md).

This bounded route connects a declared transfer table to the existing
**prepare → plan → run → score** workflow. It reuses the pinned signed-current
criterion and windowed log-slope extractors and the existing grouped ML core.
It does not add an estimator, infer an endpoint, discover a physical regime,
import arbitrary instrument exports, or migrate the whole private toolbox.

The existing [RRAM v1 contract](CLOTHO_CSV_BETA.md) is separate and unchanged.
FET uses explicit `measurement-regression-v2` with adapter
`fet-transfer-device-v1`; it is not an automatic reinterpretation of v1 files.
The standalone `clotho features` command retains its existing RRAM semantics.

## 1. One acquisition per composite physical device

The input directory contains `transfer.csv` with exactly these unique columns:

```csv
lot,wafer,device,condition,replicate,branch,temperature_K,drain_voltage,body_voltage,gate_voltage,drain_current
```

- A physical device is the exact tuple **`(lot, wafer, device)`**. Local wafer and
  device labels may repeat in different lots; do not join on `device` alone.
- An acquisition is the six-ID tuple
  `(lot, wafer, device, condition, replicate, branch)`.
- This first adapter accepts **exactly one complete acquisition branch per
  physical device**. A second condition, recorded repeat or branch for that
  device rejects. No averaging, first-passing selection, fallback acquisition,
  trimming or deduplication occurs.
- The supplied table is the declared all-row cohort. Both extractors must pass
  every branch; an audit-only or partial feature result cannot enter ML.
- The voltage columns are source-referenced `VDS`, `VBS` and `VGS`, respectively.
  A branch requires fixed nonzero VDS, fixed VBS, positive fixed Kelvin
  temperature, and strictly monotonic VGS under the upstream contracts. Supply
  signed terminal currents, not current density or geometry-normalized values.

Prepare a justified canonical table before using this route. The upstream
single-group selector is not automatically invoked and does not constitute a
batch selector for all devices. If you use a separate preparation step, retain
its original source and original-to-subset mapping. Software cannot infer raw
instrument units, measurement chronology, acquisition IDs or correct branch
meaning from a label such as `forward`.

### Exact endpoint and partition joins

`partitions.csv` contains exactly:

```csv
lot,wafer,device,partition
```

Every extracted physical device must occur exactly once, with `partition` equal
to `train` or `predict`. `training_targets.csv` contains exactly:

```csv
lot,wafer,device,target
```

Its keys must equal the **training devices only**. Later outcomes use the same
four columns, but their keys must equal the complete saved prediction cohort.
Missing, duplicate or extra keys reject; no rows or labels are silently filled,
dropped or joined by position. Identity text is preserved, including leading
zeros and case; padded or unsafe IDs reject rather than being normalized.

The actual endpoint must be separately sourced, available at the declared time,
and not the same feature or its algebraic proxy. Those are scientific
responsibilities, not facts the program can verify from column names.

### Grouping

`group_by` is explicitly one of:

| Setting | Physical group |
| --- | --- |
| `lot` | `(lot)` |
| `wafer` | `(lot, wafer)` |
| `device` | `(lot, wafer, device)` |

Training and prediction groups must be disjoint on these original tuples.
Internal `row_id`/`group` tokens and separate token namespaces do not establish
physical disjointness. Training token assignment and row order depend only on
training tuples, not prediction-cohort ordering. A saved identity map preserves
all six acquisition fields and their partition.

Enough distinct training groups must remain for nested comparison. Do not fix
an infeasible split by treating repeated acquisitions as independent devices.
Aliases, shared environments and other unrecorded dependence may still remain.

## 2. Explicit v2 configuration

Every field is required; unknown fields and mixed versions reject. This is the
[invented example configuration](../examples/fet-device-regression/fixtures/config.json),
not recommended physical settings:

```json
{
  "schema_version": "measurement-regression-v2",
  "adapter": "fet-transfer-device-v1",
  "endpoint_name": "invented_later_response",
  "target_unit": "arbitrary_units",
  "group_by": "lot",
  "features": [
    "criterion_gate_source_voltage_V",
    "log_current_slope_decades_per_V"
  ],
  "measurement": {
    "voltage_unit": "V",
    "current_unit": "nA",
    "criterion_current": "50",
    "criterion_current_unit": "nA",
    "current_polarity": "positive",
    "max_bracket_span": "0.1",
    "window_min": "0.1",
    "window_max": "0.4"
  },
  "outer_splits": 2,
  "inner_splits": 2,
  "baselines": ["dummy", "ridge"],
  "seed": 1729,
  "objective": "group_mae",
  "max_fits": 500,
  "declarations": {
    "global_physical_ids": true,
    "pre_outcome_features": true,
    "endpoint_not_derived_from_features": true,
    "measurement_assumptions_accepted": true
  }
}
```

Measurement decimals are strings without embedded units. `voltage_unit` is one
of `V`, `mV`, `uV` and applies to all three voltage columns, the positive bracket
span and both ordered window bounds. Current and criterion units are independently
one of `A`, `mA`, `uA`, `nA`, `pA`. `current_polarity` is `positive` or `negative`;
a negative polarity requires a negative criterion. Polarity is not transistor
type. There is no absolute-value rescue for a conflicting sign.

Choose a nonempty unique subset of the two feature names above:

- **Criterion gate-source voltage:** defined by the declared signed current and
  bounded local interpolation. It is not a certified physical threshold voltage.
- **Windowed log-current slope:** a descriptive slope in decades/V. Its reciprocal
  is not automatically a valid subthreshold swing or evidence of a physical
  regime. A valid zero slope is retained, not treated as missing.

IDs, fixed biases, criterion/window choices, counts and diagnostic values are
metadata, not predictors. Extraction values outside the selected ML feature
numeric domain reject; they are not silently clipped or rescaled. The same
measurement settings apply to every branch in a run; do not tune them using
outcomes or model results.

`dummy` and at least one of `ridge` or `extra_trees` are required. The existing
`group_mae` objective and nested model-selection machinery are reused. `max_fits`
bounds planned ML pipeline fits, not elapsed time or memory. Declarations are
caller assertions: do not set them true merely to make validation pass.

## 3. Same four stages

After installing b5, run these commands from its matching extracted source
archive or tagged checkout. Plan/run require the existing ML dependencies. The
commands do not fetch a private repository or download data. Choose fresh output directories
outside the input/prepared/run directories; never modify completed bundles.

```sh
clotho prepare \
  --input-dir examples/fet-device-regression/fixtures/raw \
  --partitions examples/fet-device-regression/fixtures/partitions.csv \
  --training-targets examples/fet-device-regression/fixtures/training_targets.csv \
  --config examples/fet-device-regression/fixtures/config.json \
  --output-dir fet-prepared

clotho plan --prepared fet-prepared

clotho run \
  --prepared fet-prepared --output-dir fet-run

# Only this separate command accepts the prediction cohort's later outcomes.
clotho score --run-dir fet-run \
  --outcomes examples/fet-device-regression/fixtures/heldout_outcomes.csv \
  --output-dir fet-score
```

| Stage | Scope |
| --- | --- |
| Prepare | Extract measurement descriptors, validate exact composite joins, save separate training/prediction inputs. **Zero ML fits**, but descriptive log-slope least squares is performed. |
| Plan | Validate the actual grouped schedule and ML fit budget. Zero ML fits. |
| Run | Compare within training groups, refit the selected procedure on training data and save predictions. No prediction-cohort outcome input is accepted. |
| Score | Join later outcomes to the complete saved prediction cohort and compute descriptive metrics. No ML import, training, selection or refit. |

The [original invented example](../examples/fet-device-regression/README.md)
has 105 transfer observations, 12 training devices in four lots, and three future
devices in a separate lot. Its endpoints are independently supplied invented
constants, not measured evidence or derived extraction targets. It demonstrates
software behavior, not device accuracy, physical independence, historical
blinding, uncertainty intervals or generalization benefit.

## 4. Saved evidence and standalone-score limits

A complete prepared v2 bundle contains the joined training/future tables,
`cohort.csv`, `identity_map.csv`, configuration/provenance, and eight retained
upstream artifacts under `features/`, including **the exact raw transfer bytes**.
It uses a fixed versioned inventory and a `COMPLETE.json` receipt. Missing or
unexpected files, altered bytes, symbolic links and incomplete output reject.
Do not manually repair a bundle or discard the map to make it smaller.

Prepared-bundle validation checks the feature/report lineage and equality of
feature, cohort, model-row and identity-map keys/values. A run retains the complete
training-and-prediction identity map, device-keyed predictions, engine prediction
rows and the saved prepared-file hashes.

Standalone score checks the retained map's hash against the saved preparation
metadata, composite tuple/prefix/partition consistency, canonical map ordering,
exact prediction identity and engine row-position/value alignment. It **does not
reopen or rederive omitted raw measurements or feature CSVs**. Inconsistent
remaps, duplicate/orphan identities and incomplete outcomes reject, even when
file checksums alone appear valid. These checks do not authenticate a coordinated
rewrite of all evidence or prove historical physical identity.

Scores are descriptive for this exact cohort; later outcomes are not a new tuning
or model-selection input. Keep adverse results and all cohort rows rather than
scoring only favorable devices.

## 5. Locality, source and data rights

Preparation and score use the standard library and pinned resources. The adapter
uses the same existing grouped ML core for plan/run, with no hosted API, telemetry,
new model download, credential requirement or private-toolbox runtime checkout.

Real raw transfer snapshots, six-part acquisition identities, targets,
predictions and outcomes remain **private local user artifacts** unless sharing
is separately authorized. A bundle can expose more than its visible report.
Review the whole archive before sending it anywhere. MIT software attribution
does not grant permission to redistribute experimental data or private research.

Pinned upstream assumptions:
[signed-current criterion](../src/mlworkflow/_vendor/semiconductor/tools/vth-constant-current/README.md),
[windowed log slope](../src/mlworkflow/_vendor/semiconductor/tools/transfer-log-slope/README.md),
and [joint FET workflow](../src/mlworkflow/_vendor/semiconductor/examples/fet-measurement-to-features/README.md).
Their optional generators, selectors, annotations and report companions are
upstream reference context, **not a claim that all those files ship with Clotho**.
The example above is the supported FET integration route. From an uninstalled
source checkout, replace `clotho` with `PYTHONPATH=src python -m mlworkflow`.

[← Clotho overview](../README.md) · [返回 Clotho 總覽（繁體中文）](../README.zh-TW.md)
