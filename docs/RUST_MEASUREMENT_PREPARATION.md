# Native exact measurement joins

`prepare-measurements` connects extracted numeric features to externally supplied
training endpoints without reading prediction-cohort labels or fitting a model.
It emits tables consumable by native `fit`, `evaluate`, `compare` and `predict`.
It uses schema `clotho.measurement_join.v1`, **not** legacy prepared-bundle parity.

```sh
clotho prepare-measurements --csv features.csv \
  --partitions partitions.csv --training-targets training_targets.csv \
  --identity lot,wafer,device \
  --features criterion_gate_source_voltage_V,log_current_slope_decades_per_V \
  --group-prefix 1 --out new_prepared_directory
```

For FET, `features.csv` can be the output of `extract-fet`. Identity columns are
explicit, ordered composite keys, not concatenated strings. Prefix 1 means lot,
prefix 2 means lot/wafer and prefix 3 means lot/wafer/device for the command above.
These boundaries are caller declarations, not proof of physical independence.

Input contracts:

- Feature rows: exactly one row per declared identity, explicit finite numeric
  feature columns. Additional columns are ignored, never automatically selected.
- Partitions: exactly the identity columns plus `partition`, whose values are
  `train` or `predict`. Every feature identity must occur exactly once.
- Training targets: exactly identity columns plus `target`. Every training
  identity must occur exactly once and no prediction identity may occur.
- Both partitions must be nonempty; group prefixes must be disjoint across them.
- Identity/feature names are disjoint and exclude reserved names `row_id`,
  `group`, `partition`, `target`; unsafe CSV formula prefixes and controls fail.
- Duplicate, orphan, missing, malformed and nonfinite entries reject the whole
  preparation. No missing target is imputed, derived, clipped or dropped.
- A nonzero numeric literal that parses to zero is rejected. The numerical
  contract otherwise uses binary64, not exact source-decimal arithmetic.

The fresh output directory contains:

- `training.csv`: canonical identity order, row_id, group, target, selected features.
- `future.csv`: canonical identity order, row_id, group, selected features, no target.
- `preparation.json`: exact identities, partition mapping, tuple groups, selected
  columns, counts, limits and zero-fit statement.
- `COMPLETE`: written last after successful outputs.

Group values are JSON-encoded tuples, preventing delimiter-collision joins.
They remain opaque strings to the native modelling commands. Files are exclusive
and owner-only on Unix; the destination must not already exist. All input checks
precede writing. An I/O failure may leave an incomplete new directory.

The Rust core API `prepare_measurements` is independent of CSV. It limits 20,000
rows, 100 numeric features and 1–6 identity components, each at most 128 characters.
This stage does not authenticate sources, detect identity aliases, establish
feature availability before outcomes, infer causal validity, select safe features
for an endpoint or prove absence of target proxies. The caller must justify these.
Do not publish private measurement/endpoint artifacts without permission.

No raw input snapshots, SHA manifests, saved scoring or legacy bundle loading are
claimed by this command. Existing Python prepared workflows remain available until
those migration acceptance gates are implemented and independently reviewed.

## Verification

The combined native adapter workspace passes 261 tests on Rust 1.70.0 and
1.99.1. This join slice contributes 16 core and 8 CLI tests, including a complete
prepare → native mean fit → predict example. Independent review verified tuple
joins, ordering and future-label exclusion, and identified unsafe CSV header and
non-regular-input admission gaps; both were corrected with regression tests.
These local checks do not claim hosted CI or scientific validation.
