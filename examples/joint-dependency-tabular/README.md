[English home](../../README.md) · [繁體中文首頁](../../README.zh-TW.md)

# Joint job-or-setting groups: a no-fit handoff

This source-checkout example is entirely invented, uses Python's standard library,
and needs no external repository or private tool. It teaches how to supply a
reviewed component column to Clotho, **not how to discover or compute components**.
It is not a fitted benchmark or evidence of physical independence.

## The declaration comes first

Link specimens when they share **either** the same `job_id` **or** the same complete
ordered `(power, speed, hatch, layer)` setting tuple, then keep all transitively
linked specimens together. The supplied `expected_components.csv` declares:

| Component | Specimens | Why |
| --- | --- | --- |
| C1 | s01, s02, s03, s04 | s01–s02 share job 01; s02–s03 share the full setting; s03–s04 share job 02 |
| C2 | s05, s06 | same job 03 |
| C3 | s07 | neither dependency shared with another row |
| C4 | s08 | neither dependency shared with another row |

In particular, s01 and s04 share neither a job nor a setting directly, but belong
together through the chain. Concatenating `(job, setting)` would split it wrongly.
Treating each setting coordinate as its own linking axis would also be wrong:
all rows share hatch/layer, but that alone is not a repeated **complete** setting.

Identity here is deliberately **exact CSV text**. Job `01` is not job `1`;
setting token `200` is not `200.0`. No trimming, rounding, inferred types or
numeric canonicalization occurs. Thus s07 does not join C1. This is an identity
contract, not a claim that those numeric-looking values differ physically.
If your scientific declaration instead requires numeric equivalence, explicitly
preprocess only named setting columns under a reviewed representation/unit policy,
reject invalid/nonfinite values, retain original tokens and transformation
provenance, and rebuild/review the map. Do not normalize identifiers or erase
leading zeroes. Clotho's later floating-point features do not redefine this map.

## Validate before any fit

From the repository checkout containing this example:

```bash
python examples/joint-dependency-tabular/validate.py
python -m unittest discover -s tests -p 'test_joint_dependency_example.py' -v
# Optional: creates a NEW file; fails rather than overwriting an existing path.
python examples/joint-dependency-tabular/validate.py --output joint-example.csv
```

The validator requires unique IDs, exact headers and the same complete eight-ID
roster on both sides; **both unmatched counts must be zero**. It preserves source
row order and tokens. An inner join that silently drops unmatched rows is not a
successful handoff. Its hard-coded fixture oracle rejects both split chains and
unjustified merges; direct job/setting edges must remain within components.
This is intentionally fixture-specific, not a general component generator or a
scientific audit for a different dataset. Review the dependencies and complete
mapping separately for real inputs; passing this example cannot validate them.

## Supply the component column and an explicit feature list

The following is a **plan only**, not an executed result or instruction to fit:

```bash
mlcalc plan run joint-example.csv \
  --target response --group component_id \
  --features power,speed,hatch,layer \
  --outer-splits 4 --inner-splits 2 --objective group_mae
```

Four components make four outer folds and two inner folds feasible here; the
default five outer folds is not feasible. Do not split components or revert to
random rows to meet a fold count. If dependencies collapse the cohort too far,
stop and change the evaluation question or collect appropriate independent data.

Without `--features` / `--features-file`, eligible non-target/group columns become
features. Here the whitelist excludes `specimen_id`, `job_id`, and
`post_outcome_note`; `component_id` is only the split group and `response` only the
target. Exclude identifiers, outcome-derived quantities and unavailable future
measurements deliberately—do not rely on their names being recognized automatically.

## Read both existing metrics

Clotho already reports **Group MAE** and **Row MAE**. With `component_id` as group,
Group MAE averages the four component MAEs equally (each weight 1/4), whereas Row
MAE averages eight specimen errors (component weights 4/8, 2/8, 1/8, 1/8 here).
`--objective group_mae` selects candidates using the former, not the latter;
training losses do not thereby become component-uniform. Inspect all per-group
errors too. Components are declared dependency blocks, not proven independent
samples, and neither weighting proves transfer, causal effects or sufficiency.

No models, predictions or performance scores are supplied by this example.
