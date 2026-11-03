# Optional grouped models → new rows → later outcomes

[Examples](../README.md) · [繁體中文](README.zh-TW.md)

This is a **software walkthrough with invented data**, not a scientific benchmark.
The 18 training rows form six invented groups; three new rows form two declared
scoring groups. `x` and `response` use arbitrary invented units, not physical
measurements. `kind` is an invented category; empty training categories and the
new category `new` exercise missing/unseen-category handling. The outcomes were
invented in advance for this example, not collected later or kept blind.

## Requirements and scope

Use a matching source checkout containing these fixtures and a Python environment
with Clotho's dependencies and the optional-family and `score-predictions`
commands. These are **current-source features**; do not assume an older installed
release has them. From the repository root, check `clotho tabular predict --help`
and `clotho tabular score-predictions --help`. If running from source instead,
prefix commands with `PYTHONPATH=src` and replace `clotho tabular` with
`python -m groupbench`. No extra model dependencies are needed.

The commands below use a POSIX shell. Run from the repository root and choose
fresh `runs/optional-later-evaluation` and `runs/optional-later-prediction` paths.
Do not use `--overwrite`. Training steps write bundles; planning and scoring print
JSON without writing files. Do not redirect score output onto an input: a shell
can truncate files before Clotho starts.

## 1. Preview the grouped evaluation budget (no fitting)

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
clotho tabular plan run examples/optional-model-delayed-scoring/training.csv \
  --target response --group study --features x,kind \
  --baselines nystroem_ridge,group_median --categorical-encoding distinct-v2 \
  --inner-splits 2 --outer-splits 3
```

The plan reports 30 pipeline fits, with zero fits performed by planning. Both
families are explicitly opted in; defaults remain unchanged. `study` is the split
group, never a model feature. `group_median` gives training groups equal weight;
it does not learn a separate lookup prediction for each incoming group. See the
[optional family guide](../../docs/OPTIONAL_REGRESSION_FAMILIES.md) for limits.

## 2. Evaluate on held-out training groups (30 tiny pipeline fits)

```sh
clotho tabular run examples/optional-model-delayed-scoring/training.csv \
  --target response --group study --features x,kind \
  --baselines nystroem_ridge,group_median --categorical-encoding distinct-v2 \
  --inner-splits 2 --outer-splits 3 --out runs/optional-later-evaluation
```

Open `runs/optional-later-evaluation/report.html`. These invented metrics only
check the software path. They do not show that one family is scientifically
better, nor justify tuning based on the later-outcome file.

## 3. Predict three feature-only rows (10 tiny pipeline fits)

```sh
clotho tabular plan predict examples/optional-model-delayed-scoring/training.csv \
  examples/optional-model-delayed-scoring/new_features.csv \
  --target response --group study --features x,kind \
  --baselines nystroem_ridge,group_median --categorical-encoding distinct-v2 \
  --inner-splits 2
clotho tabular predict examples/optional-model-delayed-scoring/training.csv \
  examples/optional-model-delayed-scoring/new_features.csv \
  --target response --group study --features x,kind \
  --baselines nystroem_ridge,group_median --categorical-encoding distinct-v2 \
  --inner-splits 2 --out runs/optional-later-prediction
clotho tabular validate-prediction-bundle runs/optional-later-prediction
```

`predict` selects and refits using training data only; it does not reuse fitted
models from step 2. The prediction plan reports 10 fits; the plan itself performs
none. `new_features.csv` contains no labels or groups. Its saved row positions
are zero-based: 0 = x 1.5, 1 = x 9.5, 2 = x 20. Preserve this mapping; row position
is not a physical device identifier. Bundle validation checks artifact structure
and integrity, not measurement truth or cohort independence.

## 4. Score the complete later-outcome cohort (zero fitting)

```sh
clotho tabular score-predictions --run-dir runs/optional-later-prediction \
  --outcomes examples/optional-model-delayed-scoring/later_outcomes.csv
```

The exact header is `row_position,group,observed`. The fixture deliberately lists
positions **2, 0, 1**: joining is by the explicit position, never CSV row order.
Every saved position must appear exactly once. The supplied literal scoring
groups are `queryA` (one row) and `queryB` (two rows); they are declarations for
aggregation, not inferred training groups or verified identities.

Success prints JSON with `fits_performed: 0`, three scored rows, two groups,
row-weighted and equal-group summaries, and exact input hashes. Singleton-group
R² is null. Row and equal-group averages need not match. The command consumes the
saved predictions without loading a model or rereading training features; it does
not refit, calibrate, silently drop rows or alter the bundle. Missing/duplicate
positions, invalid numbers or incomplete cohorts reject the whole score.
A valid bundle can still be numerically unscorable under stricter conversion
checks; see [the full scoring contract](../../docs/SAVED_PREDICTION_SCORING.md).

Do not treat this manufactured cohort as independent evidence. In a real study,
freeze the feature order, row-identity mapping, group definitions and outcome
protocol before inspecting outcomes. Hashes demonstrate consumed bytes, not
provenance authenticity. Score JSON includes supplied group/target/prediction
values; review it before sharing. Generated bundles are intentionally not shipped
with this example.
