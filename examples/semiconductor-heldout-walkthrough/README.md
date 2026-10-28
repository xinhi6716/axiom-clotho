# A semiconductor measurement walkthrough

[Clotho home](../../README.md) · [All examples](../README.md) · [繁體中文總覽](../../README.zh-TW.md)

Follow invented I–V measurements through feature extraction, a lot-level holdout check, model fitting and separate scoring. This source-only walkthrough connects Clotho to a trusted local Semiconductor Toolbox checkout. It runs:

1. The Semiconductor Toolbox's synthetic I–V generator.
2. Its measurement-to-features exporter, including read-point interpolation,
   paired HRS/LRS resistance summaries, units and lineage.
3. Its **actual lot-level holdout audit** on the exact device assignments used
   downstream. A failed audit stops before model fitting.
4. Clotho's fixed environment probe: dummy, feature sets A and B, and two
   fits sharing one within-training-environment label shuffle.
5. Separate exact-device scoring, followed by a readable `INTERPRETATION.txt`.

The walkthrough runs locally and downloads nothing. Use trusted source checkouts: the wrapper executes their Python scripts. Recorded source hashes identify selected files, not authenticated software or a complete dependency lock.

## Requirements and run command

Use an existing Python environment with Clotho's declared
NumPy, pandas, scikit-learn and threadpoolctl dependencies. The tested environment
is Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0. This source
example is not included in the installed `mlcalc` command.

You need a trusted local Semiconductor Toolbox checkout containing:

- `examples/measurement-to-features/generate_synthetic.py`
- `examples/measurement-to-features/build_features.py`
- `tools/physical-holdout-audit/physical_holdout_audit.py`
- `tools/iv-readpoint/iv_readpoint.py` and `tools/rram-window/rram_window.py`

You also need this Clotho checkout's
`examples/environment-heldout-comparison/probe.py` and `src/groupbench/`.
No assumption is made that these source companions are published or installed.
Supply the matching reviewed source versions available to you; missing companions
fail without downloading a replacement.

From the Clotho repository root, **export** `SEMICONDUCTOR_TOOLBOX_SOURCE`
with your existing trusted local Toolbox directory as its value, then run the
following exact commands. It must be an exported environment variable, not only
an unexported shell variable, so the later tests can see it too.
The new `runs/semiconductor-walkthrough-output` directory must not already exist:

```bash
mkdir -p runs
PYTHONDONTWRITEBYTECODE=1 python examples/semiconductor-heldout-walkthrough/run.py \
  --semiconductor-source "$SEMICONDUCTOR_TOOLBOX_SOURCE" \
  --calculator-source "$PWD" \
  --output-dir runs/semiconductor-walkthrough-output \
  --run-invented-example
cat runs/semiconductor-walkthrough-output/INTERPRETATION.txt
```

The flag explicitly acknowledges that this is the fixed invented example. The
CLI does not accept a real dataset, custom target, feature search or alternative
split. Subcommands use the same Python interpreter, a train-only preprocessing
pipeline, one numerical thread and a 120-second timeout per stage. This is not a
sandbox or a comprehensive CPU/memory quota. New output directories have mode
0700; wrapper-created files have mode 0600 where POSIX modes apply. No existing
output directory or symlink is reused. Failed runs leave a categorical
`FAILED.json`, never `COMPLETE.json`; preserve it for diagnosis and use a fresh
output destination for another attempt. The wrapper suppresses raw subprocess
stderr, so check the stage name and the upstream companion's documented input and
dependency contract when diagnosing a rejection.

## What the invented rows mean

`cohort.csv` names fabrication, lot, batch, wafer, physical device, environment,
partition and declared dependency block. Every ID starts with `invented_fab`.
There are 18 device **labels**, three lots/environments, six batches/wafers,
54 paired HRS/LRS cycle labels and 324 raw I–V rows. The first two lots supply
12 training devices and the third supplies six held-out devices. Declared wafer
blocks are disjoint, and the separate audit additionally enforces lot separation.

**The three environments repeat the same six numeric device templates under
new labels.** This is a deterministic software fixture, not 18 independently
sampled devices, three independent fabrication runs, a transfer benchmark or a
physics-validated simulator. The generator uses `I = V * G` arithmetic. Repeated
cycles are aggregated within device rather than treated as independent devices.
Neither disjoint strings nor the audit verifies real physical identities or
unmeasured dependence.

The two checked-in endpoint CSVs contain literal arbitrary test values, not
values derived from HRS, LRS, ratio or a fitted model. Their target unit is
`arbitrary_test_units`, not a named device-performance quantity. Training targets
join exactly the 12 training device IDs. The held-out endpoint file is not opened,
copied or hashed by this workflow until the separate scoring command, **after
prediction has succeeded**. `future.csv` contains no target column. Only after
scoring is the held-out fixture copied, with its bytes checked against the score's
input hash. This order demonstrates an API boundary, not historical blinding of
these public fixtures or proof that a researcher did not previously see outcomes.

## Read the comparison

The feature lists, held-out environment, seed 7 and Ridge alpha 1 are fixed before
the scoring command. Feature set A uses HRS median resistance (ohm), LRS median
resistance (ohm) and median paired HRS/LRS ratio (dimensionless). Feature set B uses
only that ratio. These are related summaries of the same measurements, **not
independent causal mechanisms**. IDs, counts, read settings and condition labels
are metadata, not predictors. Median imputation, scaling and fitting use only
training rows. There is no search, tuning or selected winner.

`INTERPRETATION.txt` reads the actual outputs and reports MAE/RMSE for all five
arms plus the paired mean absolute-error contrast B minus A on exactly the same
six scored devices. A negative contrast means lower B absolute error on these
invented rows only. A shuffled-label arm can outperform an unshuffled arm here;
the wrapper neither conceals this nor redraws the shuffle to force degradation.

The environment label here is the lot, and the comparison measures prediction error, not a causal effect. The within-environment shuffle can leave labels unchanged and may break wafer-level dependence. Treat it as a diagnostic, not a permutation test or causal negative control. The report supplies no p-values, confidence intervals or null distribution.

For real research, separately justify the endpoint, pre-outcome predictor timing,
units, dependence boundary, identity accuracy, cohort completeness, sampling,
source rights and analysis freeze. Do not create the target from the same
resistance summaries used as predictors and call that independent prediction.
This fixed example is not a license to substitute private data into public files.

## Outputs and checks

- `raw/`, `features/`: invented measurements and unchanged exporter outputs.
- `cohort.csv`, `assignments.csv`, `holdout.json`: explicit identities and the
  actual lot-boundary check for the downstream 12/6 assignment.
- `training.csv`, `future.csv`, `spec.json`: model inputs and fixed settings.
- `predictions.json`: five fitted-state fingerprints, prediction cohort,
  train-only fitting declarations, and shared shuffle details.
- `scores.json`, `heldout_outcomes.csv`: all six exact held-out outcomes and
  descriptive errors, written/read only in the post-prediction scoring phase.
- `INTERPRETATION.txt`: the human-readable result and limitations.
- `COMPLETE.json`: last-written receipt, selected component hashes and output
  hashes. Integrity is not authenticity, scientific validity or atomic execution.
  Do not change sources or inputs during a run; no filesystem lock is taken.

With the same `SEMICONDUCTOR_TOOLBOX_SOURCE` variable set, run both the focused
cross-repository tests and the full Clotho suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -p 'test_semiconductor_walkthrough.py' -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

Without that variable, cross-repository integration tests explicitly skip rather
than downloading another repository. Local wrapper contract tests still run.
These are software regression checks, not independent scientific observations.

### Native400 remains a different workflow

The Toolbox's `examples/rram-native400-compare/` requires chronological,
anchor-excluded 400-write conductance predictions. This walkthrough's opaque
repeated HRS/LRS cycle labels do **not** encode those paths. Do not silently
reinterpret them, invert the declared target quantity, or feed this device-level
feature table into the native400 comparator. Use that companion's separately
specified view guard and prediction CSV when the actual experiment supports it.
