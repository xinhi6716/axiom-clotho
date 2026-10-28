# Pick an example

[← Calculator home](../README.md) · [繁體中文總覽](../README.zh-TW.md) · [Version and installation guide](../README.md#choose-the-version-you-mean)

Start with the question you want to answer, not the first folder name.
The data shipped here are **invented teaching fixtures**, not measured semiconductor benchmarks.

This is a **source-checkout guide**. Some companions are not installed by the
`mlcalc` package or included in a frozen download. Use a checkout containing the
example, read its requirements, and run commands from the repository root.
Do not copy commands from inside this directory without adjusting their paths.

## First visit? Try the small demo

Follow [the main quick start](../README.md#try-the-demo) to create a local HTML
report from synthetic data. **It fits models.** To inspect a planned workload
before training on your own data, use [plan first, then fit](../README.md#plan-first-then-fit).

Prefer Python calls? [Regression API](python_api.py) and
[binary pass/fail API](binary_python_api.py) both evaluate and fit models;
their reused example rows are not new observations. The binary workflow must
be selected explicitly. Start with fresh output paths.

## Follow a semiconductor workflow

- **FET transfer curves → composite-device endpoints:**
  [FET device regression](fet-device-regression/README.md), included since b3 and retained in b5. Blank caller inputs are available via
  [clotho init](../docs/WORKFLOW_INIT.md); init is separate from invented examples.
  One declared acquisition per physical device; signed criterion VGS and
  descriptive windowed log slope feed the existing grouped ML core. Runtime
  code is in the wheel; invented fixture files are in its matching source archive.

- **Measurements → features → held-out predictions:**
  [Semiconductor walkthrough](semiconductor-heldout-walkthrough/README.md).
  Fits five fixed pipelines after a lot-level holdout check. Requires a trusted
  local Semiconductor Toolbox checkout as well as Calculator dependencies.
  The repeated numeric device templates are invented; the command is not a
  real-data adapter or evidence of cross-fabrication performance.
- **Compare two fixed feature sets in a new environment:**
  [Environment-heldout comparison](environment-heldout-comparison/README.md).
  The prediction command fits five pipelines; a separate scoring command reads
  held-out outcomes without refitting. Requires one row per device and declared
  dependency blocks that do not cross the split. Feature sets are not thereby causal.

## Already have predictions or scores? No fitting needed

- [Compare saved validation budgets](../docs/SAVED_BUDGET_COMPARISON.md):
  two methods × two budgets with matched observations and explicit unit/metric
  declarations. Start with the [invented input](saved-budget-comparison/invented.json).
- [Evaluate saved prediction intervals](../docs/SAVED_INTERVAL_EVALUATION.md):
  score existing observations and bounds on one complete declared cohort.
  Start with the [invented voltage input](saved-interval-evaluation/invented_voltage.json).
  This does not generate or calibrate intervals.

Both utilities print descriptive JSON to stdout. They do not train, download
models, or certify scientific validity. If redirecting output, choose a new
path: the shell can overwrite a file before Python runs.

## Explore a specific modeling assumption

- **Does changing units change predictions?**
  [A versus nA](numeric-unit-equivalence/README.md) runs twelve fixed fits and
  separates raw-model ablations from the existing train-only scaling pipeline.
  Requires scikit-learn 1.8.0 and the documented model-source hash. You can also
  read its saved table and redraw the figure without fitting.
- **What does an adjustment choice mean in an invented causal model?**
  [Causal-design rehearsal](causal-design-rehearsal/README.md) computes OLS fits
  and paired interventions under declared synthetic equations. It requires an
  explicit simulation-only flag. It is not a real-data causal estimator or a
  test that real causal assumptions hold.

## Before replacing a fixture with real data

Read the selected tool's exact input and output contract. Some examples accept
only their fixed invented data. Define units, physical grouping and which
features exist before the outcome; freeze the split and comparison before
looking at evaluation targets. A successful software run does not validate
those choices. [Evaluation rules and limits](../docs/USAGE_REFERENCE.md#limits-to-understand-before-trusting-a-result)
