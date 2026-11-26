# Examples

[Clotho home](../README.md) · [繁體中文總覽](../README.zh-TW.md) · [Install b7](../README.md#begin)

Start with a small report, follow a device-measurement workflow, or explore one modeling question. All supplied data are **invented teaching fixtures**, not measured semiconductor benchmarks.

Run commands from the repository root. Check each example's requirements: some are source-only companions and are not installed with the CLI. Use a fresh output path.

## Make a first report

```bash
clotho tabular demo --out runs/demo
```

Open `runs/demo/report.html`. This command fits models on 288 invented rows in 24 batches. To check your own inputs and proposed workload without training, use [plan first](../README.md#plan-first-then-fit).

Prefer Python? The [regression API](python_api.py) and [binary classification API](binary_python_api.py) examples evaluate and fit models. Binary classification must be selected explicitly.

## Work with device measurements

| Example | What you will do | Requirements |
| --- | --- | --- |
| [RRAM device regression](measurement-regression/README.md) | Prepare resistance features, inspect the fit plan, predict devices and score later outcomes. | Clotho b7 and its matching source fixtures. |
| [FET device regression](fet-device-regression/README.md) | Extract criterion VGS and windowed log slope for one acquisition per composite device key. | Clotho b7 and its matching source fixtures. |
| [Semiconductor walkthrough](semiconductor-heldout-walkthrough/README.md) | Connect synthetic I–V generation, feature extraction, a lot-level holdout check and five fixed model fits. | This source checkout and a trusted local Semiconductor Toolbox checkout. |

For your own measurements, [clotho init](../docs/WORKFLOW_INIT.md) creates blank inputs and a checklist. It does not copy invented fixture rows into your study.

## Explore a modeling question

| Question | Example |
| --- | --- |
| How do two fixed feature sets behave in a different environment? | [Environment-heldout comparison](environment-heldout-comparison/README.md): five fixed fits, target-free prediction, then separate scoring. |
| Does holding out devices test a new recipe? | [Held devices versus held protocols](device-versus-protocol/README.md): paired toy scores on the same 20 invented devices, with two separate holdout questions. |
| Can changing units change a model's predictions? | [A versus nA](numeric-unit-equivalence/README.md): twelve fixed fits comparing raw models and train-only scaling. Requires scikit-learn 1.8.0 and the documented source hash; saved results can be read without fitting. |
| How does adjusting for an intermediary change a causal question? | [Causal-design rehearsal](causal-design-rehearsal/README.md): OLS and paired interventions in an invented model. Requires the simulation-only flag; it is not a real-data causal estimator. |

## Choose the measurement before choosing the model

| If you need to check… | Run this small example | What to record in your own study |
| --- | --- | --- |
| Whether two sources define track height the same way | [Order of profile aggregation](profile-aggregation-order/README.md) | Transverse registration, baseline, along-track span, and whether averaging precedes peak extraction. Equal areas do not resolve a ranking reversal. |
| Whether derived printability quantities can replace measured height | [Derived bead geometry](derived-bead-geometry-alias/README.md) | Keep measured bead height, melt depth, hatch spacing and design layer thickness as distinct endpoints. |
| Whether a penetration reading establishes adhesion | [Penetration and adhesion](penetration-is-not-adhesion/README.md) | Count all attempted tracks. Keep zero depth, missing depth and measured adhesion status separate. |
| Whether two resistance reads satisfy the same prediction schedule | [Scheduled reads versus first-after reads](scheduled-anchor-vs-first-after/README.md) | Record the declared slot and actual elapsed time; a read after the cutoff cannot silently replace a missing scheduled read. |
| Whether local timer values establish a retention interval | [Retention clock identity](retention-clock-identity/README.md) | Require a common clock and epoch, retained origin, and bounded per-read timing before applying the elapsed window. |

Run these standard-library examples from the repository root. They inspect
invented observations and do not train models. To check a real forecast's
feature timing, use the separate [as-of lineage declaration](../docs/GROUP_TIME_LINEAGE.md);
the examples do not attest to instrument records or physical device identity.

## Optional models and later outcomes

Follow the [tiny grouped-model walkthrough](optional-model-delayed-scoring/README.md)
([繁體中文](optional-model-delayed-scoring/README.zh-TW.md)): preview budgets,
evaluate two opt-in families, predict three new rows, then score explicit later
outcomes without refitting. It uses 18 invented training rows and arbitrary units,
not real measurements. Current-source commands and matching fixtures are required.

## Analyze saved results without fitting

- [Compare validation budgets](../docs/SAVED_BUDGET_COMPARISON.md): two methods × two budgets on matched observations. [Invented input](saved-budget-comparison/invented.json).
- [Evaluate prediction intervals](../docs/SAVED_INTERVAL_EVALUATION.md): coverage, width and interval score for a complete declared cohort. It does not generate or calibrate intervals. [Invented input](saved-interval-evaluation/invented_voltage.json).
- [Count support for a new machine](new-machine-support/README.md): run a synthetic cutoff check where 675 scores still provide only three machine groups.
- [Compare pointwise and simultaneous coverage](fixed-grid-coverage/README.md): on a four-read toy grid, reporting a time after seeing outcomes can change coverage.
- [See why feature support does not establish coverage](support-is-not-coverage/README.md): identical inputs can yield different outcomes and coverage.
- [Score a fixed device–phase schedule](scheduled-phase-scoring/README.md): see why weighting the same paired forecasts differently can reverse the ranking.
- [Check paired MAE sensitivity to one parent](paired-parent-influence/README.md): see how repeats differ from declared parent support and why omitting one parent can reverse the contrast.

- [Test MAE under shared or separate calibration offsets](calibration-groups-mae/README.md): two reference-path dependence choices on the same frozen forecasts.

The budget and interval utilities print descriptive JSON. If you redirect it to a file, choose a new path: the shell may overwrite an existing file before Python starts.

## Before using real data

For ALD process records, the [declaration preview](ald-contract-preview/README.md)
checks temperature roles, average-versus-slope target definitions, explicit
feature timing, and thickness/cycle arithmetic using invented in-memory cases.
It uses only the standard library, fits no models, and does not admit data to
training or verify scientific claims.

The [evidence-quarantine preview](evidence-quarantine/README.md) checks whether
explicitly quarantined source/group/endpoint keys remain in an observation
inventory, while leaving unrelated quantities separate. It uses metadata only
and preserves unknown endpoint mappings rather than guessing them.

Read the selected input contract. Some examples accept only their fixed invented data. Decide the units, physical grouping, feature timing and evaluation split before looking at outcomes. Software checks cannot establish those scientific choices. [Evaluation rules and limits](../docs/USAGE_REFERENCE.md#limits-to-understand-before-trusting-a-result).
