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
| Can changing units change a model's predictions? | [A versus nA](numeric-unit-equivalence/README.md): twelve fixed fits comparing raw models and train-only scaling. Requires scikit-learn 1.8.0 and the documented source hash; saved results can be read without fitting. |
| How does adjusting for an intermediary change a causal question? | [Causal-design rehearsal](causal-design-rehearsal/README.md): OLS and paired interventions in an invented model. Requires the simulation-only flag; it is not a real-data causal estimator. |

## Analyze saved results without fitting

- [Compare validation budgets](../docs/SAVED_BUDGET_COMPARISON.md): two methods × two budgets on matched observations. [Invented input](saved-budget-comparison/invented.json).
- [Evaluate prediction intervals](../docs/SAVED_INTERVAL_EVALUATION.md): coverage, width and interval score for a complete declared cohort. It does not generate or calibrate intervals. [Invented input](saved-interval-evaluation/invented_voltage.json).

Both utilities print descriptive JSON. If you redirect it to a file, choose a new path: the shell may overwrite an existing file before Python starts.

## Before using real data

Read the selected input contract. Some examples accept only their fixed invented data. Decide the units, physical grouping, feature timing and evaluation split before looking at outcomes. Software checks cannot establish those scientific choices. [Evaluation rules and limits](../docs/USAGE_REFERENCE.md#limits-to-understand-before-trusting-a-result).
