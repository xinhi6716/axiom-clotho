[Axiom Clotho](../README.md) / Inspect the planned fit count

# Understand a fit budget before changing it

The measurement workflow evaluates grouped model selection, then selects and
refits on all training data for future prediction. Its configured `max_fits`
limits the **sum** of those two planned workloads.

The default `clotho plan` rejects a workload above that declared cap. To inspect
the counts without first raising the cap, explicitly request a budget summary:

```sh
clotho plan --prepared my-prepared-bundle --budget-summary
```

This opt-in command prints one JSON object to stdout. It performs no model or
preprocessing fits, does not create a report or output directory, and does not
change configuration, prepared inputs, or `max_fits`. It supports both the
existing RRAM and FET prepared-bundle contracts. It still needs the existing ML
dependencies to validate the actual grouped schedules. Help remains stdlib-only.
This source feature does not alter previously built wheels.

## Read the JSON and its exit status together

| Field | Meaning |
| --- | --- |
| `schema_version` | `measurement-fit-budget-summary-v1` |
| `unit` | `pipeline.fit calls` |
| `fits_performed` | Always `0` for this inspection. |
| `comparison_fits` | Existing planner's total for nested grouped comparison, including inner selection and outer family refits. |
| `prediction_fits` | Existing planner's total for full-training selection and family refits. |
| `total_planned_fits` | `comparison_fits + prediction_fits`. |
| `max_fits` | The unchanged cap declared in the prepared configuration. |
| `within_budget` | `true` when total fits are less than or equal to the cap; otherwise `false`. |
| `limitations` | Scope and resource-accounting cautions. |

- **Exit 0:** the validated schedule is at or below the cap; JSON contains
  `within_budget: true`. This is not proof that later fitting will succeed.
- **Exit 2 with complete summary JSON:** the validated schedule exceeds the cap;
  `within_budget: false`. The count inspection succeeded, but the budget check
  did not pass. No error text is added to stderr for this case.
- **Exit 2 with no summary JSON:** invalid arguments, a malformed/inconsistent
  bundle, infeasible grouping, or unavailable dependencies/resources prevented
  a valid summary. Read stderr. Do not interpret missing JSON as zero cost.

The over-budget case deliberately remains a nonzero exit. Do not discard that
status or treat the presence of JSON as permission to run. A program consuming
this opt-in mode should inspect both the exit status and the summary fields.

For the public invented RRAM example with its dummy/ridge schedule, the current
planner reports 30 comparison fits plus 10 prediction fits: 40 in total. A cap
of 1 remains 1 and gives exit 2; a cap of 40 gives exit 0. These numbers describe
that software fixture, not a recommendation or a fixed formula. The command
uses the actual planner results, so supported family/candidate and split changes
are reflected without a separately maintained family list.

## Counted work is not compute time

A counted call fits one pipeline, including its preprocessing and one estimator.
Internal estimator operations, individual trees, and individual preprocessing
steps are **not** counted separately. The selected entry reuses the winning
family's refit rather than adding another fit. Fit counts are not CPU time,
elapsed time, memory estimates, a hardware-capacity check, or a hard resource
reservation. Equal counts can represent very different computational work.

The compact summary contains aggregate counts and static explanations only,
not device IDs, target values, feature names, private paths, or the full plan.
It is not a provenance receipt or evidence that any model was executed.

## Existing safety and scientific boundaries remain

- Without `--budget-summary`, `clotho plan` retains its original full-plan JSON
  and over-budget failure behavior. `clotho run` remains strict and does not
  accept this flag. The Python `mlworkflow.regression.budget_summary(path)`
  exposes the same opt-in count report; `plan(path)` and `run(...)` retain their
  budget guards.
- A summary does not relax input checks, invent a feasible split, raise a cap,
  select a cheaper model automatically, or edit a saved bundle. Changing
  scientific grouping or fold counts just to fit a budget can change the
  question being evaluated. Make those choices explicitly and justify them.
- If you change a configuration, regenerate the prepared bundle from its inputs
  through the normal workflow. Do not edit a checksummed prepared bundle or its
  receipt to make a check pass.
- A later invocation validates again. Changes to inputs, options, or software
  can change the plan; this report does not reserve or freeze execution.
- Passing the count check does not establish successful training, absence of
  leakage, physical independence, predictive usefulness, or scientific validity.

See the [RRAM](CLOTHO_CSV_BETA.md) and [FET](FET_DEVICE_REGRESSION.md)
contracts for valid measurements, grouping and target separation.
