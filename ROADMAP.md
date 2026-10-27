# Axiom Clotho / Roadmap

**First depth, then breadth.** Clotho is the Moirai machine-learning workbench. A broadly useful ML builder is the ambition, not a claim of universal applicability. Semiconductor and materials problems are the first focus. The distribution remains `ml-super-calculator` for compatibility.

This page tracks product work, not private research or publication status. The [0.3.0b1 beta release](https://github.com/xinhi6716/axiom-clotho/releases/tag/v0.3.0b1) is pinned to `aec0c6a`; later documentation on main is not included in its frozen wheel/source archive.

## Available in the 0.3.0b1 beta

- [x] Fixed-condition, device-level measurement regression: explicit training endpoints, declared groups, prepare → plan → run → separate later-outcome scoring.
- [x] Local grouped tabular regression and explicitly selected numeric-0/1 classification. Dummy, linear and bounded-tree baselines remain visible; they are not a foundation model.
- [x] Nested group-disjoint selection, target-free new-row prediction, saved predictions, local reports and byte checksums.
- [x] One compatible distribution with `clotho`, `mlcalc` and `groupbench`; pinned semiconductor extraction resources stay bundled.
- [x] Clotho identity, English/Chinese README entry points and calculator report language controls. Measurement summary pages remain English.

These are tested software capabilities. Invented fixtures do not establish real-device accuracy, causal identification or cross-environment generalization. Not every toolbox format is integrated. See the [measurement contract](docs/CLOTHO_CSV_BETA.md) and [binary protocol](docs/BINARY_PROTOCOL.md).

## Finish the entrance before adding rooms

- [ ] Inspect the actual public README at desktop/mobile widths and in light/dark themes; check the full composition rather than individual icons alone.
- [ ] Verify local report keyboard navigation, long labels, narrow layouts and print output in a supported browser; static CSS checks are not rendering or accessibility certification.
- [ ] Keep route selection, installation and first results short; keep detailed scientific contracts reachable without opening one large reference block.
- [ ] Complete release asset readback and checksum verification; retain older release artifacts for rollback.
- [ ] Exercise declared Python/dependency bounds and supported operating systems. Current verification is not a cross-platform matrix.

## The next research-to-product loop

Each research session should close one bounded question, not add another framework. The recurring focus is semiconductor/materials ML and whether causal concepts improve prediction across environments.

1. **Choose a specific decision.** State the device/material endpoint, available-at-prediction inputs and an admissible grouping boundary. Review original prior work and the dataset's license before downloading or redistributing it.
2. **Freeze the comparison.** Record the falsifiable hypothesis, same-information strong baselines, split, seeds, metric and tuning rule before examining evaluation outcomes. Keep an exposed test set exposed; do not retune it into a new success.
3. **Run the smallest discriminating experiment.** Include negative or adverse cases. Preserve raw predictions, code/data versions, costs and a reproduction command. Compare datasets separately rather than pooling incomparable scores.
4. **Review evidence independently.** Separate literature findings, actual runs, output-integrity checks and independent reproduction. Date/domain labels alone are not exogenous interventions.
5. **Integrate only a justified increment.** Convert a demonstrated need into a small domain adapter, example or diagnostic, with regression tests and clear limits. Record a negative result or `no_change` when it does not justify a product change.

### First bounded candidate — not yet completed

- [ ] Select one licensed public semiconductor/materials table with documented batch/source groups and a scientifically defensible endpoint. Reject the candidate if its group meaning or feature timing cannot be established.
- [ ] Predeclare one group-heldout comparison using the current Dummy/linear/tree families and a single training-label-shuffle diagnostic. Do not call a batch split external validation without a separate relevant environment.
- [ ] Preserve all groupwise predictions and adverse results; inspect whether the baseline comparison changes across groups without post-hoc test-set tuning.
- [ ] Publish only a redistributable, independently reproduced example and data card. If licensing forbids copying data, publish an authorized retrieval recipe instead.

This first comparison is predictive, not causal. A later causal-inspired feature restriction or robustness method requires a separate prespecified comparison with the same information and an appropriate environment design. A DAG or extra terminology alone is not new evidence.

## High cohesion, low coupling

Keep one product, with responsibilities that can be tested separately:

| Boundary | Owns | Must not silently own |
| --- | --- | --- |
| Domain adapters | Explicit units, identifiers and feature extraction contracts | Labels, split policy or unrecorded physical assumptions |
| Evaluation engine | Train-only preprocessing, splits, selection, metrics and predictions | Domain meaning or report presentation |
| Evidence/report layer | Readable saved outputs, provenance fields and integrity checks | Retraining, test-driven selection or authenticity claims |
| CLI/workflow layer | Explicit configuration and handoffs between stages | Hidden remote computation or a second overlapping package |

Favor a narrow tested interface over a plugin framework built in advance. Preserve the existing input/output contracts and compatibility names; schema changes need a version and migration note.

## Later, when evidence warrants it

- Temporal or multi-axis holdouts with explicit estimands and leakage tests.
- Group-aware uncertainty calibration with independent coverage evaluation; current support flags are not uncertainty intervals.
- Additional model adapters and task types with fair information access and transparent compute budgets.
- A local interface and saved configurations that do not silently reuse test outcomes for selection.
- Safe model-state export only after security/reproducibility review; no arbitrary untrusted pickle loading.
- Optional foundation-model adapters only after version-specific license and local resource review; no bundled third-party weights by default.

## What stays out

Hosted compute APIs, automatic uploads, hidden fitting, unlicensed assets, private review/manuscript data and unsupported performance claims. More models are not a substitute for trustworthy evaluation. Milestones are directions, not delivery-date or accuracy guarantees.
