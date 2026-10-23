# Long-term roadmap

ML Super Calculator is a product direction. GroupBench v0.1 is the first usable local engine, not the finished long-term project. Future work is gated on tests, justified evaluation, and clear data/model licensing, not on claiming a bigger model.

## Shipped in 0.1

- Local CSV and named-DataFrame interfaces.
- Nested group-disjoint regression selection and evaluation.
- Dummy, ridge, and Extra Trees baseline families.
- Local full-training refit and new-row prediction.
- Offline report, per-group errors, exact predictions, provenance, and checksums.
- Synthetic materials-process fixture and adversarial integrity tests.
- Aggregate train-derived new-row numeric-range, source-category and missingness diagnostics, available without fitting through `plan predict`.

## Unreleased: explicit local binary classification

- A separately selected numeric-0/1 workflow with fixed group Brier, prior/Logistic/bounded-tree baselines, nested group-disjoint evaluation, new-row prediction, and fresh-only bundles.
- No-fit exact budgets, class-feasibility safeguards, raw probabilities, adverse-group/class-support diagnostics, original software fixtures, and a dedicated schema/protocol.
- This is not multiclass, calibrated risk, a real semiconductor benchmark, or automatic process control. See docs/BINARY_PROTOCOL.md.

## Next: reliability and independent real-data validation

Acceptance criteria:

- A small, licensed, public materials/process dataset with a defensible grouping boundary and a documented data card. No private research snapshots or undisclosed benchmark data.
- Multi-seed runs with uncertainty methodology that respects group dependence, justified before inspecting results.
- CI matrix for declared Python/dependency bounds; packaged-install and cross-platform tests.
- Hardened input resource limits, schema/type overrides, and report schema compatibility tests.
- Measured runtime/memory budgets, not only limits on row/feature counts.

## Next: a useful local calculator experience

- Local/self-hosted interface for selecting a CSV, checking its schema, setting a target/group, and reading results. The current release is CLI/Python, not a browser-upload app.
- Saved experiment configurations and run comparison without silently using test results for selection.
- Extend the shipped source-level new-row diagnostics only with justified, independently evaluated applicability methods; current range/category flags are not uncertainty or a multivariate support test.
- A safe, versioned model-state export format only after a security and reproducibility design review; no arbitrary untrusted pickle loading.

## Later: expand scientific capability carefully

- Temporal and truly multi-axis grouped holdouts with explicit estimands.
- Additional local model adapters with transparent budgets and fair comparisons.
- Group-aware uncertainty calibration and independently evaluated coverage.
- Beyond the explicit binary milestone, multiclass and other task types only with appropriate metrics/protocols.
- Domain feature transforms with tests for train-only fitting and explicit units.
- Optional separately installed foundation-model adapters only after version-specific license review and explicit local resource design. No bundled third-party weights by default.

## What stays out

- A hosted compute/inference API, automatic uploads, or hidden remote fitting.
- Unlicensed weights, copied research artifacts, or claims of state-of-the-art performance from synthetic demonstrations.
- Making “more models” a substitute for trustworthy evaluation.

## Maintenance policy

Keep bug reports reproducible with synthetic/minimized examples. Prioritize correctness, data boundaries, safety, and first-run usability over feature count. Every change to preprocessing, splitting, or selection requires leakage regression tests. Changes to artifact schemas require a version and migration note. Milestones are scoped directions, not a guarantee of delivery dates.
