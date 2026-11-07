# Native nested grouped regression comparison

`clotho compare` compares a bounded, explicitly supplied candidate list using
nested group-disjoint validation. It is a Rust-only **numeric regression** path,
not a claim of parity with the Python nested search or a complete migration.

```sh
cargo build --manifest-path rust/Cargo.toml --release --locked --offline
rust/target/release/clotho compare \
  --csv rust/clotho-cli/examples/synthetic_process.csv \
  --target synthetic_response \
  --features anneal_temperature_c,film_thickness_nm --group wafer \
  --candidates mean,ridge:0.1,ridge:1,knn:1 \
  --outer-folds 4 --inner-folds 2 --max-fits 36 --out nested-example
```

The included table is eight synthetic rows, not laboratory data. This command
exercises a protocol; its scores are not evidence of semiconductor efficacy.
Use appropriate real measurements, grouping, baselines and external evaluation
before making scientific claims.

## Selection and isolation

1. Hold out whole groups in each outer fold. The deterministic grouping algorithm
   is documented in the native workflow guide and is not class stratified.
2. Split only that outer fold's training groups into inner folds.
3. Fit each candidate independently in every inner training fold. Its numeric
   scaling uses only that fit's training data.
4. Reassemble inner out-of-fold predictions and compute **pooled inner RMSE**.
   Select the lowest score; exact ties choose the first listed candidate.
5. Refit that chosen candidate on all outer-training rows and predict the
   untouched outer-test rows.
6. Report one outer-held-out prediction for every original row. Outer targets
   and features never determine inner candidate selection.

The RMSE selection primitive computes RMSE only. An overflow in an unrelated
R² on a near-constant inner target cannot invalidate a finite RMSE score. Full
MAE/RMSE/R² are computed for final reporting under the metric range contract.

Candidate syntax is `mean`, `ridge:ALPHA` or `knn:K`, comma separated. Alpha must
be finite and nonnegative, and k must fit **every** inner training partition.
Invalid specifications fail before fitting. A numerically valid candidate that
fails a particular inner fit/prediction/score receives no partial score; its first
failure stage, fold and attempted-fit count are saved. Other candidates may
continue. All candidates failing, a selected outer refit/prediction failing, or
unrepresentable final metrics fail the study rather than silently omit rows.

## Budget and validation

The worst-case planned estimator-fit count is
`outer_folds * (candidate_count * inner_folds + 1)`. The final +1 is the selected
outer refit. The explicit `--max-fits` must cover the entire plan before any fit.
The implementation does not silently raise a cap. Early candidate failures may
reduce actual attempted fits; both planned and attempted counts are reported.

Limits: 20,000 rows, 100 numeric features, 16 candidates, 2–10 outer folds and
2–10 inner folds. Every outer training partition must contain enough groups for
the requested inner folds. Schema, finiteness, groups, candidate settings and
budget are preflighted. Limits are not a CPU-time guarantee; kNN exact search or
ill-conditioned repeated fits can still be expensive.

## Files and interpretation

A new private directory contains:

- `predictions.csv`: original row index, outer fold, true target and untouched
  outer-fold prediction.
- `selection.json`: candidates, feature/target/group identities, split/selection
  rules, fit budget/counts, pooled outer metrics, each fold's selected candidate,
  every inner candidate score/failure, and per-outer-fold metrics.
- `COMPLETE`: written last. Missing marker means incomplete output.

Output preflight, no overwrite and private Unix file modes follow the native
workflow contract. Target values and group identities remain private research
data. A validation or model failure produces no study directory; filesystems can
still fail during publication, hence the completion marker.

There is deliberately no single global winning model or deployment artifact in
this report: different outer folds can select different candidates. Selecting a
deployment model is a separate all-training-data procedure. Do not choose a final
candidate by maximizing this outer test report, then describe its score as an
untouched evaluation. Repeatedly redesigning the candidate list after inspecting
outer results also requires a fresh independent evaluation.

## Current boundaries

This first nested path accepts complete finite numeric features only. It rejects
`--preprocess mixed`, categorical roles, classification models and missing cells.
Do not pre-impute, globally scale, select features from full-data labels or use
query-informed transforms upstream and call the resulting study fold-safe. The
separate fixed-model `evaluate` command supports fold-fitted mixed preprocessing;
that support has not yet been combined with native nested selection.

No random forest/boosting registry, randomized Python split parity, automatic
feature selection, interval calibration, macro-group optimization, causality,
measurement ingestion, legacy artifact conversion or external scientific
validation is added by this command. See the migration inventory for remaining
work. A successful nested calculation does not guarantee unbiased estimates if
the supplied groups, targets or upstream data handling are inappropriate.

## Verification record (2026-11-07)

The final workspace passed **193 tests** on Rust 1.70.0 and 1.99.1: 34 CLI tests,
157 core integration tests and two core unit tests. There are 21 new selection
tests and four compare-command integration tests. Stable Clippy with warnings
denied, changed-file formatting and the release example (36/36 planned fits)
passed.

Independent review found that computing full inner metrics could reject a finite
RMSE when irrelevant R² overflowed. The dedicated robust RMSE primitive fixes
that issue and the exact fixture is retained as a regression test. Additional
independent structural checks covered 6,426 successful studies, 2,133 expected
preflight rejections, and 50,100 RMSE-oracle cases. These counts refer to numerical
and structural test cases, not scientific datasets or publications.

CLI checks additionally verified budget/candidate failures, no publication on
errors, private output permissions, original row/fold alignment, score/failure
retention and correct fit counts. This evidence does not certify Python parity,
real-material predictive accuracy, hosted CI, platform portability or completion
of every migration gate.
