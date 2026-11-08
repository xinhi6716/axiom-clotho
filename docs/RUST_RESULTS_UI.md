# Native offline results workspace

`clotho report` creates a self-contained HTML results interface from a completed
native `evaluate` directory. It is implemented in Rust, uses browser-native
navigation and disclosures, and needs no Python, React, JavaScript, server,
external font, CDN, or hosted account. It does not run a model or upload data.

```sh
cargo build --manifest-path rust/Cargo.toml --locked --offline
rust/target/debug/clotho evaluate --csv measurements.csv --target response \
  --features temperature,pressure --group wafer --model ridge --alpha 1 \
  --folds 3 --out runs/evaluation-01
rust/target/debug/clotho report --run runs/evaluation-01 \
  --out runs/evaluation-01.html
```

Open the HTML in a local browser. Input and output paths are chosen explicitly;
existing output files, including symlinks, are not overwritten. The CLI returns
nonzero and leaves no new output for rejected inputs before publication.
A write failure can leave a partial output file, which must not be treated as a
complete report. Output parent directories must already exist. On Unix the new
report has mode 0600 (subject to the process umask).

## Included interface

- Overview: held-out rows, groups, folds, model.
- Separate pooled and equal-group metric panels, preserving undefined values.
- Observed-versus-predicted SVG preview. At most the first 2,000 records are
  drawn, explicitly labelled as a file-order subset, not a representative sample.
- First 100 prediction rows, group details, fold details, full escaped JSON.
- Target, explicit feature list, preprocessing and selection/split protocol.
- Semantic headings, skip link, keyboard-native expandable sections, focus
  indicators, responsive layout, system light/dark mode, and print stylesheet.

Both `clotho.grouped_evaluation.v1` and
`clotho.grouped_binary_evaluation.v1` are supported. Classification uses positive
probability on the prediction axis; it does not pretend probabilities are
continuous observed outcomes. `compare`, arbitrary Python studies, fitted-model
files and other schema versions are rejected. Native comparison results need a
separate adapter with their selection and failure semantics preserved.

## What is checked and what is not

The viewer requires a matching `COMPLETE` marker and metric sections. It checks
prediction header identity, total row count, contiguous row indices, fold range,
finite target/prediction values, required metric fields, group/fold count consistency, and binary target/probability/label/threshold consistency. Each source
file is bounded to 16 MiB. Direct source-file symlinks are rejected.

This is an inspection UI, **not independent scientific validation**: it does not
recompute reported metrics, rerun fits, reconstruct group partitions, verify
hashes or prevent concurrent mutation of input files. Read completed immutable
runs from trusted directories. A marker is not a signature. A valid file shape
does not establish label quality, valid physical units, independence of groups,
external validity, causality, calibrated uncertainty or useful device prediction.
Do not overwrite a report with improved-looking numbers without a new experiment.

All dynamic strings are HTML-escaped. No scripts or remote assets are embedded;
a restrictive content-security policy blocks scripts, forms and external loads.
The report contains saved feature/target/group names, metrics and a prediction
preview. Keep it private when those contain sensitive or proprietary information.
The source data and saved models are never included automatically.

## Design provenance and migration status

The requested design reference was [Meta Astryx](https://github.com/meta/astryx),
a React 19/StyleX MIT-licensed component system. This slice uses original HTML/CSS
with a restrained token-based palette, rounded cards, clear typography and
progressive disclosure. It does **not** copy Astryx code or assets, import its
React components, claim Meta affiliation or introduce a JavaScript frontend.

This is the first native results-interface slice, not a complete desktop app or
interactive training control panel. The existing Python package remains for
capabilities not yet migrated; no legacy files or branches were deleted.

## Verification record (2026-11-08)

- Rust 1.99.1 and minimum-supported Rust 1.70.0: 278 complete workspace tests passed, including 17 new report unit/integration tests;
  `cargo fmt --all --check`; Clippy all targets with warnings denied.
- Integration coverage: regression and binary evaluate-to-report, incomplete
  runs, schema marker mismatch, row alignment, malicious HTML strings,
  unknown options, no-clobber and input/output symlinks.
- Synthetic regression smoke: 6 held-out rows / 3 groups / ridge alpha 1;
  MAE 0.8666667 and RMSE 1.0. This is a software workflow check, not real-material
  evidence or a predictive performance claim.
- Browser visual/interaction QA **not completed** in this environment: Chromium
  could not start because the environment disallows its process socket.
  Desktop/mobile layout, focus, disclosure interactions and offline network
  behavior still require real-browser acceptance before calling the UI fully
  verified. Static HTML generation and content assertions do not replace that.
