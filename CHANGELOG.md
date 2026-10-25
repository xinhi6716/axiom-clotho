# Changelog

## Unreleased

- Enforce the existing exact group-identity validation in direct regression and binary grouped scoring, preventing NumPy dtype promotion from silently merging distinct large numeric IDs and reversing equal-group model comparisons. Reject previously accepted ambiguous/invalid group vectors; preserve metric formulas and already validated high-level evaluation behavior. Caller-side lossy conversion cannot be repaired. Add no-fit hand-calculated identity, weighting, permutation, and malformed-input regressions.

- Preserve paired B-minus-A absolute-error contrasts in the saved environment-heldout scoring companion when subtracting rounded errors would falsely report a tie. Compute row contrasts and their equal-device mean exactly from the already converted binary64 values, rounding only for output and rejecting nonzero-to-zero underflow. Displayed contrasts may differ from subtracting rounded MAEs. Preserve prediction/fit behavior, MAE/RMSE, exact-cohort checks and artifact schemas; no new dependency or real-data evaluation is added.

- Require string `target` and `group` column selectors in regression/binary DataFrame workflows before planning or fitting. Non-string labels, including NumPy integer labels that previously failed JSON serialization only after fitting, must be explicitly renamed by the caller; no label coercion is performed. Preserve ordinary string names, numeric group values, selected-feature behavior, fingerprints, artifact schemas, and model protocols.

- Reject observed NUL-containing categorical feature values in the DataFrame API before planning, fitting, source-support diagnostics, or training/prediction fingerprints. This explicit input-validation tightening prevents false unseen-category flags and ambiguous pandas-based fingerprints, even when model encoders distinguish the values. Preserve clean observed values with unused NUL category metadata, ordinary input behavior, numeric/group contracts, both encoding protocols, artifact schemas, and existing CSV rejection. No automatic stripping, escaping, or new NUL-safe encoding is claimed.

- Reject NUL (U+0000) anywhere in DataFrame group identifiers before planning, splitting, fitting, or training-data fingerprinting. This explicit input-validation tightening prevents distinct string groups from silently merging through NumPy/pandas string operations; it does not preserve previously accepted NUL-group input. Ordinary Unicode/numeric group identity, non-group DataFrame feature handling, CSV NUL rejection, model protocols, and artifact schemas are unchanged.

- Add explicit opt-in `categorical_encoding="distinct-v2"` / `--categorical-encoding distinct-v2` across regression/binary estimators, execution, and no-fit plans. Escape every real marker-prefixed string before one-hot encoding to remove the literal-versus-missing key collision while retaining ordinary category ordering and cap ties. Preserve the `legacy-v1` default, existing artifact formats/run keys, and known legacy limitation; emit explicit v2 protocols/configuration and compatible schema minors only when selected. Unknown/infrequent compression, grouping, candidate grids, metrics, fit budgets, and CSV payloads remain unchanged. No model persistence or real-data performance claim is added. See docs/CATEGORICAL_ENCODING.md.

- Reject NUL characters anywhere in CSV input before pandas can silently truncate headers, group identifiers, feature values or targets. Use content-free physical-line diagnostics across regression/binary execution and no-fit planning; preserve valid-input parsing, single-read byte hashes, model protocols and artifact schemas.

- Add a separate, explicit numeric-0/1 binary workflow: grouped nested Brier selection, prior Dummy/L2 Logistic/bounded Extra Trees, raw-probability outputs, clipped log-loss diagnostics, class supports/confusion counts, adverse group comparisons, and fixed 0.5 labels. Add no-fit plans with exact fit budgets, train-class preflight guards, source-only preprocessing, finite/range/precision checks, nonconvergence failure and fresh-only local bundles. New binary schema identities start at 1.0; existing regression schemas/commands and fixed byte verifier remain unchanged. No dependency, real dataset, calibration, executable model loading, hosted API or autonomous physical decision is added. See docs/BINARY_PROTOCOL.md for migration boundaries and limitations.

- Add aggregate train-derived `input_support` diagnostics to new-row prediction and no-fit prediction plans: numeric range flags, unseen source categories, missing values and explicit unavailable support for all-missing training features. Preserve all models, transforms, selections, fit counts, predictions and CSV columns. Prediction manifests/plans advance to schema 1.1 with an additive field and migration documentation; evaluation schemas and fixed checksum bundle file sets stay unchanged. Counts are descriptive, not calibrated uncertainty, a multivariate support test or a reliability guarantee.
- Fail explicitly on float64 metric overflow, underflow, or invalid arithmetic instead of returning non-finite scores or silently zeroing tiny errors. Preserve ordinary metric formulas and weighting; keep R² undefined for exactly constant/one-row targets without computing an unsafe or roundoff-created target variance. Add synthetic edge cases and a no-training CLI regression proving selection aborts before report writes.
- Add stdout-only `mlcalc plan run` / `plan predict` and Python planning APIs: shared input/schema validation, actual seeded split sizes, candidate grids, and intended pipeline-fit counts without fitting or writing results. Execution consumes the same schedule; include dummy and per-family refits without double-counting the selected entry. No timing estimate or enforced resource cap is claimed. Prediction feature-type mismatches now fail before training using the estimator's existing type rules.
- Add optional, mutually exclusive `run`/`predict --features-file` for the version-1 feature schema emitted by Semiconductor Toolbox's synthetic RRAM example. Read only its explicit predictor allowlist through bounded, strict JSON validation; preserve existing CLI defaults, actual target/group protections, model protocol and selected-feature provenance. No target invention, executable loading, network access or scientific leakage-proof claim.
- Add offline `mlcalc verify RUN_DIR` for existing evaluation/prediction bundles: strict duplicate-aware checksum validation, bounded streaming hashes, POSIX no-follow reads, clear failure statuses, and integrity-only limitations. No fitting, model loading, network access, repair, or bundle-schema changes.
- Bind CLI training/new-row source hashes to the same incrementally read bytes used for CSV validation, including BOMs and line endings. Prevent later path replacement or edits from mismatching the recorded hash and consumed data; add deterministic mutation and bounded-read regression tests.
- Reject malformed CSV records before fitting, preventing pandas from silently shifting extra fields into an index or padding absent fields.
- Validate non-blank headers, UTF-8, and quoting with line/count diagnostics that omit raw cell values. Preserve quoted commas/newlines, blank-line behavior, explicit empty fields, and textual group identifiers.
- Apply and document Python's CSV parser field-size guard for data cells as well as headers; keep the existing bounded row limit and add synthetic input regression tests.

## 0.1.0 — 2026-10-23

Initial alpha of ML Super Calculator, powered by the GroupBench Python engine.

- Nested group-disjoint regression evaluation with train-only preprocessing.
- Dummy, ridge, and Extra Trees selection, per-group metrics, and split audit.
- Local CSV/Python full-data refit and new-row prediction.
- Offline HTML/JSON reports, exact prediction CSVs, and integrity checksums.
- Fully synthetic materials-process example, documentation, and regression tests.

No public real-data benchmark, foundation-model training, hosted service, calibrated uncertainty, or production-readiness claim accompanies this release.
