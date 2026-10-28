# Changelog

## 0.3.0b4 — Blank-input setup

- Package the existing reviewed `clotho init --workflow rram|fet --out ...` route: header-only CSVs, intentionally unfinished configuration and an English/Traditional Chinese local checklist. No observations, units, endpoints or scientific declarations are invented or accepted.
- Preserve the existing prepare → plan → run → score routes, algorithms, schemas, dependencies and synthetic fixtures. Untouched scaffolds fail prepare without a prepared output.
- Keep the trusted, idle local-directory boundary explicit: observed replacement fails, but hostile concurrent ancestor replacement can still create outside files before detection. This is not a transaction or full TOCTOU defense.
- Advance package/help/current-install guidance to b4; frozen b3 remains unchanged and lacks init. All b3/b2 and older tags/assets stay frozen.
- See the [b4 release guide](docs/RELEASE_NOTES_0.3.0b4.md) for precise verification attribution, installation and platform/privacy limits.

## 0.3.0b3 — FET device regression

- Package the accepted FET v2 adapter: signed-current criterion VGS and descriptive windowed log-current slope, one declared acquisition per composite physical device, and exact external endpoint joins.
- Reuse the existing grouped ML core; preserve RRAM v1, model families, preprocessing, splits, metrics and dependency bounds.
- Include the original invented FET example in the matching source archive, with runtime adapters, pinned MIT extractors and licenses in the wheel.
- Keep b2, b1, a4 and earlier tags/assets unchanged. See the [b3 release guide](docs/RELEASE_NOTES_0.3.0b3.md) for installation, data boundaries and verification limits.

## 0.3.0b2 — portable measurement bundle identities

- Use slash-separated receipt inventory keys on every host, fixing rejection of nested prepared/run bundles caused by Windows path separators. The regression simulates Windows path serialization; it is not native Windows validation.
- Include the wheel-first installation guidance, feature-selection examples, binary protocol guidance and refreshed product roadmap added after frozen b1.
- Preserve model families, learned preprocessing, group splits, metrics, receipt schemas, input safeguards and dependency requirements. Frozen b1 and earlier artifacts/tags remain unchanged.
- See the [b2 release guide](docs/RELEASE_NOTES_0.3.0b2.md) for verification boundaries and local installation. Source versioning alone does not establish publication.

## 0.3.0b1 — Clotho beta source

- Add the complete explicit measurement → target join → grouped comparison → target-free prediction → saved scoring route.
- Add `clotho` while retaining the `ml-super-calculator` distribution, previous CLIs and Python API.
- Make Axiom Clotho the product landing identity with original Celestial headers; keep existing scientific/workflow figures.
- Retain frozen a4 and earlier releases. This source entry is not proof of binary publication; see the [beta release guide](docs/RELEASE_NOTES_0.3.0b1.md).

## 0.2.0a4 — legacy prerelease

The [published a4 receipt](https://github.com/xinhi6716/ml-super-calculator/releases/tag/v0.2.0a4) records bilingual calculator reports, all-groups inspection, supplied-group scope, observed package-source metadata and the saved-interval CLI. Its exact source and assets remain frozen.

## 0.2.0a3 — prerelease

This prerelease packages the accepted source fixes after the immutable 0.2.0a2
snapshot (`20d17c1`). It does not change estimator families, preprocessing,
split schedules, scoring formulas or dependency requirements. A package version alone
does not establish public availability; consult the repository Releases page.

- Reject known impossible regression `--out` destinations and parent paths
  before fitting, and apply the same early rejection to binary output paths.
  Regular files and broken links cannot silently pass these preflights. The
  checks do not reserve paths or replace existing write-time race/permission guards.
- Preserve the validated CSV header identity across parsers. Reject inputs whose
  names are silently transformed (including duplicate-BOM header collisions)
  rather than selecting a target, group or feature against a changed schema.
- Reject bare-string or bytes Python API feature lists. Use an explicit sequence,
  for example `features=["voltage"]`, instead of accidentally selecting characters.
- Include the source-only A/nA synthetic unit-equivalence companion, exact saved
  predictions and figure, the examples index, and the offline version-guidance
  maintainer check. Synthetic outputs are educational software evidence, not a
  new held-out experiment, device validation or performance guarantee.
- Existing 0.2.0a2 tags, wheels, source archives and checksums remain unchanged.

The older `Unreleased` headings below are retained historical development notes;
they are not an additional feature list newly introduced by this prerelease.


## 0.2.0a2

- Reject explicit empty or blank-entry `--features` allowlists before planning or fitting. Previously `--features ""` silently fell back to all eligible columns. Omit the option deliberately to retain that default. Regression and binary commands share this validation; valid feature selections, models, splits and metrics are unchanged.
- The existing 0.2.0a1 release artifacts are unchanged and remain affected.

## Unreleased optional prediction bundle consistency

- Add an explicit, read-only regression prediction metadata/CSV consistency API
  and `validate-prediction-bundle` command with a versioned subset contract.
- Preserve existing writers, schemas, numerical algorithms and byte-only
  `verify`; reject unsupported modes/versions rather than guessing their meaning.
- Bound parsing and reuse pinned-directory no-follow regular-file checks; state
  unchecked metadata, unavailable old diagnostics, and scientific limitations.


## Unreleased

- Stage output bundles as exact UTF-8 bytes so platform-default newline translation cannot invalidate their SHA-256 manifests. Preserve caller-provided LF/CRLF text, per-file atomic no-clobber and overwrite behavior; no artifact schema or model changes. Regression tests emulate CRLF text defaults on Linux; this is not native Windows validation.

- Bound every shared regression/binary CLI CSV input to 64 MiB of original bytes and require a regular file before and after opening. Preserve accepted local regular-file aliases and exact single-stream provenance; use nonblocking open where available and a streaming cap to reject growth. This explicit input-compatibility tightening can reject previously accepted large files within the row limit. Preserve model, split, ordinary input hashes and artifact schemas; no DataFrame byte cap, hard runtime/memory guarantee or new dependencies. Retain prior provenance/NUL tests and add boundary, descriptor cleanup and no-work CLI regressions.

- Add a separate no-fit saved-interval evaluation API/module CLI: complete declared-cohort checks, closed-boundary empirical coverage, interval width and optional fixed-alpha standard interval score, with row-weighted and equal-group summaries. Preserve exact integer/decimal inputs and stored Python-float ratios through arithmetic, reuse exact group validation, and retain exact source ratios alongside explicitly approximate numeric reports. Reject missing/failed rows, invalid bounds and unreportable numeric values rather than dropping them. Include an invented voltage example and hand-checkable tests; no training, existing artifact schema, calibration, future-coverage guarantee or dependency changes.

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
