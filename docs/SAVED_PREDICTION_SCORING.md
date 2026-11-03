[Axiom Clotho](../README.md) / Delayed-outcome regression scores

# Score predictions after outcomes arrive

`mlcalc predict` fits a model and saves predictions. When the measured outcomes
arrive, the source CLI can now score that complete saved cohort **without fitting
again**:

```sh
mlcalc score-predictions --run-dir runs/new-devices --outcomes later-outcomes.csv
# Same command through the compatibility aliases:
groupbench score-predictions --run-dir runs/new-devices --outcomes later-outcomes.csv
clotho tabular score-predictions --run-dir runs/new-devices --outcomes later-outcomes.csv
```

This command prints one complete JSON result to stdout. It does not create or
modify a report, model, prediction bundle, or outcome file. It has no `--out` or
overwrite option. Shell redirection is performed by your shell and can overwrite
a file; it is not protected by this command. This source feature is not added
retroactively to an older frozen wheel.

## Preserve the original row mapping

The outcomes file must be UTF-8 CSV, optionally with one initial UTF-8 BOM, with
exactly this header and these three columns:

```csv
row_position,group,observed
2,batch_B,1.5
0,batch_A,2.0
1,batch_A,3.0
```

These are invented values explaining the format, not real measurements. The
`row_position` is the zero-based position in the **original new-feature table**
used to create the prediction bundle. Retain that table and its physical-ID map.
The position does not itself identify a specimen. Match outcomes explicitly to
that map; never assign positions simply by the order an instrument returns them.
CSV row order is immaterial: the scorer joins by position and reports canonical
prediction order.

Every saved position must occur exactly once. Missing, extra, duplicate, blank,
or failed observations reject the entire cohort; nothing is silently dropped or
imputed. Partial follow-up, censoring and missing-outcome analysis are outside
this first contract. Do not drop hard-to-measure cases or edit the prediction
bundle to make the roster complete.

- Positions use canonical nonnegative integer text: `0`, `1`, `2`, etc.
  Aliases such as `01`, `+1`, `1.0`, `1e0` and padded values are rejected.
- Groups are caller-supplied literal strings, maximum 256 characters. `01`, `1`,
  `NA` and Unicode identifiers remain distinct. Empty/all-whitespace groups and
  NUL are rejected; meaningful labels are not trimmed or normalized. These are
  **scoring groups**, not verified identities or proof of train/test separation.
- Observations must have the prediction target's meaning and units. Supply
  finite signed decimal/scientific text, at most 64 characters, without embedded
  units or whitespace. Booleans, complex text, hex, NaN, infinity and missing
  values are rejected. Target meaning, units and measurement timing are not
  inferred or independently checked.
- Integer-syntax tokens must be exactly representable in float64. For example,
  `9007199254740993` is rejected before type inference can round it away.
  Other decimal tokens intentionally use float64 rounding; decimal measurement
  precision is not audited. Overflow and nonzero-to-zero underflow are rejected.
  Existing metric arithmetic also fails closed when intermediate calculations
  overflow or underflow: rescale both observations and predictions explicitly.

## Interpret the result

The JSON uses `schema_version: tabular-saved-score-v1` and reports:

- `fits_performed: 0`, complete row and group counts;
- `overall`: existing row-weighted MAE, RMSE, R2 and signed bias (prediction minus
  observation), equal-group MAE/RMSE, and worst-group MAE;
- `per_group`: the same row metrics for each declared group;
- `scored_rows`: canonical `row_position`, literal `group`, `observed`, and saved
  `prediction` values;
- SHA-256 of the exact consumed checksums, manifest and prediction bytes, plus
  the exact outcome-file byte hash. Changing only outcome row order preserves
  scores and joined rows, but appropriately changes the outcome-file hash.

Constant-target R2 is `null`, including singleton groups. Equal-group RMSE is the
mean of per-group RMSEs, not the square root of a pooled mean-square error.
Unequal group sizes therefore give different row and equal-group scores.

This is descriptive scoring of one supplied cohort, not a new independent
holdout, uncertainty estimate, causal effect, or population-performance guarantee.
Repeated scoring does not restore test-set independence. Do not select a model
using these outcomes and then describe the same cohort as untouched validation.
Successful JSON intentionally contains group labels, outcomes and predictions;
review it before sharing.

## Read and failure boundaries

The existing regression prediction-bundle validator supplies the exact bytes
scored, including supported schema versions and its bounded metadata checks.
Scoring additionally applies its numeric conversion rules to saved prediction
tokens: nonrepresentable integer-syntax values and nonzero-to-zero underflow
reject the score even if the older structural validator accepts the bundle.
This tighter scoring subset prevents silently rounded false-perfect scores; it
does not alter the public bundle validator. Decimal-syntax rounding remains
explicit for both saved predictions and observations.
Predictions are **not reopened after validation**. Binary/evaluation bundles,
checksum mismatches, malformed rows and stale evaluation artifacts are rejected.
Unknown manifest metadata is not used to open training files or load models.
Byte hashes do not authenticate the source or prove historical model execution.

The bundle reader requires POSIX no-follow/directory-relative access and rejects
symbolic-link bundle components. Its limits remain 16 KiB checksums, 1 MiB
manifest, 4 MiB predictions CSV, 20,000 rows, and 100 features. Outcomes are read
once as a bounded regular local file, capped at 4 MiB and 20,000 rows. The existing
CSV opener allows local outcome-file symlinks; descriptor/path metadata changes
observed during the read reject the operation. Use trusted, idle local files.
These checks are not a filesystem lock or whole-directory atomic snapshot.

Exit 0 means complete scores were printed. Exit 2 means failure, with categorical
stderr guidance and no partial score JSON. Existing numerical dependencies are
still imported; **zero fitting does not mean dependency-free execution**. For
measurement-specific bundles, retain the separate `clotho score` workflow.


For two existing model bundles, the [same-query paired comparison API](SAVED_PREDICTION_COMPARISON.md)
adds mandatory original-query byte linkage and shared semantic declarations.
It does not compare arbitrary score JSON summaries or fit models.
