# Native FET measurement extraction

`extract-fet` is an experimental, Python-free **SI binary64** adapter. It extracts
a signed constant-current gate voltage and a descriptive log-current slope for
each complete acquisition. It does not train a model or establish a physical
threshold, subthreshold regime, independent devices, or predictive accuracy.

## Contract and usage

Build `cargo build --manifest-path rust/Cargo.toml --release --locked`.

```sh
rust/target/release/clotho extract-fet \
  --csv rust/clotho-cli/examples/synthetic_fet_si.csv \
  --criterion-a 1e-8 --polarity positive --max-bracket-v 1 \
  --window-min-v 0 --window-max-v 3 --accept-assumptions true \
  --out /tmp/clotho-fet-new
```

The synthetic example has one device, four observations and no target. Its
criterion is exactly 1 V and its log slope is 1 decade/V. It is not a training
dataset or a real semiconductor experiment.

CSV columns must be exactly these (order may differ):

`lot,wafer,device,condition,replicate,branch,temperature_K,drain_source_voltage_V,body_source_voltage_V,gate_source_voltage_V,drain_current_A`

All values and command parameters are already SI. There is no inferred unit
conversion. Numeric text is limited to 64 bytes. Inputs must be finite; nonzero
SI values must have magnitude from 1e-150 through 1e150. Text that underflows to
zero is rejected. Ordinary representable binary64 rounding still occurs. Exact
original decimal distinctions are **not** preserved.

- Explicit current polarity and a same-sign nonzero criterion are mandatory.
- Each acquisition is retained in CSV order, without sorting, smoothing,
  sign repair, outlier removal, branch selection or averaging.
- One acquisition per composite `(lot, wafer, device)` is required. Distinct
  condition/replicate/branch keys for the same device reject the entire cohort.
- Within a branch, temperature must be fixed and positive, VDS fixed and
  nonzero, VBS fixed, and VGS strictly monotonic. Current must have the declared
  sign and be monotonic (plateaus permitted except a nonunique criterion).
- The unique criterion is an exact measured value or linear interpolation in
  **signed current**, never log-current interpolation or extrapolation. An
  interpolated crossing must fit the declared maximum voltage bracket width.
- The log fit uses every measured sample inside the inclusive declared VGS
  window, equal weights and at least three samples. Selected currents must be
  strictly nonzero with the declared polarity. No best-window search occurs.
- Log slope is unweighted OLS of log10(current in A) against VGS in V. The
  reported log RMSE uses N. This descriptive fit is not an ML training fit.
- Distinct currents whose binary64 logarithms coincide are rejected. Numerical
  range failures reject the cohort; no partial feature export is produced.

Limits: 8 MiB CSV, 10,000 observations, 1,000 devices, 1,000 observations per
branch. Identifiers are nonempty, unpadded, control-free and at most 128 Unicode
characters. CSV formula prefixes are rejected for identifiers on CLI export.

## Output and privacy

A fresh directory contains `features.csv`, the exact retained
`transfer_input.csv`, `extraction.json` and a final `COMPLETE` marker.
`extraction.json` records acquisition identities, settings, selected and full
source record IDs, criterion bracket and descriptive log diagnostics. Source
row IDs mean **logical CSV record index plus header**, not physical line number
for multiline CSV records. Input order and acquisition order are preserved.

All validation and extraction precede directory creation. Files are created
exclusively; existing destinations and symlinks are not replaced. A filesystem
failure may leave an incomplete directory without `COMPLETE`. On Unix,
directories use 0700 and files 0600. Other-platform parent ACLs are not certified.
These artifacts contain private device identities and raw measurements. Keep
actual measurements private; public repository fixtures are synthetic only.

## Migration boundary

Schema `clotho.fet_si_binary64.v1` is a new numerical and artifact contract.
It is **not** a replacement for legacy FET v2 prepared bundles:

- Legacy extractors use exact decimal/rational arithmetic and 80-digit decimal
  logarithms. This implementation deliberately identifies itself as binary64.
- Legacy unit parsing, report inventories, hashes, prepared partitions,
  training-only endpoint joins, model selection, future predictions and
  zero-fit saved scoring are not part of this command yet.
- The retained input snapshot is not authenticated provenance. No checksum,
  source attestation, historical blinding or scientific validity is claimed.
- RRAM extraction is not implemented by this change.

The Rust API is in `clotho_core::fet`. Synthetic tests cover exact/interpolated
criteria, reverse and negative-polarity branches, malformed inputs, fixed
conditions, nonunique crossings, window admission, complete-cohort rejection,
lineage and CLI privacy. They are software evidence, not materials validation.

## Validation of this slice

The full workspace passes 220 tests on both Rust 1.70.0 and 1.99.1, including 20
core FET tests and 7 CLI FET tests. Clippy passes with warnings denied. An
independent review found nonzero text underflow into zero at the CLI boundary;
that admission bug was fixed for both CSV values and options, with regressions.
The release binary reproduces the documented synthetic criterion and slope.
These are local cloud checks; hosted CI status must be checked separately.
