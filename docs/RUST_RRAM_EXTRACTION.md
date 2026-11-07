# Native RRAM measurement features

`extract-rram` is a bounded, offline SI/binary64 workflow with schema
`clotho.rram_si_binary64.v1`. It replaces the arithmetic path from declared I–V
branches to paired device features; it is **not** a reader or byte/numerical-parity
replacement for the legacy three-file exact-decimal measurement bundles.

```sh
clotho extract-rram --csv measurements.csv --max-bracket-v 0.11 \
  --accept-assumptions true --out new_rram_features
```

The single UTF-8 CSV has exactly these unique headers, in any order:

```csv
batch,wafer,device,cycle,replicate,branch,state,condition,protocol,read_voltage_V,temperature_K,read_delay_s,voltage_V,current_A
B,W,D,001,R1,H,HRS,C,declared_dc_read,0.1,300,0.01,0.05,0.0000001
B,W,D,001,R1,H,HRS,C,declared_dc_read,0.1,300,0.01,0.15,0.0000003
B,W,D,001,R1,L,LRS,C,declared_dc_read,0.1,300,0.01,0.1,0.00001
```

This is an invented arithmetic example, not real data or switching simulation.
Current is amperes, voltage is volts, temperature is kelvin and delay is seconds
(note the deliberate difference from legacy `read_delay_ms`). No unit inference
or conversion occurs. Numeric text is at most 64 bytes; parsed inputs are finite,
zero or absolute magnitude in [1e-150, 1e150]. Nonzero literals that round to zero
are rejected. Ordinary decimal-to-binary64 rounding is accepted, including two
source decimals rounding to the same value. Equality of numeric declarations is
binary64 equality, not source-decimal equality. Derived results must remain finite
and positive where resistance/ratio is required. This is not arbitrary precision.

## Admission and physical interpretation

- IDs are unpadded, nonempty, at most 128 characters, control-free and not prefixed
  with `=`, `+`, `-` or `@`. Device IDs are globally unique across batches/wafers.
- Each `(device, cycle, branch)` identifies one acquisition-ordered branch with
  fixed metadata. Interleaving branches is allowed; samples are never sorted,
  deduplicated or merged to repair an invalid sweep. Voltage must be strictly
  increasing or strictly decreasing. A single measured point is allowed.
- State must be explicitly `HRS` or `LRS`. Each `(device, cycle)` has exactly one
  branch of each state. Both share a replicate and condition; a replicate cannot
  be reused for another cycle of that device. State is never inferred or swapped.
- Each condition label fixes signed nonzero read voltage, positive temperature,
  nonnegative delay and protocol across the entire cohort. Different labels are
  not equated even if their declarations match.
- A readpoint is either an exact binary64 voltage match or linear interpolation
  between adjacent samples. No extrapolation. Interpolated bracket span must not
  exceed the explicitly supplied positive bound. Descending branches retain their
  original source-row order in lineage.
- Static terminal resistance is signed `V_read / I_read`; zero or opposite-sign
  read current fails. No absolute-value correction, differential resistance,
  contact correction, SET/RESET detection or microscopic mechanism inference.
- The required acceptance flag declares correct terminal conventions, SI exports,
  state-preserving reads, protocol/condition labels, settling, compliance, noise
  floor, read-disturb assessment and local linearity within the bracket. Software
  cannot independently check these assumptions or certify physical accuracy.

One fixed condition per device is the default. Add
`--feature-row-unit device-condition` to retain separate rows for multiple declared
conditions on a device. Conditions and devices are never pooled. Output order is
lexical device/condition order, with cycles lexical within each feature row.

Three descriptive features are emitted: median HRS resistance, median LRS
resistance and median **within-cycle HRS/LRS ratio**. The last is not a ratio of
medians. Ratios below one remain unchanged and are counted. Even-count medians
use an overflow-safe midpoint. Cycles and condition rows are repeated observations,
not independent devices. No uncertainty estimate or population claim is supplied.

Limits: one regular file up to 8 MiB, 10,000 measurement records, 1,000 branches,
and 1,000 samples per branch. All invalid/incomplete branches reject the whole
cohort before any output directory is created. No silent dropping or imputation.

## Output and downstream use

A new output directory contains:

- `features.csv`: identity/condition/protocol, explicit SI declarations, cycle
  counts, the three numeric features, `physical_device_group=device` and
  `split_group=batch`.
- `measurement_input.csv`: the exact bounded input snapshot used for extraction.
- `extraction.json`: schema/row-unit/settings, feature allowlist, zero ML fits,
  limitations, per-pair signed currents and resistances, selected/all source rows
  and interpolation span. Source row means logical data-record index plus header,
  not physical line number when fields contain newlines.
- `COMPLETE`: written last; absent on an interrupted/incomplete write.

No existing path is overwritten, including symlinks. Unix directories/files request
0700/0600 permissions. All validation precedes output creation; an I/O failure can
leave a partial directory. Diagnostics omit input IDs/values; requested files do
contain private measurements. No downloads, telemetry or Python runtime occur.
Snapshots are not cryptographic provenance authentication or legacy hash manifests.

The feature table can feed `prepare-measurements`, with identity
`batch,wafer,device` (default row unit) or `batch,wafer,device,condition`
(condition row unit), explicit feature selection and externally defined endpoints.
For condition rows, choose a group prefix of at most 3 to hold out whole devices
or broader groups. Never treat separate conditions/cycles as independent devices.
The grouping columns are declarations, not evidence of independence. Do not predict
the ratio from the same HRS/LRS resistances. Feature availability before endpoints,
target proxies, physical identity aliases and causal validity need separate review.

## Verification and migration scope

Native tests cover signed/measured/interpolated reads, descending order, nonmonotonic
and duplicate voltages, no extrapolation, bracket bounds, medians of paired ratios,
below-one retention, incomplete/duplicate states, identity/condition/replicate
checks, source-row uniqueness, condition opt-in, finite range and literal underflow,
assumption acceptance, privacy, snapshots, lineage and no-overwrite behavior.

Reference semantics were reviewed in legacy `build_features.py`, `iv_readpoint.py`,
`rram_window.py` and the measurement-to-features README under
`src/mlworkflow/_vendor/semiconductor/` at commit
`7ae3c0fdc8957a5cd81c24690997aa14948fd464`. The native contract deliberately replaces
three-file manifest joins with repeated explicit declarations in one CSV, uses
seconds for delay and logical row indices, and does not claim legacy bundle
validation, exact-decimal parity, cryptographic manifests, exclusion-report mode,
unit conversion, statistical independence or scientific validation.

The integrated workspace passes 261 tests on Rust 1.70.0 and 1.99.1, including
11 RRAM core and 6 RRAM CLI tests. A separate reviewer inspected the implementation
and added permutation and extreme finite-ratio tests. Clippy passes with warnings
denied. These are local cloud software checks, not real-material validation or
confirmation of hosted CI success.
