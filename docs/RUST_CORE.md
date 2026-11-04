# Incremental Rust core

Current source contains an opt-in, dependency-free Rust library at
`rust/clotho-core`. Its first slice ports the source-feature support diagnostic
from `groupbench.input_support.prediction_input_support`.

This is a migration foundation, **not a Rust prediction engine**. The existing
Python commands, imports, model fitting, group/fold assignment, preprocessing,
run keys, output schemas and saved-bundle readers are unchanged. Python does
not load this crate or invoke a compiler. Frozen release downloads are unchanged.

## Why start here

For material and process tables, a new temperature outside the observed training
range, an unseen material category, or an entirely unobserved training feature
should remain visible. This diagnostic has a small target-free boundary that
can be ported and compared exactly without changing scientific predictions.

The Rust API accepts two arrays of named columns:

- `FeatureValues::Numeric(Vec<Option<f64>>)` for already-normalized binary64
  values. Use `None` for missing data; NaN and infinities are rejected.
- `FeatureValues::Categorical(Vec<Option<String>>)` for already-normalized
  strings. Literal missing-marker strings are ordinary categories. NUL-containing
  category values are rejected, matching the Python normalized boundary.

Each table must have 1–20,000 rows and 1–100 unique named columns. Prediction
column names must match training exactly, but may arrive in a different order.
An entirely missing prediction column may use either variant; otherwise types
must match. No integer parsing, boolean conversion, category normalization,
unit inference or imputation happens in Rust. Perform the existing Python
stateless normalization, including its precision checks, before this boundary.

`prediction_input_support` returns a typed `InputSupport` value containing the
same counts and metadata as the Python schema 1.0 report. `FeatureCounts` encodes
numeric versus categorical fields. No runtime serialization format or FFI is
introduced. Numeric bounds and category inventories are used internally and
are not returned. Feature names remain in the report, just as in Python, so
review names before sharing. Errors do not echo feature values.

Numeric comparisons use training extrema directly, without subtracting extreme
endpoints. Categories are compared exactly without trimming or case folding.
Prediction rows never expand the training reference. The per-feature counts
can overlap; aggregate flags count the union of affected row positions.

## Verification

With an installed Rust toolchain and the existing Python dependencies, from the
repository root:

```bash
cargo fmt --manifest-path rust/Cargo.toml --check
cargo test --manifest-path rust/Cargo.toml --offline --locked
cargo clippy --manifest-path rust/Cargo.toml --all-targets --offline --locked -- -D warnings
PYTHONPATH=src python tools/check_rust_parity.py
PYTHONPATH=src CLOTHO_REQUIRE_RUST=1 python -m unittest discover -s tests -v
```

These checks download no crates or data. The crate has no dependencies, forbids
unsafe Rust, and is marked `publish = false`. It performs no file or network I/O.
Its source has the repository's MIT license.

The cross-language check evaluates 43 wholly synthetic cases using the existing
Python function, emits temporary Rust assertions, then compiles and runs those
assertions. It compares **every report field exactly**, not an approximate
floating-point summary. Cases include reordered columns, missing-type
accommodation, all-missing training, Unicode and literal marker identity,
binary64 extrema and subnormals, duplicate DataFrame indexes, 20,000 categories,
100 features, and 32 deterministic generated tables. The temporary files are
removed when the check completes. This is development-only test generation,
not an input interface or a general serialization format.

The oracle adapter requires the exact report, row-flag and feature key sets and
known enum values. New or removed fields cannot silently disappear during the
comparison; schema drift requires an explicit review of the parity contract.

The Python suite explicitly skips its Rust execution test when `rustc` is absent.
Set `CLOTHO_REQUIRE_RUST=1` to turn that missing compiler into a test failure.
The standalone parity command always fails if Rust is unavailable. Evaluating
the Python fixtures alone is not evidence that Rust compiled or passed.

The initial slice was checked on Linux x86-64 with Rust 1.70.0 (the declared
minimum) and 1.99.1, and Python 3.12.14. Other platforms and dependency versions
are not established by these checks.

## Limits and next migration gates

These are marginal support checks, not an OOD test, a physical validity check,
calibrated uncertainty, or evidence of future accuracy. They do not detect joint
feature support gaps, interior holes, hidden batch dependencies, target shift,
or mixed physical units. The row/column limits are not total memory or runtime
quotas; category strings are caller-owned and their byte lengths are not capped.

No speedup, memory reduction, real-data accuracy, full Python replacement or
cross-platform equivalence is claimed by this first port. Keep Python as the
default while expanding the migration one independently testable boundary at a
time. Before enabling a Rust runtime path, add explicit backend selection and
tests showing unchanged plans, predictions, candidate selection, fit budgets,
failure behavior and saved artifacts. A later estimator port also needs
group-disjoint leakage tests, fold-local preprocessing checks, numeric failure
tests and a documented tolerance policy. Do not infer those guarantees from
this diagnostic's exact count parity.
