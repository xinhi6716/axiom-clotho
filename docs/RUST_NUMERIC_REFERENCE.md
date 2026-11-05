# Reuse an explicit numeric support reference

The optional `clotho_numeric_support.NumericSupportReference` snapshots the
training support once and compares successive query batches against it. This
avoids repeated training snapshots and scans when reviewing many proposed
process-condition tables. It is still the existing marginal support diagnostic,
**not a Rust estimator, uncertainty estimate, physical-validity check or proof
of predictive accuracy**. It does not change any default Clotho command,
model, split, preprocessing, fit budget, prediction or saved artifact.

## Explicit use

Build the matching optional package as described in
[Rust numeric diagnostics](RUST_NATIVE_NUMERIC_SUPPORT.md). There is no automatic
compiler, download, backend selection, cache or Python fallback.

```python
from groupbench.data import validate_features
from clotho_numeric_support import NumericSupportReference

training = validate_features(training_feature_frame)
reference = NumericSupportReference(training)
first_report = reference.compare(first_query_frame)
second_report = reference.compare(second_query_frame)
```

Training must be already normalized plain float64 features, with 1–20,000 rows
and 1–100 unique string column names. The existing query validator retains its
precision, finite-value, schema-alignment and all-missing type checks. Reports
match every field of `prediction_input_support(training, query)` for the
original training snapshot. Counts refer only to the current batch, not a
cumulative history. Query rows never expand the reference. Changing a source
DataFrame later does not update it: construct a new reference explicitly when
the intended training data changes. Use only the intended training partition;
this helper neither chooses folds nor establishes their independence.

The constructor checks a separate `numeric-source-reference-v1` capability.
An older extension fails clearly, while the existing one-shot API retains its
original `numeric-source-support-v1` protocol. Native errors never silently
fall back. Constructor errors are separate from query errors: an infinite
training value is rejected at construction, before any query exists. The
original one-shot function's validation order is unchanged.

## State and ownership

At construction, stable input columns are copied to private immutable
little-endian bytes, scanned in Rust, then released. Rust retains only row
count, feature identifiers, missing counts and numeric extrema. The Python
wrapper retains immutable names, the row count and the frozen native object;
it holds no training DataFrame, Index, NumPy view or training byte buffers.
Query snapshots last only for their call. Keep caller-owned inputs stable while
snapshots are being made. No mutable NumPy buffers or unsafe pointer casts are
used. NaNs mean missingness, and infinities fail.

The Rust core also exposes
`numeric_bytes::NumericSupportReference::from_training` and `.compare`, with
named columns and incoming reordering. Its private fields cannot be changed
through its API. The Python wrapper rejects ordinary field assignment and
reinitialization; this is API immutability, not a security boundary against
Python introspection. The object deliberately has no pickle/persistence format.
There is no debug representation of training extrema. No source values,
extrema or row flags are returned in reports, but retained state, names and
aggregate reports can still be private.

The retained native summary scales with the number of features rather than
training rows. The named Rust API additionally owns feature-name strings;
name lengths are not bounded by the feature-count limit. This is not a peak-RSS
or total-memory guarantee: caller frames and temporary snapshots remain.
The core is `Send + Sync` and read-only calls may share it across Rust threads.
The Python adapter stays attached to GIL-enabled CPython; threaded correctness
checks are not a Python parallel-speedup claim. Categories remain on the
unchanged Python path.

## Reproduce and interpret measurements

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/native_numeric_reference.py --case all --repeats 9 --batches 10
```

The benchmark compares ten successive synthetic batches using the Python
one-shot API, the existing native one-shot API, a fresh reference including its
construction, and an already constructed reference. It rotates measurement
order, retains raw samples, compares every report field after every call and
checks that inputs remain unchanged. It fits no models and reads no targets.
Construction-only samples are also reported; imports and compilation are
excluded. Do not interpret the warm host-specific numbers as default model
runtime, scientific accuracy, other-platform performance or memory savings.

The checked reference measurements and exact tested environments are recorded
in `benchmarks/native_numeric_reference/reference/timings.json`. The recorded Linux/CPython 3.12 run uses NumPy 2.3.5 and pandas 2.2.3.
For 20,000 training rows, 100 features and ten 128-row batches, median time
was 386.68 ms for native one-shot calls versus 254.16 ms for a fresh reference
including construction (about 34% lower). Ten 20,000-row batches were
786.07 versus 559.29 ms. The 288-row, 12-feature case was 37.52 versus
31.86 ms. These are one shared-host run, not stable speedup guarantees.
Small training sets may save little or nothing because Python query admission dominates.
Rebuilding for every query need not be faster; reuse is explicit and useful
only when the same training reference is appropriate for multiple batches.

## Verification

```bash
cargo test --manifest-path rust/Cargo.toml --offline --locked
PYTHONPATH=src python tools/check_rust_parity.py
PYTHONPATH=src python -m unittest discover -s native/numeric-support/tests -v
```

The core retains the 30 existing tests and adds eight reference tests for
snapshot ownership, query/failure isolation, exact numeric behavior, malformed
input, boundaries and shared-thread use. Each of the existing 27 numeric-byte
Python oracle cases now additionally checks the reference twice against every
expected report field. The other 43 original cases are retained. These are 70
fixture cases with additional reference assertions, not 97 distinct fixtures.

The optional adapter retains its original 18 tests and adds 11 reference tests.
Existing full-report fixtures, including 96 generated binary64 tables, now
also exercise the reference. New tests cover released source frames, schema
mutation, repeated and same-instance threaded queries, immutability, protocol
mismatch, malformed native output, raw-query rejection and one-shot error
precedence. The dedicated adapter suite fails, rather than skips, if its
extension is missing. Core tests alone do not establish Python adapter support.

The candidate was checked on Linux x86-64 with core Rust 1.70.0 and 1.99.1.
The adapter compiled with Rust 1.85.0 and 1.99.1; its 29 tests passed with
CPython 3.10.22 / NumPy 1.24.0 / pandas 2.0.0 and CPython 3.12.14 /
NumPy 2.3.5 / pandas 2.2.3. Root discovery ran 1,384 tests: 1,378 passed
and six existing external-toolbox integration tests skipped because their
trusted companion checkout was not configured. No Windows/macOS, free-threaded
Python, hosted CI or complete dependency-version matrix is claimed.
