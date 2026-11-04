# Explicit optional numeric diagnostics

The separate `native/numeric-support` package adds
`clotho_numeric_support.prediction_input_support_numeric(training, new_rows)`.
It returns the existing source-support schema for normalized numeric features.
**This is not a Rust estimator or a replacement prediction engine.** Existing
Python imports, planning, model fitting, prediction, candidate selection,
fold/group logic, run keys, fit budgets and saved-artifact readers keep their
existing routes. The root package's dependencies and pure-Python wheel remain
unchanged. Its input-support module only gains a protocol constant; source-byte
observations therefore correctly change for that file.

## Why this boundary

The existing support diagnostic is relevant when a proposed process condition
lies outside the observed training range or a source feature is entirely
missing. Synthetic profiling identified repeated per-column numeric support
work as a measurable cost. A borrowed NumPy-buffer shortcut was rejected:
Python's GIL does not make all concurrent native writes safe. A second full Rust
float-table copy also consumed much of the potential benefit. The shipped
boundary instead uses private immutable column bytes and safe value-at-a-time
decoding. The benchmark below measures the actual public optional function,
including validation, snapshots and report reconstruction.

## Installation and use

From a trusted matching source checkout, install the normal package and then
explicitly build the optional package:

```bash
python -m pip install .
python -m pip install ./native/numeric-support
```

This source build requires Rust 1.85 or newer and a Maturin build environment.
It may fetch declared dependencies. No compiler, download or automatic native
backend selection occurs during ordinary Clotho imports or commands. This
change does not publish a PyPI distribution or provide release wheels. The
adapter has its own lockfile and PyO3 dependency; the core Rust workspace remains
dependency-free and supports Rust 1.70. Matching protocol constants and bounds
are required at runtime, not merely a compatible package version number.

```python
from groupbench.data import validate_features
from groupbench.input_support import prediction_input_support
from clotho_numeric_support import prediction_input_support_numeric

training = validate_features(training_feature_frame)
python_report = prediction_input_support(training, query_feature_frame)
native_report = prediction_input_support_numeric(training, query_feature_frame)
assert native_report == python_report
```

Training is already-normalized plain float64 source features, with 1–20,000 rows
and 1–100 uniquely string-named columns. Use `prepare_data` or `validate_features`
first so raw integer precision and other admission checks are not bypassed.
Categorical, mixed, sparse, extension, integer and float32 training frames are
not accepted by this optional entry point. Use the unchanged Python diagnostic
for those cases. There is no category encoding or silent fallback. The existing
Python query validator handles raw query validation, schema alignment and
all-missing query type accommodation.

Keep frames stable while the wrapper creates snapshots. Both tables are copied
column by column to immutable little-endian binary64 bytes. The extension
accepts exact tuples of exact bytes, retains their owners before borrowing,
stays attached to Python, and does not call user conversion hooks, borrow NumPy
buffers, detach threads or retain references after return. Authored Rust code
forbids unsafe code; PyO3 implements its own Python C-API boundary. Safe
`f64::from_le_bytes` decoding makes alignment irrelevant. NaN means missing at
this new byte boundary; infinity is rejected. The old `Option<f64>` Rust API
still rejects NaN and uses `None` for missingness.

Feature names stay in Python, including lone surrogates. Rust sees only opaque
positional identifiers after alignment. There is no lossy UTF-8 conversion and
no new restriction on the original Python diagnostic. The core's named Rust
API, as a separate API, uses valid UTF-8 strings.

The wrapper requires GIL-enabled CPython and explicitly rejects other
interpreters and free-threaded builds. This is not a free-threaded support
claim. Inputs are bounded, but the caller still owns source frames and byte
snapshots. No peak-memory reduction is claimed.

## Verification and measurements

Run the original suite and offline core checks, then the dedicated adapter
suite with a matching installed extension:

```bash
PYTHONPATH=src CLOTHO_REQUIRE_RUST=1 python -m unittest discover -s tests -v
cargo test --manifest-path rust/Cargo.toml --offline --locked
PYTHONPATH=src python tools/check_rust_parity.py
PYTHONPATH=src python -m unittest discover -s native/numeric-support/tests -v
cargo fmt --manifest-path native/numeric-support/Cargo.toml --check
cargo clippy --manifest-path native/numeric-support/Cargo.toml --offline --locked -- -D warnings
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/native_numeric_support.py --case all --repeats 11
```

The core has 19 original tests and 11 byte-boundary tests. The cross-language
checker retains 43 original cases and adds 27 byte cases, comparing every
report field exactly. The dedicated adapter suite includes 96 generated
finite-binary64 tables, extrema/subnormals/signed zero, NaN payloads, all-missing
features, reordered/noncontiguous queries, Unicode names, row/feature bounds,
raw-query error parity, malformed results, missing/mismatched extensions and
independent read-only threaded calls. It fails rather than skips without the
compiled extension. Core tests alone do not establish adapter behavior.

`benchmarks/native_numeric_support/reference/timings.json` records one Linux
x86-64, CPython 3.12.14, NumPy 2.3.5, pandas 2.2.3 run. Each backend has two
warmups followed by 11 alternating paired repetitions. Exact report equality
is checked on every call and input immutability after each case. The check
still runs under `python -O`. Median milliseconds for Python versus optional
native were:

- 288 training rows, 32 query rows, 12 numeric features: 4.54 versus 2.98.
- 20,000 training rows, 2,000 query rows, 100 numeric features: 66.04 versus 31.31.
- 20,000 training rows, 20,000 query rows, 100 numeric features: 89.16 versus 61.24.

These are host-specific warm synthetic diagnostic timings, not forecasts for
other workloads. Import/build time is excluded. No estimator is fitted by the
benchmark; no model-training, default-planning, predictive-accuracy or
cross-platform improvement is implied. Reference JSON includes all samples,
environment and hashes of the benchmark, Python sources and measured binary.
Only Linux x86-64 CPython 3.12 was exercised; adapter Rust 1.85 and stable 1.99.1,
and core Rust 1.70 and stable 1.99.1, were compiled. Other supported-version
combinations are not established by these checks.

## Scientific and privacy limits

This remains a marginal diagnostic, not an OOD detector, uncertainty estimate,
physical-validity check, evidence verification or permission to train. It does
not detect hidden group leakage, joint support gaps, interior holes, mixed
units or target shift. Returned feature names and aggregate counts can be
private even though values, extrema and row-level flags are not returned.
Never infer model or research validity from exact count parity.
