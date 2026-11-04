# Optional numeric source-support backend

This separate package exposes
`clotho_numeric_support.prediction_input_support_numeric(training, new_rows)`.
It matches Clotho's aggregate source-support report for **normalized, plain
float64 training features** and uses a safe Rust byte decoder. It does not fit
or predict a model, change Clotho's defaults, accelerate categorical features,
or automatically replace planning/inference diagnostics.

Install the matching current Clotho source first. From a trusted checkout,
building this optional package requires Rust 1.85 or newer, Python 3.10 or newer,
and a Maturin build environment. The core Rust library separately retains its
dependency-free Rust 1.70 contract. The adapter has its own Cargo lockfile.

```bash
python -m pip install .
python -m pip install ./native/numeric-support
```

These are explicit installation commands. They may download declared build
dependencies; ordinary Clotho imports and commands never run a compiler or
download this extension. No prebuilt wheel or PyPI publication is implied by
this source addition. Only GIL-enabled CPython is accepted by the wrapper;
free-threaded Python and other interpreters are not supported by this first
adapter. See the repository's `docs/RUST_NATIVE_NUMERIC_SUPPORT.md` for the
specific platform/version checks actually run.

```python
from groupbench.data import validate_features
from clotho_numeric_support import prediction_input_support_numeric

training_features = validate_features(training_feature_frame)
report = prediction_input_support_numeric(training_features, query_feature_frame)
```

Select only appropriate source features before calling this helper. Training
must already be normalized; numeric integers must pass Clotho's precision
checks before conversion to float64. Categorical, mixed, sparse, and extension
training dtypes are not supported here. They remain available in the existing
Python diagnostic. Raw query validation and alignment always use Clotho's
existing Python function, including all-missing query type accommodation.
There is no automatic fallback after native errors.

Keep both frames stable while snapshots are created. Each column is copied to
an immutable, explicitly little-endian binary64 `bytes` value. The extension
accepts only exact tuples of exact bytes, retains their owners, stays attached
to Python, and reads one 8-byte value at a time through safe Rust APIs. It does
not borrow NumPy buffers, cast pointers, retain payload references after the
call, or decode a second full table of Rust floats. NaN marks missingness at
this explicit byte boundary; infinities are rejected. The old Rust
`Option<f64>` API still requires `None` for missing values.

Python feature names remain in Python, including names with unpaired Unicode
surrogates. Rust receives only opaque positional names after Python alignment;
no lossy string conversion occurs. This does not extend the capabilities of
other Clotho consumers that may require UTF-8 identifiers.

Reports contain the same feature names, counts and schema as the numeric Python
diagnostic. They do not contain values, extrema, per-row flags, or native build
paths. Feature names and aggregate counts can still be private. A report is
not an OOD test, uncertainty estimate, source verification, physical validity
check, proof of independent samples, or permission to train.

To test a built/installed adapter against the matching core source:

```bash
PYTHONPATH=src python -m unittest discover -s native/numeric-support/tests -v
cargo clippy --manifest-path native/numeric-support/Cargo.toml --offline --locked -- -D warnings
```

The dedicated tests require the compiled extension and fail if it is missing.
They are separate from the main Python-only installation tests. Rust-core
checks remain under `rust/Cargo.toml`; do not infer adapter support from the
core's older minimum Rust version. Source is MIT-licensed; dependencies retain
their own licenses, as declared in their respective packages.
