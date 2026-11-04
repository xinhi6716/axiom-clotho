# Prediction preflight: validate once, preserve the contract

Material/process tables can have many categorical material, deposition or
instrument settings. Before fitting, Clotho normalizes selected source features,
checks the prediction schema, builds a group-disjoint selection plan, and reports
source-feature support. Repeatedly mapping the same already-normalized strings
does not add another scientific check.

This change removes two repeated query-normalization passes from regression
prediction planning. It also removes those redundant preflight/diagnostic passes
from `fit_predict`; the estimator's public `predict` admission check remains.

## What stays checked

Every public raw-data entry point still runs the existing `validate_features`
checks. Numeric dtype/precision, finite values, unique column names, row/feature
limits, categorical normalization and NUL rejection are unchanged. No DataFrame
attribute, caller flag, cached identifier or process-wide switch marks arbitrary
raw data as trusted.

Private helpers operate only on the normalized copies the workflow already
owns. They still align exact names and types, including entirely missing query
columns. They are not public validation-bypass APIs. Standalone
`validate_prediction_features` and `prediction_input_support` keep validating
their raw inputs. Binary workflows retain their existing normalization path.

The aligned frame is separate from the normalized frame used for historical
prediction fingerprints. In particular, a wholly missing object-typed numeric
query column can be aligned to float64 for prediction without changing the
original hash input. Tests retain the historical hash recipe and four frozen
query identities from pandas 2.2.3. Cross-version pandas hashing is not newly
guaranteed.

Public signatures, prediction schemas, model families, candidate selection,
folds, fit budgets, source support counts and saved-reader behavior are unchanged.
Source-code provenance naturally records the changed package; timestamps and
source digests are not expected to be identical across runs or revisions.

## Measured scope

A warm, synthetic, fit-free benchmark compared base commit
`7edaaabf0da95ac938cb7daf59a1ca6bede48544` with this change on one Linux x86-64
environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0.
Each function had two warmups and nine measured repetitions. Base and candidate
ran sequentially in separate processes with BLAS thread settings set to one.

| Training rows | Query rows | Features (categorical) | Base median plan | Candidate median plan |
| --- | --- | --- | --- | --- |
| 288 | 32 | 12 (4) | 14.2 ms | 9.8 ms |
| 20,000 | 2,000 | 32 (8) | 160.2 ms | 139.9 ms |
| 20,000 | 20,000 | 100 (20) | 785.9 ms | 473.1 ms |

The plans' diagnostic reports and fit budgets matched exactly. Raw timing
samples, environment and source-byte observations are in
[the timing record](../benchmarks/prediction_preflight/reference/timings.json).
These are observations, not performance guarantees. Host load, table shape,
category cardinality and dependency versions affect the result. The fixture is
invented and is not a physical materials model.

No estimator is fitted by this benchmark. It does **not** establish faster model
training, improved accuracy, physical validity or generalization. Runtime Rust
integration remains off: the measured first bottleneck was repeated Python
normalization, not a reason to add an unmeasured serialization layer.

## Reproduce and verify

With existing Python dependencies, from a checkout:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/prediction_preflight.py --case all --repeats 9
PYTHONPATH=src python -m unittest discover -s tests -p test_prediction_normalization.py -v
PYTHONPATH=src CLOTHO_REQUIRE_RUST=1 python -m unittest discover -s tests -v
```

The benchmark prints JSON and makes no model fits, downloads or output-file
writes. Use the same script and dependency environment with each source revision
selected through `PYTHONPATH`; it does not silently replace the import path.
The last test command requires an installed Rust compiler for the existing
cross-language diagnostic checks. Nothing is downloaded automatically.
