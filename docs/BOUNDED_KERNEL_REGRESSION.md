# Bounded Nyström RBF ridge primitive

## Status and scope

`groupbench.kernel.NystroemRidgeRegressor` remains directly importable and is
also the explicit optional `nystroem_ridge` GroupBench family. Existing default
families and their schemas remain unchanged. Request the optional family in
`baselines` or CLI `--baselines`; see [extended family usage](OPTIONAL_REGRESSION_FAMILIES.md).

This is a bounded smooth nonlinear comparator, not a physical or causal model,
uncertainty estimator, extrapolation guarantee, or demonstrated improvement on
semiconductor data. All accompanying tests use anonymous invented software
fixtures; no real/closed research dataset is fitted for this addition.

## Training contract

The primitive receives **already training-partition-preprocessed numeric X**.
It does not impute raw missing features or encode raw categories itself. When
used in a pipeline, the existing numeric median/scaler and categorical imputer/
encoder must be fitted inside each training partition before the kernel step.
Dense and CSR outputs from that preprocessing are supported. Raw missing values
are not silently imputed here, and unseen raw categories do not establish valid
prediction support merely because their encoding produces finite numbers.

Constructor parameters:

| Parameter | Contract |
| --- | --- |
| `alpha=1.0` | Positive finite real value representable as positive float64 |
| `rank_cap=64` | Integer from 1 through 64, not Boolean |
| `random_state=0` | The same explicit bounded integer seed contract used by GroupBench |

Each fit chooses `k = min(rank_cap, n_training_rows)` and sets RBF gamma to
`1 / n_transformed_features`. The basis is sampled uniformly without replacement
from those supplied training rows only, with the declared seed. No targets,
future rows, held-out groups or global vocabulary are consulted to choose it.
Sampling and fitting weight rows equally, **not groups equally**. Any later
group-MAE selection would be a separate evaluation/selection objective, not a
claim of group-balanced training.

`k` is a component/basis-count bound, not measured algebraic rank: duplicate or
constant rows can have lower rank. Gamma can vary between folds when their
training-only vocabularies have different widths. It can also change if extra
constant encoded columns are supplied; no invariance to arbitrary feature-list
changes is claimed. Prediction retains the fitted k and gamma, even for one
incoming row.

The fitted feature map is passed to Ridge with a fitted intercept and explicit
`solver="svd"`. This choice is deliberate. Inspection of the installed
scikit-learn 1.8.0 `_ridge_regression` source found that `solver="cholesky"` can
silently switch to SVD after a linear-algebra failure. Selecting SVD directly
avoids that hidden solver retry. The dense solve operates on at most 64 mapped
columns; its dimension-dependent work is on the order of `n * k**2`, rather
than a dense exact-kernel solve in n dimensions. This is not a wall-time promise.
The inspected installed `_ridge.py` has SHA-256
`c11784ed57fb21b95eb77d1bd3eccfe00aac2bd283b748417972c9dc952a7d61`;
its SVD implementation requests reduced matrices, not an n-by-n left basis.

No `groups` or `sample_weight` argument is accepted or silently ignored by this
primitive. Group-aware baseline routing must pass such metadata only to the
model that explicitly requires it. Prediction requires neither groups nor y.

## Inputs: reject meaning lost by coercion

Targets are handed to the shared `validate_target(y, n_rows)` **as the original
object**, before any local float conversion. Its current numeric and precision
policy remains authoritative; this primitive does not fork that contract. In
particular, an updated central policy can reject non-exact integer conversion
or wider floating dtypes without an earlier kernel conversion hiding their
types. Legacy text/Boolean parsing, where retained centrally, is not widened
or independently reinterpreted here.

Before array coercion, both fit and predict reject active masks or temporal
values in supported concrete containers. This includes nested lists/tuples,
object arrays, pandas object containers, and masked CSR buffers. Sparse
`.data`, `.indices` and `.indptr` are inspected before validation or conversion;
a hidden masked payload is not an observed value. All-false and `nomask`
representations remain accepted. Structured masks, cyclic containers and
excessively nested containers reject; arbitrary generators are not consumed
by the preflight walk. These checks cannot reconstruct semantics already
discarded by a caller before invoking the primitive.

Only **CSR** sparse format is admitted. CSC, COO and other formats must not be
silently converted at this boundary. After admission, CSR structure is checked
on a private sparse copy; duplicate indices are summed and sorted to obtain
equivalent canonical storage. The caller's buffers are not changed. The full
original sparse feature matrix is never densified.

## Resource and numerical bounds

- At most 20,000 rows and 6,400 transformed columns.
- At most 64 basis rows/components.
- One BLAS/OpenMP thread and Nyström `n_jobs=1` in fit and predict.
- Before constructing a dense map, check `n_rows * k * 8` against **16 MiB**.
  At 20,000 by 64, one float64 map occupies 10,240,000 bytes; a 64 by 64
  normalization matrix occupies 32,768 bytes.

This is a **per-array size bound, not a whole-process RSS cap**. Kernel maps,
normalization/SVD workspaces, centered Ridge copies and existing input storage
can coexist. Resource evidence must state the actual workload and measurement
scope; small unit-test RSS is not a maximum-size workload benchmark. Exceeding
a bound raises before map allocation; no lower-rank retry, hidden subsampling,
chunked rescue or dummy fallback is attempted.

Finite values alone are insufficient for safe squared-distance arithmetic.
Compiled distance calculations can overflow without a NumPy warning, and RBF
can then conceal infinity as a finite zero kernel. Features therefore must
satisfy `abs(x) <= sqrt(float64_max / (16 * transformed_width))` before distance
evaluation. This conservative width-derived check leaves arithmetic headroom
and asks the caller to rescale unsafe units. It reduces values directly, without
allocating a full squared-feature matrix. Legitimate finite-distance RBF tail
underflow to zero remains allowed; overflowing distance arithmetic does not.

Checks cover basis values and their training-row identity, component indices,
normalization, resolved map configuration, dense float64 maps, Ridge
coefficients/intercept and predictions. Numerical RuntimeWarning, relevant
SciPy linear-algebra warnings, invalid/overflowing arithmetic and solver failure
raise contextual ValueError. Deprecation, future and unrelated user warnings
are not indiscriminately suppressed or promoted to failure.

A successful fit publishes all learned state together. Any failed or interrupted
fit preserves the previous complete fitted state, if present. There is no skipped
candidate or fallback model. No executable estimator is serialized.

## Inspectable fitted details

The JSON-safe `fit_details_` records implementation identifier, training row
count, transformed width, requested cap, effective component count, gamma policy
and resolved gamma, seed, basis-sampling policy, a canonical little-endian
training-relative index digest, alpha, solver, thread count and map-byte bounds.
It contains no basis feature values. The digest is not source authenticity,
proof of execution or evidence that the data are scientifically appropriate.

The integrated selector retains both inner-fit and final-refit
details and does not label a small-fold basis as though the configured cap were the
effective count. A no-fit plan may report the policy/cap and a row-based map
bound, but cannot learn the encoded width/gamma by fitting a vocabulary during
preview.

## Candidate and compatibility boundaries

The optional registry integration declares alpha candidates `0.1, 1.0, 10.0`,
with the same partition seed/basis across alphas and no gamma search. Explicit
opt-in adds `3 * inner_splits + 1` pipeline fits per selection/refit schedule:
31 rather than 21 at three inner splits, or 155 rather than 105 at five outer
folds. These are scheduled fit counts, not a time or predictive-performance
guarantee. No numerical near-tie policy changes.

Opt-in output uses evaluation schema 1.2 or prediction schema 1.3 and explicit
capability markers. Old strict readers reject these versions. A generic checksum
verifier may still pass the bytes; it does not understand their semantics.
The fixed source inventory includes the kernel implementation.

The focused software checks run on the existing Linux/Python 3.12.14,
NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0 runtime. They
cover reference parity, dense/CSR handling, fold-local preprocessing, tiny and
singular folds, mask/temporal admission, bounded allocation, targeted warnings,
finite-state checks, failed-fit recovery and ordinary numeric-unit conversion.
The declared minimum dependency versions and other platforms are **not executed
verification results**. Public API compatibility is intended, not asserted as
tested across the full supported range.
