# Positional group membership in regression scoring

Current source asks pandas for positional membership arrays directly when
computing grouped regression scores. The previous implementation requested a
separate pandas `Index` for each group, then converted it into an array.
Both paths operate on the same validated group identifiers and a newly created
RangeIndex. Encounter order, row order within each group, numeric arithmetic,
validation, per-group-then-overall scoring order and equal-group weighting remain
unchanged. This is not a new model, split strategy, metric or Rust backend.
Binary scoring is unchanged.

## Reproduce a bounded end-to-end comparison

With the current source and existing dependencies:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/grouped_metric_positions.py --case all --repeats 9
```

This **fits models on wholly synthetic data**, unlike the no-fit preflight
benchmarks. It calls the public `fit_predict` workflow with dummy and ridge,
three inner folds, ten numeric predictors and 100 query rows. Fixed cases are
2,000 rows with 20 or 500 declared groups, and 5,000 rows with 1,250 groups.
No measurements, model weights or remote services are downloaded. No result
bundle or other file is written; JSON is printed to stdout. Shell redirection,
if used, is the caller's write operation.

Each case warms both paths once, then alternates their order within each pair.
At nine repetitions this is 20 calls per case, with the current plan reporting
14 pipeline fits per call. Allowed repetitions are 1 to 21; these counts are not
a bound on solver runtime or memory. The reference changes only the estimator's
group-membership construction back to the labeled-index path. Timings include
the public workflow and exclude patch setup and result comparison.

Every call's serialized predictions, manifest, selection records and audits
must match the first result exactly, including feature and training fingerprints
and source provenance. Fitted-estimator internals and DataFrame dtype/index
metadata are not compared by this benchmark.
Only timestamps and elapsed time are excluded. Any mismatch aborts the
comparison. Full raw samples, per-pair ratios, separate medians and the number
of faster pairs are reported. There is no timing assertion in the test suite.

Shared-host load can change during a run, so a ratio of two separate medians
can disagree with the typical paired ratio. Read the individual samples and
repeat on the intended machine and workload. Paired timings reduce but do not
eliminate noise. Few-group workloads may see no material benefit; different
models, group sizes or feature types need their own measurements. These checks
establish neither real-data accuracy nor a broad speed guarantee.

## Regression checks and fallback

The frozen mask-based reference in `tests/test_grouped_metric_positions.py`
checks exact metrics and error behavior, including invalid scalar types,
numeric precision, overflow/underflow, unequal groups, singletons, literal group
identity and nonmutation. Larger interleaved cases check the many-group path.
A separate test confirms that scoring no longer requests labeled group indexes.

If a supported pandas version changes positional ordering or exact outcomes,
do not relax the assertions or change scores to retain a timing result. Revert
the membership accessor to the labeled-index reference, rerun the full suite
and repeat the end-to-end comparison. A single host's result does not establish
the untested Python, pandas or platform compatibility matrix.
