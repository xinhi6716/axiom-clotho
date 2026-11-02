# A constant baseline must match the question

This **original synthetic software diagnostic** exposes a gap in the existing
Clotho candidate menu: the default selection objective is equal-group MAE, but
the only dummy candidate predicts the unweighted training-target mean. A
training-only group-weighted median is a different, elementary comparison.
This companion does **not** change the model API, candidate menu or defaults.

The example is deliberately constructed, not a new statistical discovery or
evidence that a median wins on real data. Its important counter-result is that
the group median improves group MAE while substantially worsening row MAE.

## Fixed question and actual run

The [protocol](protocol.json) was fixed before the saved author execution:

- Twelve anonymous invented groups, 156 rows. Three groups each contain 40
  copies of target 10; nine groups each contain four copies of target 0.
- The only predictor is the constant 1. IDs and group names are not predictors.
- Seed `20261103`, three outer group folds and two inner group folds.
- One actual public `groupbench.benchmark` call, with existing dummy, ridge and
  ExtraTrees families and their unchanged six-candidate inner search.
- **45 actual pipeline fit calls**, plus nine analytical training constants.
  There was no final refit, seed search or outcome-driven retry.
- Row median, equal-group mean and equal-group lower weighted median use only
  each outer training partition. All seven reported entries use identical
  held-out rows. The `selected` entry is not an eighth fitted model.
- Primary: equal-group MAE. Secondary: row MAE, row RMSE and group RMSE.
  Every result is retained; there is no significance or advancement gate.

The scalar response has arbitrary units. Group identifiers are programming
labels, not evidence of independent physical devices or experiments. Target
values are intentionally public and known by construction: this is not a blind
or prospective experiment.

| Entry | Group MAE | Row MAE | Row RMSE | Group RMSE |
| --- | ---: | ---: | ---: | ---: |
| Product dummy mean | 6.346154 | 3.550296 | 4.213250 | 6.346154 |
| Product ridge | 6.346154 | 3.550296 | 4.213250 | 6.346154 |
| Product ExtraTrees | 6.346154 | 3.550296 | 4.213250 | 6.346154 |
| Product selected | 6.346154 | 3.550296 | 4.213250 | 6.346154 |
| Training row median | 7.500000 | 2.307692 | 4.803845 | 7.500000 |
| Training group mean | 3.750000 | 6.346154 | 6.686669 | 3.750000 |
| Training group median | 2.500000 | 7.692308 | 8.770580 | 2.500000 |

Full-precision [predictions](reference/predictions.csv),
[summary](reference/summary.json), [fixture](reference/fixture.csv),
[training partitions/constants](reference/constant_audit.json),
[actual product manifest](reference/product_manifest.json), and
[cost](reference/cost.json) are saved. The reference directory is a copy of the
complete completed author run, including its integrity manifest. It is not an
independent retrain or independent scientific validation.

Each outer fold has one large group and three small groups. Every training fold
therefore has two large groups and six small groups. Its row mean is `100/13`,
row median `10`, equal-group mean `5/2`, and group-weighted median `0`.
The dummy group MAE is `165/26`; group mean gives `15/4`; group median gives
`5/2`. These analytic identities make the example auditable without fitting.

The recorded selector chooses ExtraTrees in all three folds because its
floating-point inner score is approximately `6e-15` smaller than the other
constant predictors. This is arithmetic-level variation, **not meaningful
evidence that trees learn signal**. No tolerance or selection policy was changed
after observing this. Selection tie policy is a separate possible design task.

## Why weighting and loss are separate

For training group sizes `n_g`, equal-group absolute loss for a constant `c` is

`L(c) = (1/G) * sum_g [(1/n_g) * sum_(i in g) abs(y_i - c)]`.

Thus a minimizing constant is a weighted median of **all training rows**, using
weights `1/n_g`; the common factor `1/G` does not affect the minimizer. This
companion chooses the first sorted target whose exact rational cumulative weight
reaches half of total weight. It is not generally the median of group medians.
No future group sizes or labels determine these training weights.

The row median minimizes training row absolute loss; the equal-group mean
instead matches a group-balanced squared-loss constant. Training optimality
does not guarantee a held-out win. Group-weighted constants remain unchanged
when every row of one training group is repeated equally; the current
row-weighted mean need not. This is not a claim that a full tree or ridge
pipeline would become duplicate-invariant merely by adding sample weights.

Primary documentation supports the existing mean/median dummy strategies and
sample-weight interfaces: [scikit-learn 1.8 DummyRegressor](https://scikit-learn.org/1.8/modules/generated/sklearn.dummy.DummyRegressor.html)
and [mean absolute error](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.mean_absolute_error.html).
The equal-group weighting derivation above follows directly from the declared
loss; it is not an empirical causal explanation.

**Next product proposal:** add an explicitly named, group-balanced median
constant as a same-fold baseline, with group weights recomputed inside every
training fold. Review API, candidate-selection and report compatibility first.
Do not silently replace the existing dummy, reinterpret row MAE, or infer that
this constructed result establishes general superiority.

## Reproduce the frozen reference

This is a version-pinned reference diagnostic, not a general benchmark CLI.
It requires Linux, Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0
and SciPy 1.17.0. Dependencies must already exist; it installs/downloads nothing.
It verifies the four relevant product source hashes in `protocol.json` before
fitting. A changed runtime or source is rejected, rather than silently called
the same reference run.

Use an isolated source checkout at commit
`71c0ebeeedb56ce84cbb801be84aa0baffe05d3f`, and copy this companion directory
from the current checkout into its `benchmarks/method_validation/` directory.
This copying is needed because the diagnostic was authored after that baseline
commit. From that isolated checkout, run:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/method_validation/group_mae_constants/run.py \
  --output /tmp/clotho-constant-diagnostic-new
```

The output directory must not exist. The reference execution took 0.926 CPU
seconds / 0.939 wall seconds inside the recorded runner timing boundary and
126.6 MiB process peak RSS. This is not a general performance benchmark or a
platform-billing measurement. Imports and preflight are outside that timing
boundary. One-thread library limits and a 600 additional CPU-second process
limit are applied on Linux. No Windows runtime support is claimed.

Existing outputs, including symlinks, are refused. An exception may retain
`failure.json`; a hard process kill may not. Only `manifest.json` with
`complete: true` marks completion. Keep incomplete outputs instead of silently
overwriting or retrying them. Hashes check accidental changes, not authenticity,
honest execution or pre-exposure chronology.

## Software tests, separately from fitting

The focused tests perform **no estimator fits**:

```sh
PYTHONPATH=src python -m unittest discover -s tests/method_validation -v
```

They check the original fixture, inverse-training-group-size weights, an exact
half-weight tie, the difference from median-of-group-medians, within-group
replication invariance, finite-target rejection, preflight and no-overwrite
behavior. They do not independently reproduce the recorded trained estimators.

All fixture values and code in this companion are newly authored synthetic
material under the repository's MIT license. There is no external dataset,
private measurement, manuscript/review text, credential or trained model file.
