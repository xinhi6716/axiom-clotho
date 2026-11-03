# Optional grouped regression comparators

Defaults remain `dummy,ridge,extra_trees`. Add optional families explicitly:

```python
from groupbench import benchmark, fit_predict, plan_benchmark

families = ("dummy", "ridge", "extra_trees", "nystroem_ridge", "group_median", "group_balanced_ridge")
plan = plan_benchmark(frame, target="response", group="study", baselines=families,
                      outer_splits=5, inner_splits=3)
result = benchmark(frame, target="response", group="study", baselines=families,
                   outer_splits=5, inner_splits=3)
```

`frame`, feature units and the scientific grouping axis are supplied by the caller;
this example does not invent or load a research dataset. CLI `--baselines` accepts
the same explicit comma-separated list. Measurement/FET workflow baseline lists
accept only the older `nystroem_ridge` and `group_median` extensions after their
ordinary data/feature/protocol gates. They reject `group_balanced_ridge`. Binary
classifiers reject all three regression extensions.

- `nystroem_ridge`: fold-local Nyström RBF map, up to 64 training landmarks,
  gamma = 1 / transformed training width and Ridge alpha 0.1, 1 or 10. Its basis
  sampling and fitting are row-weighted. Selection can use equal-group MAE; this
  does not make the kernel fit or landmark sampling group-balanced.
- `group_median`: one candidate, the lower observed target minimizing absolute
  error with equal total weight for each current training group. It is not a
  median of group medians. Under selection objective `mae`, it remains a comparator,
  not a claim of optimal pooled-row absolute error.

- `group_balanced_ridge`: three explicit candidates with alpha 0.1, 1 or 10.
  Each fit assigns its current training row weight `n / (G * n_g)`, where `n`
  is the training row count, `G` the training group count and `n_g` that row's
  training group size. The final estimator minimizes weighted **squared** error
  plus `alpha * ||coefficients||²`, with an unpenalized intercept. It does not
  fit absolute error or promise any group-MAE benefit. Both selection objectives,
  `group_mae` and `mae`, remain available and separate from its fitting loss.

All preprocessing is fitted on the current training partition. Only group median
and group-balanced Ridge receive that partition's groups; kernel and ordinary
families do not. Ridge weighting applies only to the final estimator: numeric
median imputation, StandardScaler and category frequency/one-hot handling remain
ordinary unweighted training-row operations. Ridge's intercept centering is weighted.
Groups are not needed for prediction, and caller `sample_weight` is rejected.
A failure aborts selection/refitting without publishing partial fitted state.
Caller family order determines exact score ties; defaults and the near-tie rule
are unchanged. Successful refitting without an optional family clears its old
metadata.

## Fit budgets and diagnostics

At three inner folds, defaults schedule 21 pipeline fits per selection/refit;
adding median schedules 25, adding kernel 31, and both 35. Adding group-balanced Ridge to defaults schedules 31 fits, or 155
across five outer folds (versus 105 for defaults). It adds exactly `3*K + 1`
fits per selection with `K` inner folds, including its one family refit. Adding all
three optional families schedules 45 fits per selection. Five outer folds with
kernel and median alone schedule 175 fits. Plans derive counts from the real schedule without any
fit. Counts do not bound wall time or process memory. Kernel's 16 MiB dense-map
limit is per array, not total RSS. There are no downloads or hidden fallbacks.

Each optional candidate retains per-inner-fit metadata and each optional family
retains final-refit metadata. Actual training rows, transformed width and effective
component count are recorded rather than inferred from configured caps. Landmark
hashes describe local training partition indices, not global row IDs. Median
metadata records only counts, fitted constant and weighting/tie rules, not target
vectors or group IDs. Group-balanced Ridge details record its explicit LSQR
recipe, current training rows/groups and transformed width, alpha, loss,
preprocessing rule, smallest/largest group sizes and row-weight min/max/sum.
These compact summaries do not authenticate actual executed weights. No target,
group-ID or per-row weight vectors are retained by its fitted primitive/details.
Ordinary-family selection records gain no empty detail fields.

## Saved formats

Kernel/median opt-in evaluation: schema 1.2, protocol `nested-group-kfold-extended-families-v1`.
Kernel/median opt-in prediction: schema 1.3, protocol `group-fit-predict-extended-families-v1`.
Both explicitly record categorical encoding (`legacy-v1` or `distinct-v2`) and a
canonical exact capability list matching optional families:
`group_median-v1`, `nystroem_ridge-v1`, or both. Empty, unknown, duplicate or
mismatched capability lists are rejected. Kernel-only and median-only records
require only their own detail schema. Old-only requests keep previous formats.

Requests including `group_balanced_ridge` instead use evaluation/plan schema 1.4,
protocol `nested-group-kfold-extended-families-v2`, and prediction/plan schema 1.5,
protocol `group-fit-predict-extended-families-v2`. They require
`group_balanced_ridge-v1`; canonical capability order is `group_median-v1`,
`nystroem_ridge-v1`, `group_balanced_ridge-v1`, restricted to requested families.
These new contracts freeze the complete candidate grid, order, objective, winners,
tie rules and fit-detail coverage independently of mutable runtime defaults.
The new family supports ordinary `group_kfold` only. Domain-inner requests reject
before fitting; old domain-inner schemas/grid and fixed temporal recipes stay
unchanged. Old-only requests do not move to these new versions.

Strict old readers reject new versions. Generic checksum verification can still
pass and means only byte consistency. Updated validators check saved type/count,
capability, shape and numeric-bound consistency; they do not authenticate execution,
reconstruct training state, prove landmark identity, or establish absence of leakage.
The existing regression/evaluation checks remain limited as documented.

Synthetic software tests cover routing, leakage perturbations, fit accounting,
state recovery and saved-format consistency. They are not semiconductor performance,
causal identification, uncertainty calibration or extrapolation evidence. Minimum
dependency versions and other platforms remain untested unless separately recorded.

## Standalone group-balanced Ridge and numerical limits

Import `GroupBalancedRidgeRegressor` from `groupbench.baselines`. Its standalone
alpha accepts any finite positive real other than bool; the integrated grid stays
fixed. It uses `Ridge(solver="lsqr", fit_intercept=True, tol=1e-4, max_iter=None,
copy_X=True, positive=False)` with one computational thread. `score` is inherited
row R-squared, not the fitting loss or group MAE. Finite output is not a convergence
certificate. For a matrix with `n` rows and `p` transformed columns, the
conservative magnitude limit is `L = (float64_max / (1024 * max(n,p)^2))^(1/8)`.
Feature magnitudes must not exceed `L`; training-target magnitudes must not exceed
`L * min(1, sqrt(alpha)/sqrt(n))`, and alpha must not exceed `float64_max/1024`.
These headroom checks precede weighted fitting; prediction also checks its dot
product envelope and finite outputs. They are conservative numeric limits, not
accuracy or convergence guarantees.

Dense, CSR and CSC inputs support 1–20,000 rows and 1–6,400 transformed features.
Named feature order is preserved. Retained masks, malformed groups/sparse buffers,
invalid numeric ranges and numerical warnings fail rather than clipping, skipping,
fallback or retrying. Weight sums and equal group totals use float64 tolerance
`64 * epsilon` (relative, zero absolute); equal-sized groups use literal ones.
The at-most-160,000-byte weight vector is not a solver/preprocessing memory bound.
A failed refit preserves prior complete state. Changing units can be necessary to
avoid unsafe numeric intermediates.

Equal-size groups, a single group and all-singleton groups yield ordinary Ridge's
optimization problem at the same alpha, subject to numerical tolerance. This is
not a row-duplication invariance claim: changing row count changes the balance of
data loss and penalty at fixed alpha, and preprocessing remains row-weighted.
For a constant feature, targets `[0,0,0,0,9]` and groups `[a,a,a,b,c]`, the
weighted-squared-loss constant is 3, ordinary row mean is 9/5 and group median is 0.
Their group MAEs are 4, 18/5 and 3. This analytic counterexample shows why fitting
weighted squared error cannot guarantee better group MAE; it is not study evidence.

Group balancing does not repair a bad dependency-group declaration,
pseudoreplication, missingness, selection bias, covariate shift or out-of-distribution
prediction. No scientific study, hyperparameter tuning on exposed outcomes, or
empirical improvement claim accompanies this source feature. New empirical claims
require a separately authorized, preregistered evaluation.

This feature is available in reviewed current source only, not the frozen b7
release downloads. Minimum dependency/Python versions and other operating systems
are not claimed tested without separate receipts.
