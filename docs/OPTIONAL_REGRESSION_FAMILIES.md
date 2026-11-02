# Optional grouped regression comparators

Defaults remain `dummy,ridge,extra_trees`. Add optional families explicitly:

```python
from groupbench import benchmark, fit_predict, plan_benchmark

families = ("dummy", "ridge", "extra_trees", "nystroem_ridge", "group_median")
plan = plan_benchmark(frame, target="response", group="study", baselines=families,
                      outer_splits=5, inner_splits=3)
result = benchmark(frame, target="response", group="study", baselines=families,
                   outer_splits=5, inner_splits=3)
```

`frame`, feature units and the scientific grouping axis are supplied by the caller;
this example does not invent or load a research dataset. CLI `--baselines` accepts
the same explicit comma-separated list. Measurement/FET workflow baseline lists
also accept them after the ordinary data/feature/protocol gates. Binary classifiers
do not accept either regression family.

- `nystroem_ridge`: fold-local Nyström RBF map, up to 64 training landmarks,
  gamma = 1 / transformed training width and Ridge alpha 0.1, 1 or 10. Its basis
  sampling and fitting are row-weighted. Selection can use equal-group MAE; this
  does not make the kernel fit or landmark sampling group-balanced.
- `group_median`: one candidate, the lower observed target minimizing absolute
  error with equal total weight for each current training group. It is not a
  median of group medians. Under selection objective `mae`, it remains a comparator,
  not a claim of optimal pooled-row absolute error.

All preprocessing is fitted on the current training partition. Only the group
median receives that partition's groups; kernel and ordinary families do not.
A failure aborts selection/refitting without publishing partial fitted state.
Caller family order determines exact score ties; defaults and the near-tie rule
are unchanged. Successful refitting without an optional family clears its old
metadata.

## Fit budgets and diagnostics

At three inner folds, defaults schedule 21 pipeline fits per selection/refit;
adding median schedules 25, adding kernel 31, and both 35. Five outer folds with
both schedule 175 fits. Plans derive counts from the real schedule without any
fit. Counts do not bound wall time or process memory. Kernel's 16 MiB dense-map
limit is per array, not total RSS. There are no downloads or hidden fallbacks.

Each optional candidate retains per-inner-fit metadata and each optional family
retains final-refit metadata. Actual training rows, transformed width and effective
component count are recorded rather than inferred from configured caps. Landmark
hashes describe local training partition indices, not global row IDs. Median
metadata records only counts, fitted constant and weighting/tie rules, not target
vectors or group IDs. Ordinary-family selection records gain no empty detail fields.

## Saved formats

Opt-in evaluation: schema 1.2, protocol `nested-group-kfold-extended-families-v1`.
Opt-in prediction: schema 1.3, protocol `group-fit-predict-extended-families-v1`.
Both explicitly record categorical encoding (`legacy-v1` or `distinct-v2`) and a
sorted exact capability list matching optional families:
`group_median-v1`, `nystroem_ridge-v1`, or both. Empty, unknown, duplicate or
mismatched capability lists are rejected. Kernel-only and median-only records
require only their own detail schema. Old-only requests keep previous formats.

Strict old readers reject new versions. Generic checksum verification can still
pass and means only byte consistency. Updated validators check saved type/count,
capability, shape and numeric-bound consistency; they do not authenticate execution,
reconstruct training state, prove landmark identity, or establish absence of leakage.
The existing regression/evaluation checks remain limited as documented.

Synthetic software tests cover routing, leakage perturbations, fit accounting,
state recovery and saved-format consistency. They are not semiconductor performance,
causal identification, uncertainty calibration or extrapolation evidence. Minimum
dependency versions and other platforms remain untested unless separately recorded.
