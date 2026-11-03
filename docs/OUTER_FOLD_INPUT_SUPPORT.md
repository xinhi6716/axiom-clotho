# Optional outer-fold input support (no fitting)

```python
from groupbench.input_support import outer_fold_input_support

support = outer_fold_input_support(
    frame[["x", "z"]], frame["batch"], outer_splits=3, random_state=7)
```

Supply exactly the **caller-selected inference-time feature columns**, row order,
group labels, outer fold count and seed used by the benchmark. Targets, identifiers,
and unavailable-at-inference proxies must not be supplied as features. This API
cannot detect a misdeclared target/proxy column or establish that features are
scientifically appropriate. It accepts no target and uses no target to select folds.

This opt-in Python API extends the existing `input_support` module; it is not
called by `benchmark`, CLI commands, or existing plans. It returns a fresh sidecar
mapping, performs no fit or model construction, and writes no files. Default
benchmark schemas, selection, predictions and fit counts remain unchanged.

The existing package `GroupKFold` creates the same outer partitions given the
same selected feature rows, groups, seed and outer fold count. This API does not
check inner-selection feasibility and is not a full benchmark plan. Splitter
randomness is the existing seeded split algorithm, not research data generation.

For each fold, only its training feature rows define the reference. It reports
numeric below/above-range counts, unseen categories, missing query values and
nonmissing query values without observed training reference values, all before
imputation. The reference label is `outer_training_features_before_imputation`.
All-missing training numeric references retain null range counts rather than
fabricating zero. Counts may overlap; `flagged_row_fraction` is the number of
query rows with any flag divided by that fold's query row count, not a per-feature
count sum. Equality at a numeric boundary is not flagged.

Only feature names, aggregate row/group counts and fold numbers are emitted;
no row/group identities, row-level indicators, extrema or category inventories
are returned. Join to a compatible benchmark by fold number only after confirming
identical inputs and split configuration; this sidecar does not authenticate
another bundle or infer linkage from a seed alone.

**Limitations:** min/max checks cannot detect holes inside a range, unseen joint
feature combinations, target/concept shifts or future errors. Flags are not a
calibrated uncertainty estimate or an OOD test; no flags is not evidence of safety.
A flagged input may still be predicted accurately. These summaries do not alter
selection, abstain, remove rows or change predictions. They have not been shown
to remedy the frozen synthetic study's selector extrapolation miss. No study
was rerun or retuned to develop this diagnostic.
