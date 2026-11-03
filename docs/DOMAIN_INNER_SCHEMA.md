# Domain-inner saved declaration contract

Evaluation schema 1.3 (`nested-group-outer-domain-inner-v1`) and prediction
schema 1.4 (`domain-inner-fit-predict-v1`) are explicit opt-in formats. Their
validators check declared domain/group partitions and counts, not whether a
training run occurred or the domains are scientifically meaningful.

Older formats reject explicit domain strategy/configuration fields and the
reserved inner-audit fields `train_domains`, `test_domains`,
`train_domain_count`, `test_domain_count`, and `domain_overlap_count`, even if
configuration fields were stripped while relabeling a bundle. Unrelated legacy
metadata is still permitted. Removing every explicit marker cannot be detected
as a downgrade: saved declarations are not execution provenance.

## Frozen v1 candidate contract

These two domain-inner protocols freeze the following candidate definitions,
independent of the installed estimator's future defaults:

- `dummy`: `dummy_mean`, strategy `mean`.
- `ridge`: `ridge_alpha_0.1`, `ridge_alpha_1`, `ridge_alpha_10`, with the
  corresponding alpha.
- `extra_trees`: `extra_trees_depth_6`, `extra_trees_depth_none`, with max_depth
  6 or null, n_estimators 64 and min_samples_leaf 2.
- `group_median`: `group_median_lower`, with empty parameters.
- `nystroem_ridge`: `nystroem_ridge_alpha_0.1`, `nystroem_ridge_alpha_1`,
  `nystroem_ridge_alpha_10`, with the corresponding alpha and rank_cap 64.

Families follow the saved `config.baselines` order; candidates within each family
follow the order above. A changed grid needs a different protocol/schema contract,
not a reinterpretation of archived v1 files using live estimator defaults.
Evaluation also checks its saved `config.candidates` against this contract.

Every declared candidate must occur exactly once in that order, with matching
family, parameters and objective. Each must have one finite, nonnegative MAE
score per declared inner fold and a finite, nonnegative pooled selection score.
The selected candidate/family and evaluation's per-family winners must match
minimum declared selection scores, resolving ties by candidate order.
Existing optional-family fit-detail checks still apply.

## Limits

These are consistency checks on declarations. Selection scores are not recomputed
from retained inner predictions (none are saved), and are not independently
verified against targets, fitted models or execution. Fold scores need not equal
the unweighted average of domain scores: selection uses pooled row MAE or pooled
group MAE, not equal-domain weighting. A self-consistent fabricated bundle can
pass. Checksums are integrity checks, not signatures or proof of training.

The validator does not verify scientific domain labels, row-level domain digest
preimages, preprocessing correctness, genuine training isolation, unbiased
performance, or arbitrary other metadata. Outer evaluation still holds out
groups, not necessarily domains. Prediction diagnostics are not held-out test
performance. Software tests are hand-authored fixtures, not new scientific runs.
