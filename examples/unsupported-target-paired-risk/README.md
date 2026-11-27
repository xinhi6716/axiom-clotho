# An overlap win can reverse on the full target roster

```sh
python examples/unsupported-target-paired-risk/run.py
```

On an invented held chip, three overlap units have known binary endpoints
and two units in a recipe absent from the labeled source have no terminal
labels. Two frozen models issue probabilities for the same five units.
The three known units give a mean Brier difference A−B of −1/4.
Reporting that as the full-chip result silently drops 40% of the roster.

For each unlabeled unit, probabilities (A,B)=(1/4,3/4) give a loss difference
of −1/2 if the endpoint is 0 and +1/2 if it is 1, using the **same** endpoint
for both models. Enumerating both unknown labels gives sharp full-roster
bounds [−7/20, 1/20], with denominator five in every completion. Using only
the generic per-unit loss-difference range [−1,1] instead gives
[−11/20, 1/4]. The paired interval tightens the bounds but does not determine
the winner. A recipe with no reference rows needs labeled target validation,
not just a finite estimated weight.

[Sugiyama et al., Importance Weighted Cross Validation, §2.1](https://www.jmlr.org/papers/v8/sugiyama07a.html)
motivates the distinction: for covariate shift the labeling law is assumed
unchanged and importance weights are density ratios. With source mass zero
and positive target mass in a discrete recipe, no source reweighting
represents that mass. Even on overlap, matching features does not establish
an unchanged labeling law. This fixture uses directly observed target
labels on overlap, not labels imputed from the source. Its bounds are
finite-roster statements, not confidence intervals for future chips.
