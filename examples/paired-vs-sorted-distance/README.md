# A distribution match can assign the wrong value to a unit

```sh
python examples/paired-vs-sorted-distance/run.py
```

On the same three invented units, observed values are (0,0,2). One
forecast is (0,0,3), the other is (2,0,0). The first is better by
identity-matched mean absolute error (1/3 vs 4/3). The second is
better by independently sorted distance (0 vs 1/3): it has exactly
the correct empirical marginal distribution but not the same-unit values.

With equally weighted scalar samples and absolute-distance cost, the
independently sorted distance is the minimum over output reassignments
(1-Wasserstein distance). The fixture checks all allowed index
permutations; repeated zeros produce two optimal permutations in each
case without changing the optimal distance. Sorting therefore neither
supplies a unique alignment when values tie nor establishes any physical
alignment, even when the optimum is unique. It is invariant to prediction
identities; the paired score instead joins on the original unit keys.
Both distances have the readout's units. A reorder of rows with keys
preserved does not change either result.

See [Peyré and Cuturi, *Computational Optimal Transport*, §2.6](https://arxiv.org/abs/1803.00567)
for the one-dimensional sorting result. Report the two scores separately if
both scientific questions matter. The example neither estimates a causal
effect nor uses measurements, external files, or trained models.
