# Compare reciprocal acquisitions by position and by identity

```sh
python examples/mux-order-elapsed-control/run.py
```

Three invented cells are observed in opposite serial orders. Their
elapsed centers, in milliseconds, are 1000, 1018, 1030 by position.
An identical additive offset in the program-end and read-start
timestamps cancels; hard residual errors [−3,3] for program end and
[−2,2] for read start instead give a difference uncertainty [−5,5].
At band (0,1020], the three positions are definitely inside, unresolved
and definitely outside. The middle point must not pass on its center.

The outside cell swaps from C to A on reversal, while the outside
position stays fixed. The code joins both comparisons explicitly:
reordering export rows does not replace reversing the physical order.
This pattern is compatible with a queue effect; it does not rule out
order-dependent drift, fixtures, settling or route-specific effects.
Both attempts are synthetic; no hardware procedure is authorized or
proposed by the fixture. Native nominal identity is not inferred.

[JCGM 100:2008, §5.2](https://doi.org/10.59161/JCGM100-2008E)
motivates tracking shared versus separate uncertainty inputs; the hard
interval arithmetic here is not that guideline's standard uncertainty.
The [Keysight B1500 programming guide](https://www.keysight.com/cn/zh/assets/9018-01851/programming-guides/9018-01851.pdf)
distinguishes reset-referenced measurement-start timestamps in supported
modes. It does not establish alignment to program end, clock scale, or
any numerical bound used here.
