# Bound a fixed rank over the uncertainty of an encoded reading

```sh
python examples/quantized-rank-enclosure/run.py
```

The [Keysight B1500 programming guide, Data Resolution](https://www.keysight.com/cn/zh/assets/9018-01851/programming-guides/9018-01851.pdf)
documents different output increments by format. An output increment is
not an error bound. The invented box below already assumes validated,
compatible operand bounds; it does not infer a device configuration,
rounding mode, confidence interval or total measurement uncertainty.

For each frozen reference point, bound squared distance coordinate by
coordinate: nearest point in the interval below, farthest endpoint
above. Apply the same kth order statistic to both vectors of distance
bounds. These enclose the score since kth order statistics are
coordinatewise monotone. Then reverse the endpoints under
`p(s) = (1 + count(calibration >= s))/(n + 1)`. A whole p interval
at or below alpha certifies alert; a lower endpoint above alpha
certifies no alert. Otherwise the enclosure alone is inconclusive.

With references (−1,0),(1,0), input box [−1,1]×[0,1], and k=2, individual
distance bounds yield score [0,5]. But the maximum of the two distances
is `(1+abs(x))² + y²`, giving exact [1,5]. The cheap rank interval
[1/2,1] is ambiguous even though all feasible points alert: with the
tight interval, p is 1/2 everywhere. For k=1, feasible points (−1,0)
and (0,1) instead demonstrate truly opposite decisions. They are
hypothetical latent operands, not probabilities or observed reads.

Keep the reference, calibration and cutoff fixed. Unknown format,
compliance flags, absent provenance, or a transformation coupling inputs
require separate checks; a score bound cannot establish an input box.
Outer-box bounds may stay valid but loose under known operand dependence.
