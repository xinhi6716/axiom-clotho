# A two-terminal drift sign need not be the cell's drift sign

Run `python examples/series-path-drift-sign/run.py`.
The example assumes linear nonnegative series resistance at each
matched read, with observed totals 10 then 12 in arbitrary units.
If the path's change is independently bounded to [-3,3], the cell's
change is sharply bounded to [-1,5]. Two physically nonnegative
decompositions agree with the same observations: path 1→4 gives
cell 9→8, while path 4→1 gives cell 6→11. Tightening only the path
change to [-1,1] identifies a positive cell change [1,3].
These are deterministic conditional bounds, not statistical intervals.

The [Keysight semiconductor parametric test guide](https://www.keysight.com/us/en/assets/9018-02731/installation-guides/9018-02731.pdf)
describes separate Kelvin force and sense terminals. A four-wire reading
can exclude series loss **outside** the sense nodes, not contacts,
access transistors or shared paths still inside them by assumption.
The toy does not model a nonlinear cell, parallel sneak paths, bias
changes or read disturb.

Prespecify the sensed nodes, read bias, routing state and time origin.
Use an independently validated same-path reference or co-located
sense taps to bound the path change before the comparison. Hold the
cell-level sign if the bounds touch or cross zero or if the linear
series topology is not verified. The two-terminal result may still be
reported as a system measurement.
