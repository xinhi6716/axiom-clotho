# A segment reset does not supply elapsed time or a nominal anchor

```sh
python examples/segmented-clock-gap-bounds/run.py
```

[Published retention study](https://doi.org/10.1126/sciadv.adg1946) and
[its Cambridge archive](https://doi.org/10.17863/CAM.89741) motivate
checking segmented capture. The numbers and bounds in this example are
invented. No source measurements, rows, endpoints or gaps are used.

Each segment's exact local clock starts at zero. A bridge adds the
previous segment's last local timestamp, then the elapsed gap from
that sample to the next segment origin. Under assumed non-overlap and
a known clock scale, a missing gap is bounded by [0,∞), not set to zero.
A verified program-to-first-origin interval anchors the resulting sum.
Finite bounds are sharp for the stated rectangular model, where every
combination of bridge gaps and initial offset is possible; dependencies
can make the enclosure conservative. Absent program-epoch evidence
remains unresolved.

In the toy the first segment ends at 8 and the selected second-segment
row is at local 3. For an elapsed acceptance band [13,15], gap [1,4]
gives [12,15], ambiguous; gap [2,4] gives [13,15], entirely inside.
A missing upper bound does not preclude proving outside if the lower
bound already exceeds the band. Inside the elapsed band does not
establish the nominal tag. The combined toy gate requires a separately
verified nominal identity, not a rounded, stitched or substituted
measurement. Same device and episode are assumed in the example and
would require separate provenance in a real audit.
