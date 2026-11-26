# Marginal coverage does not establish a joint holdout

From the checkout, run `PYTHONPATH=src python examples/crossed-factor-roster-support/run.py`.
The two invented rosters each have four dies and four acquisition blocks with two
rows per die and two rows per block. For each occupied crossing, Clotho's
`plan_crossed_holdout` returns a valid singleton plan with all four quadrants
nonempty. The example then checks a **different**, two-die/two-block design by
enumerating its 36 possible held identity sets. Its train rows have neither
held factor and its test rows have both. Rows with only one held factor are
excluded from this comparison (or can be reported as separate probes).

In the connected eight-cycle at least two rows must be excluded. In the two
disconnected four-cycles all rows can be used by putting a component on each
side. Each row is one edge in a bipartite graph; a zero-exclusion assignment
cannot split an edge across sides and therefore cannot split a connected
component. See the [NetworkX connected components definition](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html).
The two-die/two-block enumeration is a teaching check, not an extension of
the singleton API. A zero-exclusion split has empty side probes and is not
a valid four-nonempty-quadrant plan under that API.

Predeclare whether the scientific question requires two crossed held
factors, separate side probes, or a single held factor. Audit the joint
schedule before outcomes, including empty and excluded crossings, and fix
minimum train/test support and physical identity rules. If the design fails
those requirements, report a gap rather than treating combined IDs or more
measurements at the same crossings as joint support. Disconnected components
alone do not guarantee balance, independence, or transfer.
