# Regression grouped scoring: positional grouping

Regression grouped scoring previously compared all N group labels once for each
of G groups: O(N × G) label comparisons. It now collects row positions once using
the same pandas grouping approach as binary scoring. Input validation, encounter
order, within-group row order, metric formulas, checked arithmetic, and per-group
then overall scoring order are unchanged. Exact paired-contrast helpers are
unchanged. No fitting or model changes are involved.

## Local evidence, not a performance guarantee

Anonymous synthetic interleaved groups were measured on one Linux CPU with
Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and numerical thread counts set to one.
At the supported 20,000-row limit with 5,000 groups, three original measurements
were 1.437–1.548 seconds; the scratch positional-grouping candidate measured
0.207–0.273 seconds. A separate 20,000-singleton-group observation was 5.032
versus 0.646 seconds. These are local wall-clock observations, not universal
speedups, timing gates, or evidence about model quality. Small inputs may show
no improvement. The optimization retains O(N) additional grouping storage.

Full output equality was checked at each measured size; a separate 76-case
scratch matrix checked exact JSON or exception type/message parity. The committed
regression tests compare against the prior mask-based algorithm and cover group
identity, order, nonmutation, source validation and arithmetic failures. They do
not assert unstable wall-clock thresholds. Existing metric, saved-scoring and
paired-contrast tests remain applicable. Profiling used only synthetic arrays,
no fitting, and less than 256 MiB generated data. An initial harness filename
shadowed stdlib `profile`; its log was retained and the corrected rerun succeeded.
