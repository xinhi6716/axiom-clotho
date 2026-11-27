# A midpoint section remains a local endpoint

```sh
python examples/midsection-and-peak-equality/run.py
```

[Aversa et al., 316L deposition study](https://doi.org/10.3390/met11060932)
and [Jardon et al., single-track geometry evaluation](https://doi.org/10.2351/7.0000516)
motivate retaining the distinction between a local section protocol and
an endpoint obtained after registered profile averaging. No measured
profiles or values are used here. The earlier
[aggregation-order example](../profile-aggregation-order/) shows rank
reversal; this one adds the midpoint and the exact equality condition.

Three invented registered sections have left (4,4,0), midpoint (0,2,0),
and right (0,4,4). The midpoint maximum is 2; the mean of the section
maxima and the maximum of the pointwise mean are both 10/3. Both outer
sections have ties, but their argmax sets and the midpoint's share one
grid coordinate. Change only the right profile to (0,0,4). The midpoint
and mean of maxima are unchanged; the peak of the mean becomes 2.

On a finite common grid with strictly positive section weights summing to
one, peak of mean ≤ mean of peaks. At any grid point their difference
from the weighted maximum sum is a sum of nonnegative deficits. Thus
equality holds exactly when some common grid coordinate maximizes every
positive-weight section. Do not compare arbitrary first selected argmaxes
when values tie. Grid equality does not imply a common continuous
maximizer outside the sampled representation.

If the planned right section is absent, the midpoint endpoint is still
available, but neither planned full-roster aggregation is silently
computed on the two remaining sections or by filling a profile. An
available-section average or an uncertainty interval is a separately
declared estimand. Grid registration and coverage in this example are
assumed, not inferred from profile length.
