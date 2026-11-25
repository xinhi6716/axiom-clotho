# Report penetration and adhesion as different observations

```sh
python examples/penetration-is-not-adhesion/run.py
```

The four invented tracks include a reported zero with confirmed
adhesion, positive penetration with failed adhesion, positive penetration
with unknown adhesion, and unavailable depth with confirmed adhesion.
These synthetic combinations demonstrate how to preserve independent
fields; they do not imply that zero penetration establishes bonding
or describe a physical process window.

The depth ledger counts (positive, zero, unavailable) = (2, 1, 1).
The adhesion ledger counts (confirmed, failed, unknown) = (2, 1, 1).
The confirmed-through-observation fraction is 2/4 over all attempted
tracks. The unresolved track is not a failure classification, and the
fraction is not an estimate assuming missingness at random. Preserve
the reported zero without converting it into a missing value or an
adhesion label. A published rounded zero does not certify exact
physical zero; preserve its measurement status and resolution.

[El Cheikh et al., Table 1](https://doi.org/10.1016/j.optlaseng.2011.10.014)
reports penetration geometry. [Piscopo et al., Tables 2 and 5](https://pmc.ncbi.nlm.nih.gov/articles/PMC6747773/)
uses the related records and tracks adhesion separately. They are not
independent experimental replicates. No published rows are included
here. Real records need qualified adjudication and stable attempt
identity; a positive depth alone is not such adjudication.
