# A balance screen changes the assignment reference distribution

Run `python examples/prespecified-balance-rerandomization/run.py`.
Six invented physical units have a pre-assignment covariate
`(0,0,0,1,1,2)`. Choose three uniformly, redrawing until the sum of
their covariates is 2. Exactly six of twenty assignments are eligible,
and they are equally probable conditional on acceptance. The two-sided
sharp-null reference distribution uses those six, not all twenty. It
keeps the outcome fixed at its unit and uses the absolute difference of
group sums (equivalent in order to the absolute difference of means).

Two fixed toy outcomes yield correct vs unrestricted tail fractions of
2/3 vs 9/10 and 1/3 vs 1/10: the wrong reference can err in either
direction. These are arithmetic witnesses, not a powered experiment
or evidence for an average-effect hypothesis. The test uses inclusive
ties. [Morgan and Rubin (2012), original paper, §2](https://doi.org/10.1214/12-AOS1008)
describes preregistered rerandomization and its analysis.

For a physical experiment, save the exact unit roster, assignment
probabilities, pre-assignment covariate versions and acceptance rule.
Reject the inference if the observed allocation is impossible under
that design or the actual generator cannot be authenticated. A
deterministic optimizer, unequal proposal probabilities, or a screen
chosen after outcomes is not the uniform rerandomization used here.
