# An absorbing policy stop is not lost follow-up

```bash
python examples/policy-stop-and-lost-followup/run.py
```

Four invented attempts are initiated and the success horizon is tick 5.
One permanently stops without success at tick 1, two succeed at ticks
2 and 3, and one is observed through tick 5 without success. The observed
deployed-policy success fraction is 2/4. Treating the absorbing stop as
censored at tick 1 instead makes the Kaplan–Meier complement 2/3 for this
same log. That complement does not answer the deployed-policy question.

Change only the outcome at tick 1 to loss of follow-up. The unknown attempt
could succeed before the horizon or never succeed, so the sharp bounds
for this roster are 2/4 to 3/4, not a confidence interval. This example
assumes policy stop means known to be absorbing under the proposed
endpoint; pending adjudication does not qualify. It estimates neither
success had a stopped policy continued nor a real-device risk.
[Gooley et al., competing-risks primary reference](https://pubmed.ncbi.nlm.nih.gov/10204198/).
