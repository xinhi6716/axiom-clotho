# A paired score needs a declared target cycle-history mix

```sh
python examples/cycle-history-mixture/run.py
```

Two frozen rules are scored on the same four invented rows: two uncycled
and two cycled. The candidate minus baseline mean loss is −1 for the
uncycled rows and +2 for the cycled rows. Lower loss is better.

The benchmark contains equal numbers, giving a contrast of +1/2.
A declared target with 4/5 uncycled yields −2/5 instead. No prediction,
label, or within-history loss changes: the contrast is 2−3p for target
uncycled share p and ties at p=2/3. If a target share is only known
to lie between 1/2 and 4/5, both directions are possible. This is
sensitivity to the target definition, not a confidence interval.

Use predeclared deployment weights and the same frozen physical devices
for both rules. Score each supported history separately. Calling the
weighted score external performance assumes the within-history
contrasts transport; the example does not establish that assumption.
Do not select a mix to make a rule win or impute a missing target history.
Group membership is not randomized treatment, and this is not a
causal effect of cycling.

[Product Reliability of Embedded RRAM in 28 nm (IRPS 2026)](https://ieeexplore.ieee.org/abstract/document/11499346)
reported a product-specific difference by prior cycling history. That
abstract motivates retaining history as a declared axis. The invented
losses and population shares are not drawn from the study, and its
result does not determine the direction for other products.
