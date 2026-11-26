# Pooled calibration can hide opposite recipe errors

Run `python examples/pooled-calibration-mask/run.py` from the checkout.
Two invented recipe strata have event rates 1/4 and 3/4, fixed under
both mixtures. A frozen baseline predicts 1/2 in both strata. At a 50:50
mixture its single forecast bin is exactly calibrated, but the recipe
residuals (event rate minus prediction) are −1/4 and +1/4. At a 75:25
mixture the residual becomes −1/8 without changing any within-recipe
rate or prediction. The baseline Brier score remains 1/4 throughout;
a reference with exactly known recipe rates scores 3/16 in both.
Neither near-zero aggregate bias nor stable aggregate Brier by itself
demonstrates cross-recipe calibration.

[Murphy (1973), original probability-score partition](https://journals.ametsoc.org/configurable/content/journals%24002fapme%24002f12%24002f4%24002f1520-0450_1973_012_0595_anvpot_2_0_co_2.xml)
distinguishes reliability from resolution. The exact stratum rates
are available by construction here, not an estimator fitted to held labels.
The recipe reference can be a real comparator only if learned on
separate data using recipe information available to both methods.

Freeze physically meaningful strata and any pooling weights before
scoring on untouched parents. Report actual held stratum counts and
paired calibration residuals; if practical, also report a declared
target mixture. Small or unsupported groups do not support a conditional
calibration claim. No distribution-free guarantee is implied.
