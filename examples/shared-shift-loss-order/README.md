# A shared target correction cannot reconcile different losses

Run `PYTHONPATH=src python examples/shared-shift-loss-order/run.py`.
These are invented fixed forecasts and zero nominal targets on two small
paired rosters. All targets on a roster share one possible additive
correction, not independently chosen row corrections. Positive contrasts
mean model B has higher loss than model A.

For residuals A=(-4,0,4) and B=(-3,-3,3), every correction in [-1/4,1/4]
favors A by MAE (B−A in [1/6,1/3]) and B by MSE (B−A in
[-13/6,-7/6]). Thus neither robust ranking proves superiority for the
other loss. For A=(0,0), B=(-1/2,1/2) over [-1,1], the two MAE endpoint
contrasts are both zero; the interior contrast reaches 1/2. Paired MSE
is 1/4 throughout. Unlike absolute errors, paired squared errors cancel
the common quadratic term, so their difference is affine in the shift.

[Gneiting, Making and Evaluating Point Forecasts, original JASA article](https://doi.org/10.1198/jasa.2011.r10138)
distinguishes consistent loss functions for distinct forecast targets.
[JCGM 100:2008 §5.2](https://doi.org/10.59161/JCGM100-2008E)
explains that shared references can induce input correlation; it does
not assert that every system has exactly one shared additive shift.
The chosen bounds are deterministic sensitivity inputs, not a
calibration estimate, confidence interval or coverage guarantee.
Freeze the loss, roster and permissible shift structure before
comparing. If the ordering differs by loss, report the trade-off
rather than relabeling the prespecified primary metric.
