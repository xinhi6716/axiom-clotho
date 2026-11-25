# A common read gain can flip a frozen absolute-bin comparison

```sh
python examples/fixed-log-bins-under-gain/run.py
```

The invented early conductances are 1, 2 and 8; later conductances are
2, 4 and 4. A pooled rule predicts no movement on the log scale. The
frozen binned rule adds +1 in base-2 log units below absolute
conductance 4 and −1 otherwise. It uses the same early measurement
as both anchor and bin selector.

Apply a common positive gain to every reading in each episode. The
pooled absolute errors stay 1, 1 and 1 since the log gain cancels in
the difference. If all episodes share an eightfold shift, the absolute
bin cut stays 4, all three readings land in the upper bin, and binned
errors become 2, 2 and 0 instead of 0, 0 and 0. The model comparison
reverses without changes in device drift. If a factor is a known
unit conversion rather than unknown gain, normalize all measurements
to the original units (or convert all compatible fitted cuts with
them) before assigning bins.

This is a fixed-rule counterexample, not training or measurement.
It does not imply all transfer bias is a common gain: changed bias,
state-dependent gain, or drift between reads can also change the
log differences themselves. Do not choose cuts using target outcomes.
