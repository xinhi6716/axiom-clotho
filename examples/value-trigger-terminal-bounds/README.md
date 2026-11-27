# Value-triggered sampling and a missing terminal value

```bash
python examples/value-trigger-terminal-bounds/run.py
```

[Keysight EasyEXPERT training, Classic Test Environment, printed p. 3-60](https://www.keysight.com/us/en/assets/9018-01993/user-manuals/9018-01993.pdf)
documents that sampling stops when a configured measured-value event reaches
its configured count. The fixture invents a one-event trigger and a terminal
value at tick 2. It assumes a readout-only gate: stopping recording does not
change the underlying scoring trajectory in this toy. If stopping changes
stimulus or reads change the state in a real experiment, specify the target
policy before defining the unobserved terminal value.

Both arms attempt two units. A records (first, terminal) as (2,1) and
(6,missing); B records (4,4) twice. The attempted-roster terminal mean
difference (A minus B) is `(x - 7)/2`, where `x` is the unrecorded terminal
value. Neither the complete-case difference −3 nor the carried-forward
difference −1/2 identifies this quantity.

- With only the toy endpoint's declared clipped integer scale 0–8, exact
  enumeration gives [−7/2, 1/2]. The two extremes generate the same observed
  log and opposite signs, with all four attempted units in the denominator.
- If an independently justified certificate instead bounds one-tick change
  by 1 on that same scale, x can only be 5, 6, or 7, giving [−1, 0].
  Neither the instrument manual nor the stopped log supplies this certificate.

These are attainable finite-roster bounds under the named assumptions, not
confidence intervals, absorbing policy failures, or settings recommended for
a device. Keep planned terminal time, stop value/reason, capture status and
the attempted roster separate; a stop value never fills the terminal field.
