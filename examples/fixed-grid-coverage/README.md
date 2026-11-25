# Four pointwise intervals are not a simultaneous band

```sh
python examples/fixed-grid-coverage/run.py
```

There are 20 invented devices with the same four read times. Predictions are
frozen at zero. Four devices have one nonzero absolute residual each, at
different times with magnitudes 1, 2, 3 and 4. Every device is held once; the
other 19 provide calibration. At 95%, the conformal rank here is 19. These
are possible held-device allocations of one toy panel, not independent
experimental runs.

- Calibrating and scoring each fixed time separately covers 19/20 at every
  time.
- Looking at the held device's outcomes and reporting only the time with its
  largest realized residual leaves 16/20 covered by the
  pointwise intervals. Ties use the first grid time.
- Taking one maximum per complete calibration device and scoring the held
  device's entire grid covers 19/20 in this example. A simultaneous pass
  implies coverage of every grid time, including a later chosen grid time.

The example uses complete observations on a fixed grid; it rejects missing or
off-grid keys. It says nothing about between-read times, the unmeasured part
of a shorter trace, or clinical or real-device evidence. Selecting a time
with information available before prediction is another target: calibrate
the same rule on each calibration device if its selected scores are
exchangeable. The original functional idea is described by
[Diquigiovanni, Fontana and Vantini](https://arxiv.org/abs/2106.01792);
this is only a finite-grid, unscaled toy and uses no source measurements.
