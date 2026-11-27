# A feature timestamp does not certify forecast-time availability

```sh
python examples/forecast-origin-aperture/run.py
```

The [Keysight 3458A Quick Reference, APER command](https://www.keysight.com/zz/en/assets/9023-00006/quick-start-guides/EPSG084364.pdf)
defines an aperture as A/D integration time. The fixture uses ideal boxcar
averages, represents no particular instrument's timestamp convention, and
does not model real read disturbance or electronic readout filters.

At a forecast origin of 0, the synthetic worlds have identical pre-origin
readout and different future readout. The saved center timestamp 0 of an
average over [−1,1) passes a naive time filter. Its average uses a
post-origin bin and exactly reveals the later endpoint in this toy.
With a matched-duration window [−2,0) and identical held labels, the
two forecasts cannot tell the worlds apart. The leaked mean Brier loss
is 0, the fixed rule using the valid past-only window gives 1/2, and
the fixed neutral predictor gives 1/4. No rule is fitted on the held rows.

The gate admits a finite window only when its entire physical acquisition
is complete and its feature is ready by the origin on a verified common
clock. The toy includes equality by assuming event ordering is known:
the ending boundary has no duration and a ready feature at the origin
precedes the forecast. Without a known ordering, require a margin.
Even a past-only acquisition with delayed processing is unavailable.
The rule applies to already recorded features; it does not prescribe
acquisition settings. If an instrument/filter has earlier state or
later data dependencies, audit the actual dependency support instead
of inferring it from a single nominal aperture field.
