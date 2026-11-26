# An optical map gap can hide a rank reversal

```sh
python examples/height-map-gap-bounds/run.py
```

Keep the original no-return mask and coordinate window before processing
a height profile. For an independently justified slope limit `L`, each
missing location `x` is bounded below by the maximum of
`y[j] − L |x − x[j]|` over valid anchors, and above by the minimum of
`y[j] + L |x − x[j]|`. Reject incompatible anchors. Each envelope is
itself a valid profile under this exact one-dimensional model and both
attain their limits simultaneously. Thus monotone summaries, including
a maximum and a fixed nonnegative weighted mean, attain their extrema
on these envelopes. With noisy heights, registration error, or no
trusted slope limit, these exact-data bounds do not apply unchanged.

On the invented grid `[0, 1, 2]`, profile A returns `[2, missing, 2]`;
profile B is `[2.1, 2.1, 2.1]`. With `L=1`, the missing height can be
1 through 3. Linear interpolation gives A below B. Both the full-grid
mean difference [−13/30, +7/30] and the peak difference [−1/10, +9/10]
instead cross zero. Do not replace the full-grid peak with a nearby
section or a mean over only the returned pixels.

Check the independently justified slope or height limit on a separate
reference or registered follow-up scan. Preserve the mask and fixed
window in the result. Without a defensible restriction on the unseen
height, a finite upper peak bound does not follow from the visible
data. Report the unresolved contrast rather than guessing missing heights.
This is a synthetic demonstration, not performance on a real track.

[Mason et al., optical surface characterization](https://orca.cardiff.ac.uk/id/eprint/179818/3/mason-et-al-2025-a-robust-methodology-for-surface-roughness-characterisation-of-additive-manufactured-parts.pdf)
discuss measurement non-returns; they do not supply a slope bound for
this example.
