# Classic linear I/V-t setup: plan-only sample budget

Use this only after checking the saved configuration is the Classic I/V-t
**linear standard-grid** setup described by the [Keysight EasyEXPERT guide,
Vol. 1 ed. 5, printed pp. 2-24 and 4-17–4-18](https://www.keysight.com/bw/en/assets/9018-05908/user-manuals/9018-05908.pdf).
It uses the documented 2–65,535 ms interval on a 1 ms grid and the integer
per-channel cap `floor(100001 / active measurement channels)`. Do not infer a
mode or channel count from exported columns. Shorter high-speed intervals,
B2900 exceptions, logarithmic modes, segmented runs, and other modes are not
covered.

```sh
python tools/preflight_rram_classic_linear.py \
  --confirm-classic-linear-standard-grid --measurement-channels 2 \
  --sample-interval-us 1000000 --samples-per-channel 50000
```

The displayed duration is `N × interval`, not the final planned sample start.
Under the idealized uniform-start assumption, counting from the first
captured origin (index 0), the final start is `(N−1) × interval`. The tool
counts planned positive starts in (0, 1.02] s and planned starts in
[80,000, 120,000] s. Exact 1 s and 100 ks grid alignment is reported
separately; alignment alone does not assign a nominal anchor identity.
At 1 s, 1/2/4 channels give last starts of 100,000/49,999/24,999 s at their
caps. For two channels, the largest early-compatible interval, 1.02 s,
only reaches 50,998.98 s. The first standard-grid interval that can reach
80 ks at that cap is 1.601 s, which loses the early window.

Valid inputs return 0 with plan arithmetic, not instrument or study
approval. Invalid or unconfirmed inputs return 2 with no JSON. Actual
intervals can differ when measurement time exceeds the programmed interval;
ADC integration, skips, stop conditions, clock linkage, and exposure are not
inferred. This tool never contacts hardware or modifies the retention audit.
