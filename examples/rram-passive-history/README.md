# Passive bounded history for RRAM I–V: exact positive-control example

Run `python examples/rram-passive-history/run.py`. No data, training, or extra dependencies. The [Stanford–PKU compact model](https://nano.stanford.edu/downloads/stanford-rram-model) motivates latent-state-dependent conduction. This example is a separate normalized first-order relaxation law, not its implementation.

Given state x and a drive-defined equilibrium u in [0, 1], the exact constant-drive update is x' = qⁿx + (1−qⁿ)u for q in (0, 1) and n integer ticks. It keeps the state in range and gives the same end state whether the interval is split or not. I = V(1+x) is sign-preserving, so VI ≥ 0 for the conduction-only branch. Explicit Euler with rate 1 and time step 2 instead sends x=0 to 2 at u=1. This does not describe displacement current, compliance switching, or an arbitrary real device.

With identical known initial state and final voltage, two waveforms can have the same on-time, off-time, tick count, and final drive but different ordering. Here [1,0,1] and [0,1,1] give currents 13/8 and 7/4. The triangle inequality gives MAE ≥ 1/16 for any single prediction shared by the pair, whereas the chosen state law reproduces its own constructed data. This is a positive control against order-insensitive summaries, not evidence of superiority to history-aware baselines or external measurements.

For a measured comparison, keep whole waveforms and physical devices disjoint, use separately specified reset/initial states and waveform/read conditions, and compare with models given the same history. Prespecify the paired current error and ordering contrast; no paired improvement against a fair history-aware comparator, or incorrect sign of the ordering contrast, means this small state law did not provide the intended gain.

## Distinguishing a single relaxation rate in an exact control

On a uniform sampling grid consider successive increments d_i of a monotone relaxation. If two decay modes have nonnegative oriented weights a,b and contractions q,r in (0,1), d_i = a qⁱ + b rⁱ. Then d_i d_(i+2) − d_(i+1)² = ab(qr)ⁱ(q−r)². This identity is invariant to a constant shift of the samples and to reversal of the sign of every increment. It is zero for one active rate or two identical rates; two distinct active modes with the same sign make it positive. For a=1, b=1/2, q=1/4, r=3/4, the first determinant is 1/8. Fitting a constant ratio to the first two increments underpredicts the third by 1/12.

The exact helper accepts already ordered equally spaced samples. Do not treat a small noisy determinant as a mode count or use uncatalogued timestamps as a uniform grid. Mixtures of more than two rates can give a positive determinant, and mutually opposing weights can give a negative one. The fixed-drive relaxation derivation is a positive control, not a fitted result.
