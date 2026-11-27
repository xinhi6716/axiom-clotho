# Channel averages do not identify waveform energy

Run `python examples/paired-waveform-energy/run.py`. Two invented,
perfectly synchronized, equal-duration bins have voltage (1,3).
Current is (1,3) in one world and (3,1) in the other. Both have the
same marginal sample multisets and both have voltage and current
means 2. Product-of-means times duration gives 8 for each, but the
paired energies are 10 and 6. A separately fixed paired reference of
7 therefore gives opposite contrast signs, +3 and −1. The discrepancy
is duration times within-window voltage-current covariance, not clock
skew, nor different averaging windows.

[Tektronix's instantaneous power measurement guidance](https://www.tek.com/en/support/faqs/can-i-look-at-and-measure-instantaneous-power-waveform)
describes multiplying voltage and current waveforms. This idealized
example uses piecewise-constant signals, unit-duration bins and no
instrument error. It does not imply that real exported waveform bins
have those semantics or resolve within-bin transients.

For an energy comparison, require validated node and current direction,
a shared timing grid with qualified channel bandwidth and alignment,
and signed samplewise products before integration; or independently
qualify a true-power measurement over the specified window. If only
separate window means or channel marginals survive, hold the energy
ranking unless the lost joint term is independently bounded tightly
enough. Keep invalid or clipped paired samples explicit.
