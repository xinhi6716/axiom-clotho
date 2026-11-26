# A finite read aperture is not a point sample

Run `python examples/finite-aperture-peak-alias/run.py` from the checkout.
Three invented waveforms are constant inside each of four equal time bins.
An ideal full-width rectangular read reports mean 1 in all three, but the
peaks are 1, 4 and 2. Splitting the read into two half apertures separates
the constant waveform from the other two; the sharp and broad waveforms
still agree at that resolution and disagree on whether the signal exceeds
2. Identical timestamp, gain and full-aperture mean are not enough.

[Keysight's aperture documentation](https://helpfiles.keysight.com/Standalone_BenchVueSoftware_PW_HelpFiles/PWDMMApp/Content/GUI/Instrument%20Settings/Measurement/Aperture%20Time%20and%20Resolution.htm)
describes the distinction between a configured integration interval and
a sample timestamp. The rectangular kernel is an example assumption,
not a claim that an instrument or an existing dataset implements it.

Record aperture and range configuration, and validate the effective
kernel and bandwidth on an independently recorded reference waveform.
With a fixed prediction and the acquisition kernel known, compare a
forward-filtered prediction against the measured read. An alternate,
independently validated shorter aperture or reference digitizer can
test whether an agreement survives better temporal resolution. Do not
treat shorter-aperture agreement as proof of a peak unless the reference
can resolve it. If the available measurements cannot separate candidate
traces, report only aperture-level agreement and hold the peak or
threshold-exceedance claim.
