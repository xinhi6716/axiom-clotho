# Summarize a full track roster, not just the sections you can measure

Run the Python API with a roster registered before section inspection.
Record every attempt, including confirmed no-bond events and tracks whose
bonding status was not adjudicated. The supplied height is the *bonded-track*
height in mm; there is no imputed height for a failure.

```python
from groupbench.process_contract import BondedTrackAttempt, summarize_bonded_track_attempts

report = summarize_bonded_track_attempts(
    ["track-1", "track-2", "track-3", "track-4"],
    [
        BondedTrackAttempt("track-1", "bonded", "0.21"),
        BondedTrackAttempt("track-2", "failed"),
        BondedTrackAttempt("track-3", "bonded"),  # section not measured
        BondedTrackAttempt("track-4", "unknown"),
    ],
    height_window_mm=("0.20", "0.25"),
)
```

In this invented example the bonding fraction is between 1/2 and 3/4, the
fraction meeting both bonding and the specified height range is between 1/4
and 3/4, and the measured bonded-track mean height is 0.21 mm (one reading).
Both bounds use all four attempts. A bonded track without a usable height is
known bonded, but its threshold result is unknown; a known failure cannot
pass regardless of missing geometry. Report these separately from inspection
failures and recoverable section losses. Choose the height method, window
and physical grouping before analyzing observations. These are deterministic
bounds for the documented roster, not a model score, causal effect, or CI.
