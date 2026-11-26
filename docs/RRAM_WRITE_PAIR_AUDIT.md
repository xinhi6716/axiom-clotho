# Audit next-write data before fitting

Run `python tools/audit_rram_write_pairs.py mapped_events.csv` from a source
checkout. It uses only the Python standard library, fits no models, and prints
counts rather than measurement values or identities. It accepts an explicitly
mapped event table, not arbitrary instrument exports:

```csv
chip,cell,episode,transition,event_index,kind,voltage_V,width_s,conductance_S,protocol
chipA,cell1,ep1,t1,0,pre_read,0.2,,0.000010,example
chipA,cell1,ep1,t1,1,write,-1.2,2e-7,,example
chipA,cell1,ep1,t1,2,post_read,0.2,,0.000005,example
```

These are invented values, not device settings or experimental results. Export
chip, cell, episode, transition and event index from known acquisition lineage.
Cell names may repeat on different chips. Indices are nonnegative and increase
within each chip/cell/episode; the three records must occupy consecutive indices.
Use the original event index, not a new row number after discarding events. Other
cells' records may interleave. This strict contract rejects shared-read layouts
and ambiguous groups rather than repairing or imputing them.

Use signed nonzero voltage in volts for all roles, positive write width in
seconds for write only, and positive measured conductance in siemens for reads
only. Every transition must have one consistent declared protocol label. The two read
voltages must match exactly as declared decimal values. Explicit nonzero numbers are
bounded to magnitudes between 1e-150 and 1e150. The write
voltage is the recorded command; do not substitute a nominal sweep range or
infer timings from plots. Do not use this audit alone to verify device identity,
whether a command was delivered, read delay, or unlogged events or aliasing.
Printed chip and cell counts reflect claimed identifiers, not verified
independence, comparable conditions or cross-chip performance.

The aggregate report separates shared read strata (protocol and read voltage)
from exact action strata (also the signed write voltage and width). Both read
and write settings use exact decimal matching: equivalent spellings match, but
distinct settings that round to the same binary64 number do not. Each counts
eligible transitions in strata with at least two distinct declared chip labels.
Two labels with no shared strata mean there is no match under these keys,
not necessarily no continuous overlap. Matching neither
harmonizes units nor bins nearby settings. By default the diagnostic does not
change the exit status. Use `--require-shared-action` to also return exit 1
when there is no exact action stratum shared across at least two declared chip
labels; JSON is still emitted unchanged. This opt-in gate does not certify
physical provenance, independence, or adequate support per action.
Before fitting, reconcile protocol aliases, verify independent die IDs, and
preserve actual post-write delay and read window in the mapping receipt.

To distinguish an omitted transition from a recorded but ineligible one,
supply `--expected-transitions roster.csv` with an independently frozen,
nonempty identity-only roster: `chip,cell,episode,transition`, no other
columns and no duplicate exact identities. The audit adds redacted counts for
missing expected, unexpected captured, and present-but-ineligible expected
transitions. Any mismatch returns 1 even when every captured triplet is valid;
a malformed roster returns 2. A valid roster declares the intended collection
plan, not proof that a device was triggered or a pulse delivered. The audit
does not authenticate or freeze the roster and never imputes a missing event.
Without the option, the legacy report and exit behavior are unchanged.

The tool accepts up to 8 MiB / 100,000 records. Exit 0 means nonempty and all
declared transitions structurally eligible; 1 means zero eligible or one or more
rejected (JSON is still emitted); 2 means the file/timeline is invalid.
See [native RRAM extraction](RUST_RRAM_EXTRACTION.md) for a separate static I–V
workflow, not a next-write dataset.
