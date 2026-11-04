# Check invented ALD declarations without fitting

From the repository root, run:

```bash
PYTHONPATH=src python -S examples/ald-contract-preview/preview.py
```

Only the Python standard library is needed. The program creates invented
declarations in memory, makes zero model fits, performs no file or network
reads, writes no files, and prints a JSON report to stdout. Do not use these
numbers as scientific data. The synthetic 12.5 nm / 250 cycles is an average
of 0.5 angstrom/cycle, not a measured steady-growth slope.

Six cases demonstrate consistent declarations, a precursor-vessel/substrate
role conflict, an average mislabeled as a slope, incorrect arithmetic, a
post-process feature, and an unknown replicate. Assertions check the expected
issue codes. No rows are changed, dropped, or admitted to training.

A `no_declared_conflicts` report does not verify a source, physical plausibility,
freedom from leakage, or independent samples. The program is an opt-in Python
example, not a new CLI command or an automatic modeling gate. See the full
[ALD declaration contract](../../docs/ALD_PROCESS_CONTRACT.md) for supported
labels, exact arithmetic, report limits, and scientific responsibilities.
