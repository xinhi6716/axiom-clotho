# Complete invented CSV reproduction

All fixture bytes are synthetic. The raw I-V files and literal endpoint values
are reused from the earlier semiconductor walkthrough; labels are not calculated
from resistance features. Three environments repeat six numerical templates, so
these rows do not establish independent sampling, transportability or usefulness.

Run from the repository root with the calculator's existing dependencies. Nothing
is installed or downloaded by these commands. All outputs must be fresh siblings.

```bash
PYTHONPATH=src python -S -m mlworkflow prepare \
  --input-dir examples/measurement-regression/fixtures/raw \
  --partitions examples/measurement-regression/fixtures/partitions.csv \
  --training-targets examples/measurement-regression/fixtures/training_targets.csv \
  --config examples/measurement-regression/fixtures/config.json \
  --output-dir clotho-prepared

PYTHONPATH=src python -m mlworkflow plan --prepared clotho-prepared
PYTHONPATH=src python -m mlworkflow run --prepared clotho-prepared --output-dir clotho-run

PYTHONPATH=src python -S -m mlworkflow score --run-dir clotho-run \
  --outcomes examples/measurement-regression/fixtures/later_outcomes.csv \
  --output-dir clotho-evaluation
```

For an installed distribution built from this source, use `clotho` in place of
`PYTHONPATH=src python -m mlworkflow` with locally supplied fixture/input paths.
The current source tree supplies this example; it is not a separate download
performed by the command. The older `clotho demo` remains bundled independently.

Expected structural results:

- 18 declared devices → 12 training devices in four batches, six prediction
  devices in two disjoint batches. Repeated cycles aggregate within each device.
- Three outer folds, two inner folds, dummy/ridge comparison and target-free
  refit: **40 planned pipeline fits** in this configuration.
- A complete run with comparison report, predictions and matching device keys.
- A separate six-device score, **zero additional fits**, row and equal-group
  metrics, plus explicit undefined-R² status where appropriate.

Open `clotho-run/report.html`, then `clotho-evaluation/report.html`. Do not compare
these invented errors to real scientific results. Timestamps and elapsed times
can differ; test contracts compare substantive row identities, predictions,
source hashes, leakage boundaries and numerical scores.

The [real-input guide](../../docs/CLOTHO_CSV_BETA.md) explains declarations, units,
joining, group isolation, failure receipts, and the beta's deliberate boundaries.
