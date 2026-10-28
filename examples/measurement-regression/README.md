# RRAM: measurements to predictions

[Clotho home](../../README.md) · [All examples](../README.md) · [繁體中文總覽](../../README.zh-TW.md)

Run the complete **prepare → plan → run → score** workflow on invented I–V measurements. You will get device-level resistance features, a grouped model comparison, six device predictions and a separate evaluation against later outcomes.

## Run the example

Use the tagged checkout or matching source archive with the existing Clotho dependencies. Run from the repository root and choose fresh output directories outside the inputs. These commands install and download nothing.

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

With the matching distribution installed, `clotho prepare`, `clotho plan`, `clotho run` and `clotho score` are the equivalent commands. Supply the same local fixture paths; the CLI does not download them. The older `clotho demo` remains a separate bundled example.

## What to expect

| Stage | Expected result |
| --- | --- |
| Prepare | 18 devices: 12 training devices in four batches, plus six prediction devices in two disjoint batches. Repeated cycles are aggregated within each device. |
| Plan | Three outer folds, two inner folds and a Dummy/Ridge comparison: **40 planned pipeline fits**, including refit for prediction. Planning itself does not fit. |
| Run | A comparison report and predictions with the complete device-key set. |
| Score | All six later outcomes, row and equal-group metrics, and **zero additional fits**. Undefined R² is reported explicitly. |

Open **`clotho-run/report.html`**, then **`clotho-evaluation/report.html`**. Timestamps and elapsed times can vary; the meaningful checks are device identities, predictions, input hashes and scores.

## What the fixture represents

Every value is invented. The raw files and handwritten endpoints are reused from the earlier semiconductor walkthrough; the labels are not calculated from resistance features. Three environments repeat six numeric device templates. The example demonstrates the software workflow, not independent sampling, transfer performance or real-device accuracy.

For your own measurements, use the [input guide](../../docs/CLOTHO_CSV_BETA.md) for units, declarations, identity joins, groups and failure handling. Keep private inputs and results out of public example folders.
