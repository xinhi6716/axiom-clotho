# FET: transfer curves to device predictions

[Clotho home](../../README.md) · [All examples](../README.md) · [繁體中文總覽](../../README.zh-TW.md)

Extract two measurement features, compare grouped regression models, and predict outcomes for three devices from a separate lot. This example uses the **Clotho b7** prepare → plan → run → score workflow. Install the [matching release](../../docs/RELEASE_NOTES_0.3.0b7.md) and extract its source archive for the fixtures.

The transfer curves and endpoints are invented. They demonstrate the file format and workflow; they do not establish physical accuracy or the usefulness of these features for a real endpoint. [FET input guide](../../docs/FET_DEVICE_REGRESSION.md).

## The sample data

There are **15 physical device keys and 105 transfer observations**: four training
lots with three devices each, and a separate prediction lot with three devices.
Every device has exactly one declared seven-point acquisition branch. The local
wafer label `01` and device labels `001`, `002`, `003` deliberately recur across
lots: physical identity is **`(lot, wafer, device)`**, never `device` text alone.

For zero-based lot index `l`, local device index `d`, and step `k = 0..6`, the
invented terminal current is:

```text
base = 2 + (l + d) mod 3
amplitude = 1 + (3*l + d) mod 5
ID[nA] = amplitude * base**k
VGS[V] = k/10
```

The input declares fixed `VDS=0.8 V`, `VBS=0 V`, and `temperature=300 K`, with
positive current, criterion `50 nA`, maximum interpolation bracket `0.1 V`, and
log-slope window `0.1..0.4 V`. These choices make the invented example valid for
the existing extractors; they are **not recommended measurement settings**.
All current and gate-voltage points are retained. No branch, repeat, or device is
selected because an endpoint or a model result is attractive.

`training_targets.csv` and `heldout_outcomes.csv` contain separate handwritten constants in arbitrary units. Neither is calculated from criterion VGS, log slope or another extracted feature. They demonstrate exact device joins and separate training/scoring inputs. Both vectors are public in the generator, so this is not a blinded experiment. Prepare and run do not accept the later-outcome file.

## Input files

- `fixtures/raw/transfer.csv`: the exact canonical eleven-column transfer schema.
- `fixtures/partitions.csv`: all fifteen composite device keys and train/predict.
- `fixtures/training_targets.csv`: only the twelve training device endpoints.
- `fixtures/heldout_outcomes.csv`: only the three prediction device outcomes;
  supplied **only** to the separate score command.
- `fixtures/config.json`: explicit `measurement-regression-v2`, adapter
  `fet-transfer-device-v1`, measurement units/criterion/window, and grouped ML
  choices. Scientific declarations are caller statements, not software proofs.

Regenerate the same bytes in a **new directory**:

```sh
python -S examples/fet-device-regression/generate_synthetic.py \
  --output-dir invented-fet-input
```

The generator uses only the Python standard library, no randomness, timestamp,
network, external files, extracted endpoint, or model fit. It never overwrites an
existing path. An I/O failure can leave incomplete new output; discard that output
and use a new destination rather than repairing it silently.

## Run the four stages

After installing b7, run from the matching extracted source archive or tagged
checkout, with the existing ML dependencies available. These commands do not
fetch dependencies. Choose fresh output directories outside the input directory:

```sh
clotho prepare \
  --input-dir examples/fet-device-regression/fixtures/raw \
  --partitions examples/fet-device-regression/fixtures/partitions.csv \
  --training-targets examples/fet-device-regression/fixtures/training_targets.csv \
  --config examples/fet-device-regression/fixtures/config.json \
  --output-dir fet-prepared

clotho plan --prepared fet-prepared
clotho run --prepared fet-prepared --output-dir fet-run

# Later outcomes enter only this separate saved-prediction evaluation.
clotho score --run-dir fet-run \
  --outcomes examples/fet-device-regression/fixtures/heldout_outcomes.csv \
  --output-dir fet-score
```

For an uninstalled source checkout, replace `clotho` with
`PYTHONPATH=src python -m mlworkflow`.

**Prepare** extracts the measurements without fitting an ML model; log-slope extraction uses descriptive least squares. **Plan** reports the fit budget without training. **Run** uses the grouped regression core. **Score** evaluates the complete saved prediction cohort without refitting. Read the resulting errors as a description of this invented cohort.

## Use with your own measurements

This first device-endpoint adapter accepts **one complete declared branch per
composite physical device**. Additional conditions, repeats or branches reject;
there is no implicit averaging, first-passing branch, fallback acquisition,
automatic sweep labeling, or vendor raw-file importer. Both pinned extractors
must pass every branch before a prepared model bundle exists. A source containing
other acquisitions needs an explicitly justified preparation/lineage step first.

Only `criterion_gate_source_voltage_V` and
`log_current_slope_decades_per_V` are candidate predictors. IDs, fixed biases,
criterion/window settings, sample counts, fit diagnostics and other metadata are
not model inputs. Criterion VGS is not a certified physical threshold voltage;
the descriptive log slope is not automatically subthreshold swing or a regime
identification. Independently justify features for the actual endpoint: a target
or its algebraic proxy is not a valid predictor of itself.

The bundled [upstream FET workflow reference](../../src/mlworkflow/_vendor/semiconductor/examples/fet-measurement-to-features/README.md)
and [signed-current criterion assumptions](../../src/mlworkflow/_vendor/semiconductor/tools/vth-constant-current/README.md)
and [log-slope assumptions](../../src/mlworkflow/_vendor/semiconductor/tools/transfer-log-slope/README.md)
are byte-identical pinned source documentation. **Their optional generator,
single-group selector, annotation, report and standalone-tool instructions are
upstream context, not an assertion that all those companion files ship with
Clotho.** This example's commands and generator are the supported Clotho route.
No private toolbox checkout, network fetch, account, or credential is needed.

Prepared bundles retain input transfer files, physical/acquisition IDs, targets and predictions. Review them before sharing. Hashes and identity checks detect inconsistencies but cannot verify physical independence, experimental history or scientific validity. RRAM `measurement-regression-v1` remains a separate input contract.
