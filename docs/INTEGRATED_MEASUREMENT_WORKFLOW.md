# One distribution: measurement features and an invented end-to-end walkthrough

**Unreleased source candidate.** This additive integration does not rename the
project or change `mlcalc`, `groupbench`, their Python APIs, or frozen a4 assets.
It removes the second-toolbox-checkout requirement for this bounded workflow.
It is not yet a consolidation of all ML tools, a general real-data training
workflow, or scientific validation.

## Start here

From this source tree, with the calculator's existing dependencies available:

```bash
PYTHONPATH=src python -m mlworkflow demo --output-dir semiconductor-demo --run-invented-example
```

For a distribution built from this candidate, the equivalent new entry point is:

```bash
mlcalc-workflow demo --output-dir semiconductor-demo --run-invented-example
```

No `--semiconductor-source`, second repository, source download, extra package,
hosted model or network operation is needed. The usual calculator dependencies
are needed for prediction. No dependency installation is performed by the tool.
Use a **new output directory with an existing parent**; reruns never overwrite it.

Read `INTERPRETATION.txt`, then `COMPLETE.json`. Completion means all five software
stages ran: invented inputs → existing measurement extractor → explicit lot
holdout audit → target-free prediction → separately supplied held-out scoring.
The six held-out endpoint values are first parsed and used by the legacy modeling
runner at scoring, after prediction succeeds. The packaging wrapper **does hash
and copy the fixture bytes earlier** for component integrity and temporary
staging; those bytes are not supplied to fitting. The unchanged legacy receipt's
`heldout_endpoint_first_reader` field describes the modeling/scoring runner,
not the first filesystem byte access by the wrapper. This is not a blinding or
historical preregistration guarantee.
Do not treat an output directory or partial files as success: failure can leave
`FAILED.json` and partial stage files, and no `COMPLETE.json` is produced.

All data are invented. The three environments repeat the same six numerical
templates under different labels. The fixture demonstrates software contracts,
not independent sampling, distributional transfer, causal effects, model quality
or calibration. Real targets are never inferred from resistance summaries.

## Extract features from your own declared measurement CSVs

This route is **not** limited to the synthetic fixture. It preserves the existing
toolbox's strict format and scientific assumptions without adding an automatic
label join, units inference, grouping selection or ML fit:

```bash
mlcalc-workflow features --input-dir measurements --output-dir feature-export \
  --max-bracket-span-V 0.11 --accept-measurement-assumptions
```

The source-tree equivalent starts with `PYTHONPATH=src python -m mlworkflow`.
The `0.11` bound is only an example: choose and justify a bound for your actual
acquisition protocol. `--accept-measurement-assumptions` is a real declaration,
not an option to bypass validation. The input directory must contain:

| File | Exact columns |
| --- | --- |
| `manifest.csv` | `batch,wafer,device,cycle,replicate,branch,state,condition` |
| `conditions.csv` | `condition,read_voltage_V,temperature_K,read_delay_ms,protocol` |
| `iv_sweeps.csv` | `device,cycle,branch,voltage,current` |

Voltages must be in volts and currents in amperes. Explicit HRS/LRS declarations,
matching acquisition keys, nonzero same-sign voltage/current, complete HRS/LRS
pairs, stable physical device identity and the existing extraction bounds are
required. No state or physical identity is inferred. Review the full pinned
[measurement contract](../src/mlworkflow/_vendor/semiconductor/examples/measurement-to-features/README.md)
before adapting the format; `features --help` exposes all existing options.

Output is `features.csv` plus five schema/provenance reports. Use
`feature_schema.json`'s allowlist deliberately; audit IDs, cycle counts and the
condition/group columns are not automatic model predictors. The default row is
one device at one fixed condition. `--feature-row-unit device-condition` keeps
multiple conditions separate; repeated conditions do not become independent
devices. Choose physical-device or a scientifically justified broader grouping
boundary downstream. No endpoint is created or joined and no model is fitted.

Malformed inputs fail before output creation. Extraction I/O failures may leave
partial files; there is no feature-bundle completion receipt in this upstream
contract. Review all six files and provenance before use, and retry only into a
fresh destination. Exit status 0 is software completion, not physical validity.

## Boundaries and maintenance

```text
mlworkflow (stdlib dispatch + pinned resource integrity)
  ├─ features → unchanged stdlib measurement tools → CSV + JSON contract
  └─ demo → unchanged invented walkthrough → existing groupbench algorithms

groupbench / mlcalc (existing ML implementation and APIs; unchanged)
```

The separate `mlworkflow` namespace avoids importing the eager `groupbench`
initializer for help, resource checks or measurement extraction. The extractor
runs with Python `-S`, so it cannot silently acquire site-package dependencies.
The distribution still retains its existing ML installation dependencies; this
is a runtime import boundary, **not** a new dependency-free distribution.

`mlcalc-workflow components` prints verified upstream commits, paths, SHA-256
digests, Git blob identities and MIT license locations. The manifest lives at
`src/mlworkflow/_vendor/manifest.json`. The legacy bridge pins 13 files from:

- Toolbox `4c0ec2ae3eaa4e0b7afc9aaf8f9bada544e7c0de`.
- Calculator examples `1f26d265411c86b0c6b30a01bc04d3605f312997`.

The current inventory additionally pins the existing research-workflow-utilities
scorecard and its MIT notice for the newer [real-CSV beta](CLOTHO_CSV_BETA.md).

This is **transitional pinned vendoring**, not a second algorithm implementation.
Do not edit these copied components independently. Refresh only from reviewed
upstream bytes; preserve the upstream MIT notices, update exact source/version
identities and rerun byte-parity and end-to-end tests. The calculator source used
for ML is the distribution's existing `groupbench` package, not another bundled
copy. The unchanged legacy runner needs a source-shaped tree: the dispatcher
stages its packaged examples and current calculator Python files into a private
temporary directory, executes locally, and removes that staging tree afterwards.
The output's existing component hashes describe those executed source bytes.
Manifest/hash agreement is integrity evidence, not authenticity or scientific
validation. Python source files must be present; zip-import/bytecode-only
distributions are not supported by this bridge.

The old two-checkout example and frozen downloads remain unchanged. This legacy demo
does not integrate other toolbox tools. The newer real-CSV beta integrates only
the existing research-workflow-utilities scorecard, not that repository in full. It does not depend on agent controllers
or skill bundles. Generalized endpoint joins, installed native adapters and
cross-repository release coordination remain separate future design work.

## Verification without installation or downloads

```bash
PYTHONPATH=src python -S -m mlworkflow components
PYTHONPATH=src python -m unittest discover -s tests -p test_ml_workflow.py -v
PYTHONPATH=src python -m unittest discover -s tests -v
```

The new end-to-end test is not gated on a second checkout or environment variable.
It checks the complete demo against the unchanged legacy runner byte-for-byte,
including all output receipt hashes. Measurement extraction is also compared
byte-for-byte with the upstream tool under `-S`; failure and overwrite cases are
tested. Wheel resource validation must additionally be run on an extracted or
installed wheel outside the source tree. No fresh dependency resolution or
cross-platform support is implied by these local checks.
