# Axiom Clotho 0.3.0b1 — beta release guide

**Preparation-stage document, not a publication receipt.** This source contains
version `0.3.0b1`. Confirm actual published artifacts at the
[existing repository Releases page](https://github.com/xinhi6716/axiom-clotho/releases).
No b1 asset URL is asserted here. The canonical repository is now `axiom-clotho` (renamed in place);
the distribution/import/CLI identities remain unchanged.

## A complete, bounded local workflow

Axiom Clotho, in the Moirai family, connects caller-supplied measurement CSVs to
explicit training endpoints and later outcome evaluation:

1. `clotho prepare`: pinned measurement extraction, exact device/endpoint join,
   complete cohort membership and declared physical-group checks.
2. `clotho plan`: no-fit nested schedule and explicit fit-budget validation.
3. `clotho run`: existing GroupBench regression comparison and training-only
   refit, followed by target-free prediction for the declared future cohort.
4. `clotho score`: exact join of later outcomes to saved predictions, row and
   equal-group evaluation, and an offline report with zero new fits.

Follow the [four-stage example](../examples/measurement-regression/README.md)
and [caller-CSV contracts](CLOTHO_CSV_BETA.md). Fixture values are invented; no
labels are manufactured for caller data and no real-device accuracy is claimed.
The supported measurement route is fixed-condition device-level regression,
not a universal semiconductor adapter or migration of every toolbox tool.

## Compatibility

The product is **Axiom Clotho**, while the distribution intentionally remains
**`ml-super-calculator`** to retain one owner for existing Python packages and
console scripts. `clotho` is additive. `mlcalc`, `groupbench`, `mlcalc-workflow`
and `import groupbench` remain; `clotho tabular` delegates to the existing
calculator, including its explicitly selected binary workflow and saved analyses.
Existing algorithm and dependency contracts are not renamed or replaced.
[Migration and overlapping-distribution warnings](CLOTHO_BETA_MIGRATION.md).

Legacy [v0.2.0a4](https://github.com/xinhi6716/axiom-clotho/releases/tag/v0.2.0a4)
and all earlier assets/tags remain unchanged. Do not use their checksums for beta
files, replace them, or mistake their older README for this beta's instructions.

## Install a verified local wheel

Only after the maintainer publishes an accepted release, download **that release's**
wheel and checksum receipt. Compare the full SHA-256 digest before installation.
Prefer a separate virtual environment. For an environment already containing the
unchanged compatible NumPy/pandas/scikit-learn/threadpoolctl dependencies:

```bash
python -m pip install --no-index --no-deps ./ml_super_calculator-0.3.0b1-py3-none-any.whl
clotho --version
mlcalc --version
clotho components
```

Expected version: `0.3.0b1`. `--no-deps` does not supply missing dependencies.
Do not silently add an index/download step if the environment is incomplete.
Ordinary source installation may fetch build/runtime requirements; that is a
separate user-controlled action. The intended release route is GitHub-only, not
PyPI or GitHub Packages. No credentials or external services are required at runtime.

The wheel contains executable packages, pinned resource files and licenses.
Use the matching source archive for the four-stage fixture CSVs, examples,
guides, original Celestial identity headers and tests. Runtime commands do not
retrieve those inputs or use remote report assets.

## Verification scope

The reviewed execution core at `79fa4331dc8f6e4362ba040762bf1d652fcc78e0`
passed **546/546 source tests** (zero skips), including optional legacy source
integration. Its independent review confirmed exact joins, grouping/leakage
boundaries, byte-identical pinned components, no-fit saved scoring and complete
source/wheel resource inventories. The subsequent visual refinement changes documentation, original identity assets
and report-rendering identity/CSS only. Algorithms, splits, metrics, input contracts
and existing scientific report copy remain unchanged; source-byte digests and
new report/checksum bytes legitimately change. It requires its own final tests.
A release receipt must identify the exact final source, newly rebuilt wheel and
sdist hashes, and final extracted/installed-wheel test evidence. Earlier private
candidate archive hashes must not be reused for different final documentation.

The core's actual owned-wheel installation used a disposable environment with
already-installed dependencies and offline `--no-index --no-deps` installation;
29 focused tests and the four-stage CLI example passed. This is not fresh
dependency resolution. Minimum dependency versions, alternate platforms,
in-place upgrades/uninstalls and a full cross-version matrix remain unverified.

Browser launch for the new report pages was denied by the OS before rendering;
no bypass was attempted. HTML escaping, CSP/no-external-assets checks, local link
checks and numerical tests do not substitute for browser/mobile/accessibility
acceptance. Scientific causality, calibrated uncertainty, unrecorded dependencies
and real-device generalization are outside this beta's evidence.

## Source and licensing

The selected Celestial spindle/thread headers are original first-party identity
assets, with bundled [glyph notices](assets/CLOTHO_IDENTITY_NOTICES.md). Existing
workflow/scientific diagrams are preserved. Code and invented fixture sources
retain MIT notices; pinned toolbox and scorecard components retain their exact
upstream licenses. Source hashes identify bytes, not authenticity or scientific
validity. Review run outputs for private IDs and values before sharing.
