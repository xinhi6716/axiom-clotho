# Axiom Clotho 0.3.0b2 — portable measurement bundle identities

**Release-preparation guide, not a publication receipt.** The distribution remains
`ml-super-calculator`; the product, imports and four console scripts retain their
existing identities. Confirm accepted artifacts on the
[repository Releases page](https://github.com/xinhi6716/axiom-clotho/releases).
No b2 asset URL or PyPI publication is asserted here.

## Changes from frozen b1

- Measurement bundle inventories now compare canonical slash-separated paths.
  Previously, nested `features/*`, `comparison/*` and `prediction/*` files were
  stringified using Windows backslashes and rejected as an unexpected inventory.
  This blocked consuming a valid prepared bundle and scoring a saved run.
- Four focused tests cover native and simulated Windows path serialization,
  all three bundle kinds, unexpected files and POSIX literal-backslash filenames.
  Inventory, receipt, checksum and symlink safeguards are retained.
- This source includes the post-b1 wheel-first installation instructions,
  feature-selection examples, binary workflow guidance and updated roadmap.

Model families, preprocessing, split schedules, metrics, data/receipt schemas and
dependency bounds are unchanged. Frozen `v0.3.0b1` at `aec0c6a` and all earlier
tags/assets remain untouched; b1 does not contain this fix. Do not replace an old
asset or reuse its checksums for b2.

## Installation and reproducible example

After the maintainer publishes accepted b2 artifacts, download its own wheel,
matching source archive and SHA-256 receipt. Verify the full digests first. In a
separate environment with the existing compatible dependencies:

```bash
python -m pip install --no-index --no-deps ./ml_super_calculator-0.3.0b2-py3-none-any.whl
clotho --version
mlcalc --version
groupbench --version
mlcalc-workflow --version
clotho components
```

All version commands should report `0.3.0b2`. `--no-deps` does not supply missing
dependencies; no download or environment repair is performed automatically.
Declared requirements remain Python >=3.10, NumPy >=1.24,<3, pandas >=2,<3,
scikit-learn >=1.4,<2 and threadpoolctl >=3,<4.

Extract the matching source archive for the CSV fixtures, then follow the
[prepare → plan → run → score example](../examples/measurement-regression/README.md).
These are invented fixtures demonstrating software behavior, not device accuracy.
The wheel supplies executable packages, pinned components and licenses; it does
not fetch fixtures or contact a hosted ML service.

## Verification scope

The reviewed fix at `89bf4fa4ce7b35897a738ab6ab34cddbecd4573e` passed **552 source
tests with zero skips** on Linux/Python 3.12.14 using the existing NumPy 2.3.5,
pandas 2.2.3, scikit-learn 1.8.0 and threadpoolctl 3.6.0. The optional legacy
integration cases used the already bundled, manifest-verified semiconductor
source. The regression failed before the fix for the simulated Windows prepared
and run inventories and passed afterward. Four stdlib-only inventory tests also
passed with the already available Python 3.13.5 interpreter on Linux.

**Windows behavior was simulated with `PureWindowsPath`, not executed on Windows.**
No native Windows/macOS run, Python 3.10 run, minimum-dependency matrix, fresh
dependency resolution, uninstall or in-place upgrade is claimed. The existing
POSIX-only safe byte/schema verification restrictions are unchanged. A
`py3-none-any` wheel tag does not establish cross-platform support.

Build/install checks and exact artifact digests belong to the final accepted
release receipt; this preparation guide cannot substitute for that receipt.
