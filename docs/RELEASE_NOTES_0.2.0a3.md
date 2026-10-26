# 0.2.0a3 prerelease changes

These notes describe source version **0.2.0a3**, not live artifact availability.
Check the [releases page](https://github.com/xinhi6716/ml-super-calculator/releases)
for versions actually published, or use an explicitly reviewed artifact handoff.
Match the exact artifact to its supplied SHA-256 record; a filename alone is
not verification. The existing
[0.2.0a2 download](https://github.com/xinhi6716/ml-super-calculator/releases/tag/v0.2.0a2)
is unchanged.

[Calculator overview](../README.md) · [繁體中文總覽](../README.zh-TW.md)

## What changes from a2?

This version starts from reviewed source
[`50b2bcbaa7f8e7dc8c72e1b62529cfd1e5d74298`](https://github.com/xinhi6716/ml-super-calculator/tree/50b2bcbaa7f8e7dc8c72e1b62529cfd1e5d74298).
The frozen a2 source is
[`20d17c1d18dcb5b7ecc50681f9287e7b5ea7250a`](https://github.com/xinhi6716/ml-super-calculator/tree/20d17c1d18dcb5b7ecc50681f9287e7b5ea7250a).
Version metadata and these release documents are applied on top of the former
for a3; **50b2 is the starting revision, not the final a3 archive
hash or release tag**.

| Change | What the user sees |
| --- | --- |
| Regression output-path preflight | `demo`, `run` and `predict` reject already impossible destinations, such as a file or broken symlink in the directory path, before model work. |
| Binary output-parent preflight | Binary output checks also reject those impossible parent paths before training. The existing fresh-bundle rule remains; binary overwrite is not added. |
| CSV header identity | Parsing fails if pandas changes the header names that the earlier CSV validation checked. Re-export ambiguous input as UTF-8 with at most one leading BOM and unambiguous names; the tool does not silently rename columns. |
| Python API feature lists | A bare string or bytes value for `features` is rejected instead of being interpreted as an iterable of characters. For one column, use `features=["xy"]`, not `features="xy"`. |
| Source examples and navigation | A task-first [examples guide](../examples/README.md), a reproducible synthetic unit-equivalence example, and a selected version-table consistency check are included in the source distribution. |

The [reviewed source diff](https://github.com/xinhi6716/ml-super-calculator/compare/20d17c1d18dcb5b7ecc50681f9287e7b5ea7250a...50b2bcbaa7f8e7dc8c72e1b62529cfd1e5d74298)
contains the four guards above in three runtime files. Model families,
preprocessing algorithms, splits, metrics and candidate grids are not changed
by those fixes. The unit-equivalence example is an exposed synthetic software
demonstration, not a new model, device benchmark, causal result or performance
claim.

## Install a verified a3 wheel

Only do this after obtaining the reviewed artifact and its matching
SHA-256 manifest. Select an actually published artifact or an explicitly reviewed handoff; these commands do not download a3.
Use a fresh virtual environment so an earlier installation cannot mask the
version being checked.

Expected filenames:

- `ml_super_calculator-0.2.0a3-py3-none-any.whl`
- `ml_super_calculator-0.2.0a3.tar.gz` (source distribution)

In the directory containing the wheel and supplied checksum manifest, check the
wheel's bytes against that manifest. On a system with `sha256sum`:

```sh
sha256sum ml_super_calculator-0.2.0a3-py3-none-any.whl
```

Compare the complete printed digest with the independently supplied record.
A checksum detects a byte mismatch; it does not establish the source's identity
or trustworthiness. Do not continue if the file or digest is missing or differs.

Then, in a fresh environment (POSIX shell example):

```sh
python -m venv .venv-a3
. .venv-a3/bin/activate
python -m pip install ./ml_super_calculator-0.2.0a3-py3-none-any.whl
python -c "from importlib.metadata import version; print(version('ml-super-calculator'))"
mlcalc --help
```

The version must print `0.2.0a3`. Installation may fetch dependencies from your
configured package index; this wheel does not bundle them. Installing or reading
help does not fit a model. Do not treat a successful installation as acceptance
of every supported Python version or operating system.

For a modeling smoke run, `mlcalc demo --out runs/a3-demo` **fits models** and
writes local synthetic results. Use a new path. It is optional for a reader,
and is not a real-data validation or a hard compute-budget promise.

The wheel installs the Python package and entry points. Repository fixtures,
illustrations and source tests belong to the source distribution or checkout,
not an assumed current working directory after wheel installation. Follow each
example's dependency and source-hash requirements; not every example is a
wheel-installed command.

## Compatibility and limits

- These guards intentionally reject previously accepted ambiguous feature-list
  or header input, and detect some output failures earlier. Valid ordinary
  inputs retain the existing scientific evaluation protocol.
- Output preflight is not a filesystem reservation or permission guarantee.
  Paths can change between checking and writing; write-time checks still apply.
- No timestamp repair, calibrated uncertainty, causal identification, model
  persistence, hosted compute or new real-data evidence is introduced.
- The source documentation checker covers its marked version tables, not every
  sentence, the installed wheel's capabilities, or scientific validity. Its a2
  comparison is historical and does not establish a3 artifact availability.
- Package validation must record exact source/artifact hashes,
  installation environment and results separately. These notes alone do not establish
  that a particular wheel or source archive passed those tests.
- Keep a2 artifacts and their hashes unchanged. To go back, use a separate
  environment with the exact previously verified a2 wheel, rather than assuming
  an in-place downgrade also restores dependencies or local result files.
