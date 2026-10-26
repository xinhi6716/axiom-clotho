# Contributing

We want a small local ML calculator that people can actually trust and use. Correct protocols and understandable failure messages matter more than a long model list.

## Development

Use a clean Python environment and install this checkout. No GPU or hosted API is required.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
mlcalc demo --out runs/contribution-check
```

With dependencies already installed, use `PYTHONPATH=src` instead of installing. Do not add runtime network calls or install/download model weights during tests.

### Keep source and download documentation separate

Before changing version guidance, run this standard-library-only, read-only check:

```bash
python tools/check_doc_versions.py
python -m unittest discover -s tests -p test_doc_versions.py -v
```

It parses the two marked tables in [the usage reference](docs/USAGE_REFERENCE.md#version-scope)
and compares their field identities and capability ownership against
`tools/doc_version_contract.json`. Whitespace, row order and inline-code formatting
may change without failing. Missing, duplicate, unknown or contradictory rows fail.
A competing old `**Status:**` paragraph before the table also fails.

This is a deliberately narrow guard: it does not interpret arbitrary prose,
other documents, later historical sections, installed wheels, Git ancestry or
runtime behavior. A pass is not a release or scientific validation. The contract
is reviewed maintainer input, not an automatically discovered capability database.
For a release change, review its actual source/artifacts and regression evidence
before updating both contract and table; never move a frozen tag just to pass.
The current three rows are backed by the public comparison linked below the table
and `test_cli_output_preflight.py`, `test_csv_header_identity.py`, and
`test_api_feature_allowlist.py`. Keep release source, reviewed fixes snapshot and
current checkout identity distinct. No network access, fitting or writes occur.

### Check the installed wheel

Builds require setuptools 77.0.3 or newer for standard SPDX license metadata, following the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/). This is a build-tool minimum, not a new runtime dependency or a change to the declared Python 3.10+ support. The MIT license text is included explicitly.

The following POSIX check uses an already provisioned environment with compatible setuptools, pip, and runtime dependencies. It builds without downloading anything, installs only this project's wheel in a disposable environment, and tests outside the source tree. The disposable environment reuses existing dependencies; this is not a fresh dependency-resolution or version-matrix test.

```bash
checkout="$PWD"
check_env="$(mktemp -d)"
python -c "import setuptools.build_meta as backend; backend.build_wheel('$check_env/dist')"
python -m venv --system-site-packages --without-pip "$check_env/venv"
"$check_env/venv/bin/python" -m pip --isolated install --no-index --no-deps --no-cache-dir "$check_env"/dist/*.whl
(
  cd "$check_env"
  unset PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  "$check_env/venv/bin/python" -m unittest discover -s "$checkout/tests" -v
  "$check_env/venv/bin/groupbench" --version
  "$check_env/venv/bin/mlcalc" demo --out demo --groups 6 --rows-per-group 3 --outer-splits 3 --inner-splits 2 --baselines dummy,ridge
  "$check_env/venv/bin/groupbench" verify demo
)
```

Installed-wheel validation has been run on Linux with Python 3.12.14 only. Other operating systems, Python versions, and minimum dependency versions remain unverified. A `py3-none-any` wheel tag is not a cross-platform test result.

## Good contributions

- A minimal regression test and fix for a split, leakage, schema, or artifact bug.
- Clearer local workflows or accessibility improvements to the offline report.
- Well-scoped local adapters with explicit model/data licenses and resource budgets.
- Documentation that makes assumptions and limitations easier to understand.

For new model families, task types, data downloads, or persistence formats, discuss a design before implementation. Include why it helps a real user, its scientific estimand, resource cost, privacy boundary, and failure behavior.

## Required verification

- Run the complete test suite and add focused tests for the changed behavior.
- Preserve group-disjoint inner and outer splits. Prove that changing outer test targets cannot change their predictions or candidate selection.
- Keep all learned preprocessing inside each training partition.
- Keep `row_position`, predictions, and provenance aligned.
- Test new writes for pre-existing files, races, path aliases, interruption, and symlinks as applicable.
- Never claim synthetic results establish real-data performance, fairness, calibration, or state of the art.

## Data and licensing

Use synthetic test data by default. Do not include private measurements, credentials, personal information, manuscript/review content, or data with unclear redistribution rights. Public-data additions require a source, license, data card, and explicit scientific grouping rationale. Do not copy model weights or another project's code without a separate license review. Contributions must be compatible with this project's MIT license; dependencies retain their own terms.

AI-assisted contributions need the same human review, reproducible tests, and provenance as any other change. Do not submit generated results you did not actually run.

## Reporting a problem

Include Python/library versions, exact command or small API example, expected versus observed behavior, and a synthetic input that reproduces it. Omit private file paths and data. See SECURITY.md before reporting a security issue publicly.
