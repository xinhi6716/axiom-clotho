# Minimum direct-dependency runtime check

## Scope

Recorded on 2026-11-04 on Linux x86_64 (glibc 2.41), using an isolated
CPython 3.10.22 environment without system or user site-packages. The four
runtime dependencies declared in `pyproject.toml` were pinned to their exact
minimum versions: NumPy 1.24.0, pandas 2.0.0, scikit-learn 1.4.0 and
threadpoolctl 3.0.0. This is not a claim about Python 3.10.0, minimum transitive
dependencies, every supported combination, Windows/macOS or the separately
built optional native Rust adapter.

The recorded transitive/build-tool resolution is in
[minimum-runtime-requirements.txt](minimum-runtime-requirements.txt), including
SciPy 1.15.3 and setuptools 77.0.3. Its hashes select the exact downloaded wheels
for this platform. They are not a portable lock for other interpreters or OSes.
The project is built from this checkout, not installed from a public package index.

The comparison environment was Python 3.12.14, NumPy 2.3.5, pandas 2.2.3,
scikit-learn 1.8.0, SciPy 1.17.0 and threadpoolctl 3.6.0 on the same Linux host.
No runtime dependency bounds, public signatures, scientific protocols, model
selection or score arithmetic were changed.

## What the check found

The pre-fix source at `a21ffc7425a9dc766345565ccb0a99c5d3303e4b`
ran 1,379 tests in the minimum environment: 1,370 passed, six skipped,
two failed and one errored. The failures exposed three distinct assumptions:

1. pandas 2.0 exposes even complete native nullable numeric arrays as object
   arrays through `np.asarray`. Binary probability scoring therefore rejected
   explicitly numeric nullable inputs that worked with pandas 2.2. The narrow
   fix reads the declared native dtype only for exact pandas `FloatingArray`
   and `IntegerArray` types, directly or inside a Series/Index. Missing values
   still fail, and existing range/finite/precision checks run before float64
   conversion. It does not infer numbers from arbitrary objects or strings,
   convert Boolean labels, or accept custom nullable subclasses via this fallback.
2. A sparse-kernel test forbade every CSR-to-dense conversion. scikit-learn 1.4
   legitimately densifies small rank-limited Gram results, not the raw feature
   matrix. The corrected trap rejects the original feature width, including
   copies and subsets; allows only the estimator's actual rank width; and checks
   that its learned basis remains sparse. Production kernel code is unchanged.
3. A fixed numerical illustration intentionally requires scikit-learn 1.8.0.
   Its source-hash negative test previously hit the version guard first on 1.4.
   The test now isolates only that version check locally and traps both data
   generation and fitting. The separate version rejection, source hashes and
   production illustration remain unchanged. This does not regenerate or
   validate the illustration's fixed numerical reference with scikit-learn 1.4.

A separate installed-wheel test exposed a checkout-path assumption in an
ownership assertion. It now checks the helper's exact active package directory,
which holds for both source and installed packages without weakening ownership.

The first broader installed/source discovery ran 1,384 tests but reported three
failing subtests in one dependency-guidance test. That fixture put the active
package parent on `PYTHONPATH` under `-S`; for an installed wheel, this is the
whole site-packages directory, so the supposedly unavailable ML dependencies
became visible again. The corrected fixture stages only the two owned packages,
proves those packages are present and NumPy/pandas/scikit-learn are absent under
the exact subprocess environment, and keeps staging separate from no-write
sentinels. This is a test-environment correction, not a production CLI change.

Five new nullable-probability tests cover ten native dtypes across arrays,
Series and Indexes; exact grouped and ungrouped score parity; non-mutation;
missing values; large/out-of-range values; generic objects; Boolean/string
containers; and custom nullable subclasses whose typed conversion must not run.

## Results and their limits

The corrected source ran 1,384 tests in each of the two recorded environments:
1,378 passed and six existing external-toolbox tests skipped in each (166.596 s
in the minimum environment; 179.519 s in the comparison environment).
`CLOTHO_REQUIRE_RUST=1` required the existing standalone Rust parity check;
this is not testing the optional Python native-extension ABI on Python 3.10.
The tests use synthetic fixtures. They establish compatibility and guard
behavior, not semiconductor prediction accuracy or a performance improvement.

Separate installed-only checks used `python -I` from outside the checkout,
verified both package import locations and the changed module's bytes, and
removed source paths from subprocess environments:

- Independent checks passed 83 nullable-input boundary cases and the public
  synthetic measurement workflow. Preparation, zero-fit planning (40 planned
  fits), running, and later scoring six devices completed. Prediction-cohort outcomes were absent
  from the active input tree until after prediction, scoring performed no fits
  and left the saved run unchanged, and repeated preparation refused overwrite.
- The existing [draft portability smoke tool, pinned to its source](https://github.com/xinhi6716/axiom-clotho/blob/9bab6ffd23a590e79c5c4b85ea605c52e476b67c/tools/check_installed_smoke.py)
  also passed against version 0.3.0b7. It checked four installed CLI entrypoints,
  all bundled component hashes, three portable receipts, Linux tabular
  verification, and identical zero-fit scoring with ML imports blocked.
  The draft PR was not changed, approved, merged or dispatched.
- The five nullable regression tests also passed against the installed package
  under `python -I -O`. This checks that these guards do not depend on assertions.

After the fixture correction, full discovery from outside the checkout in the
minimum environment ran 1,384 tests: 1,378 passed and six skipped in 174.225 s.
This includes tests that deliberately add source paths to child processes. It must not be reported as wholly
source-isolated wheel coverage; the separate checks above establish that scope.

## Reproduction

Use a fresh, already available Python 3.10 environment on Linux x86_64.
The recorded interpreter was an Astral python-build-standalone CPython 3.10.22
build (20261009). Dependency wheel hashes were checked against PyPI release
metadata at download and rechecked locally against the installed payloads.
The interpreter itself is not covered by the dependency lock; an additional
attempt to retrieve its original archive for a saved checksum returned HTTP
403, so no independent upstream Python-archive checksum is claimed.

With approved package access, provision the exact wheel set:

```sh
python -m pip install --only-binary=:all: --require-hashes -r docs/minimum-runtime-requirements.txt
```

Or use an already provisioned wheel directory with `--no-index --find-links`.
Installing the hash-pinned set may fail on an unsupported platform; do not
silently substitute another wheel and call it the recorded environment.

From a clean checkout, with the locked dependencies and a Rust toolchain
already installed:

```sh
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=src CLOTHO_REQUIRE_RUST=1 python -m unittest discover -s tests -v
python tools/check_doc_versions.py
python -c "import setuptools.build_meta as b; b.build_sdist('dist'); b.build_wheel('dist')"
python -m pip install --isolated --no-index --no-deps --no-cache-dir --force-reinstall dist/*.whl
```

Then change to a temporary directory outside the checkout, unset `PYTHONPATH`
and `PYTHONHOME`, and use `python -I` for installed-only smoke checks. Confirm
that `groupbench.__file__` and `mlworkflow.__file__` are in the isolated
environment's site-packages. The pinned smoke tool linked above can be saved
outside the checkout and invoked with the installed interpreter:

```sh
python -I check_installed_smoke.py --source /path/to/checkout --work-dir /path/to/new-smoke-output --expected-version 0.3.0b7
python -I -O -m unittest discover -s /path/to/checkout/tests -p test_nullable_probabilities.py -v
```

The smoke reads only copied public synthetic fixture inputs from `--source`;
it does not import the checkout package. Do not make source paths available
to its subprocesses. See
[the contributing guide](../CONTRIBUTING.md#check-the-installed-wheel) for the
broader, explicitly mixed installed/source suite.

These were local maintainer checks, not hosted CI or a release certification.
The open draft portability workflow remains separate from this check.
