# 0.2.0a4 — UNPUBLISHED local candidate r3

This is a local packaging candidate, not a download announcement. No a4 tag,
GitHub Release, wheel upload or source-main update is authorized by these notes.
The published [a3 release](https://github.com/xinhi6716/ml-super-calculator/releases/tag/v0.2.0a3)
and its checksums remain unchanged. The README's a3 installation route still
refers to that frozen download, not to this candidate.

## What would a4 make available in a package?

The source basis is accepted main
[`e4eea047e3d2ba65635441335289dc0abbbe1323`](https://github.com/xinhi6716/ml-super-calculator/tree/e4eea047e3d2ba65635441335289dc0abbbe1323),
followed only by candidate version metadata, this note and the changelog.
That base is not a final candidate commit or archive hash.

This r3 refresh is separate from the frozen local a4 r2 candidate, whose source
and archives remain unchanged. Both use the candidate version string 0.2.0a4;
identify them by their source-manifest and archive hashes, not the version alone.
Compared with r2, r3 adds the interval CLI alias, regression bilingual controls
and the small-value display correction. Neither candidate is published.

- Regression and binary reports have native English/繁體中文 controls and local section links.
- `mlcalc evaluate-intervals` delegates to the existing no-fit saved-interval evaluator; the original module command remains available.
- Regression reports use rounded scientific notation for nonzero metrics below 0.0001 in magnitude, so small target-unit errors do not appear as zero. Stored numbers and ordinary-range formatting stay unchanged.
- Reports keep the worst 15 groups and offer **View all N groups** for the
  complete saved per-group metrics in ID order when more groups exist.
- Comparisons explain the supplied group boundary and the all-singleton case;
  they do not add another splitter, domain-validation engine or causal label.
- Result manifests and evaluation reports include the
  [observed package-source record](PACKAGE_SOURCE_IDENTITY.md). It records a
  fixed set of observed own-source bytes or unavailability, not loaded-code,
  build, hardware, reproducibility, authenticity or safety certification.
- The source distribution includes the related documentation and tests;
  the wheel installs the Python package and entry points.

These changes do not add model families, alter prediction/metric formulas,
change grouped selection or introduce dependencies. Existing binary and
regression artifact schema identities and output-file sets are retained;
package-source metadata is additive. Unknown metadata accepted by an older
byte verifier is not authenticated by that verifier.

## Local acceptance gate

Before any separate publication decision, build an sdist and a wheel from that
sdist; check source and archive contents; install the wheel in a fresh local
venv outside the source checkout; then run the full tests and real installed
CLI evaluation/prediction smoke checks. Verify package imports come from the
venv and that the 26 owned Python files match the candidate, including the
source-identity helper. Independent review must use the exact archived bytes.

The approved offline installation reuses existing dependencies and uses
`--no-index --no-deps`; it is a clean package installation, not fresh dependency
resolution, minimum-version testing or a cross-platform matrix. Optional test
skips and source-companion subprocesses must remain visible in the evidence.
No browser-pixel, mobile, assistive-technology or CI acceptance is implied.

Candidate artifacts and their hashes remain local. A version label, successful
build or these notes are not release authorization. Any later tag, Release or
asset upload requires separate owner approval of the exact candidate/artifacts.
