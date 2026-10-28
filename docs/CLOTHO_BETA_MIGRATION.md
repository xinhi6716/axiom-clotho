# Clotho beta version and installation identity

## Beta identity

| Identity | Chosen value |
| --- | --- |
| Product | **Axiom Clotho** (Moirai family) |
| Canonical repository | [`axiom-clotho`](https://github.com/xinhi6716/axiom-clotho), renamed in place from the calculator repository |
| Distribution name for this beta | **`ml-super-calculator`**, intentionally retained |
| Current beta version | **`0.3.0b7`** |
| Primary additive CLI | `clotho` |
| Preserved interfaces | `mlcalc`, `groupbench`, `mlcalc-workflow`, `import groupbench` |
| Release route | GitHub-only prerelease; no PyPI upload |

The repository rename is verified (same repository ID `1431307608`); beta
publication is still a separate action and is not implied by the name. Frozen `v0.3.0b2`, `v0.3.0b1`, `v0.2.0a4` and earlier tags, source archives
and binaries must remain untouched. New beta assets must use their own version,
not replace an older download under an old filename.

The retained distribution name lets one distribution continue owning the existing
`groupbench` package and console scripts. Renaming the distribution immediately
would introduce a second owner for overlapping files. Product/repository names
and import/distribution identities need not change together; this is a deliberate
compatibility boundary, not an additional product or duplicate repository.

## User-controlled upgrade after an accepted beta is published

These are instructions, **not actions performed by the workflow**. First preserve
the prior environment description and known-good wheel for rollback; prefer a
separate test virtual environment if the current one is used for research work.
Download only the intended, accepted GitHub release asset and compare its SHA-256
against that release's verified receipt.

For an environment already containing the existing compatible dependencies:

```bash
python -m pip show ml-super-calculator
python -m pip check

# Local reviewed beta wheel; no index access or dependency downloads:
python -m pip install --no-index --no-deps --upgrade ./ml_super_calculator-0.3.0b7-py3-none-any.whl

python -m pip check
clotho --version
mlcalc --version
groupbench --version
python -c "import groupbench; from importlib.metadata import version; print(groupbench.__version__, version('ml-super-calculator'))"
clotho components
```

The version checks should agree on `0.3.0b7`. `--no-deps` is intentional only when
the already-provisioned environment satisfies the unchanged declared requirements;
it does not install missing dependencies. If checks fail, stop and provision or
repair a separate environment deliberately rather than allowing an unexpected
download. The official [pip install documentation](https://pip.pypa.io/en/stable/cli/pip_install/)
describes local archives, upgrades and these index/dependency controls.

No manual uninstall is required merely because the repository/product is renamed:
this beta keeps the same distribution identity. Do not run these commands against
an unreviewed private test build; use the intended release and its own checksums.

## If a later release changes the distribution name

Do **not** co-install `axiom-clotho` and `ml-super-calculator` if both own
`groupbench` and the same console scripts. Prefer a fresh environment containing
only the chosen distribution. An explicitly authorized in-place name migration
would require preserving the old environment, removing the old distribution
**before** installing the renamed one, then verifying every import and CLI. Do
not uninstall the old overlapping distribution after the new one has already
been installed: overlapping recorded files can make the environment ambiguous.
The [pip uninstall documentation](https://pip.pypa.io/en/stable/cli/pip_uninstall/)
also lists older installation forms that cannot be cleanly removed automatically.

That distribution rename is **not part of this beta**. No renamed distribution,
PyPI project, package-index credential or account permission was created here.

## What was verified, and what was not

For b7, see the [release guide](RELEASE_NOTES_0.3.0b7.md). It packages the Clotho celestial artwork, report presentation and documentation refresh without changing the distribution name, model defaults, splits or metric formulas. The finite-tree-state guard from frozen [b6](RELEASE_NOTES_0.3.0b6.md) is retained unchanged. Frozen [b5](RELEASE_NOTES_0.3.0b5.md) contains the earlier visual refresh but not that guard. The [historical b4 guide](RELEASE_NOTES_0.3.0b4.md) covers blank-input `init`, which b7 retains. The [historical b3 guide](RELEASE_NOTES_0.3.0b3.md) covers FET v2; frozen b3 has no `init`. The [historical b2 guide](RELEASE_NOTES_0.3.0b2.md) describes its portable inventory fix. Native Windows, Python 3.10 and a minimum-dependency matrix remain unverified.

### Historical b1 verification

An isolated `0.3.0b1` source probe passed the complete 545-test suite with the
optional legacy integration cases enabled. Its wheel metadata retained the
existing import/entry-point owners, and extracted-wheel execution exercised the
new CSV workflow without installing it. The final candidate additionally tests
agreement among distribution, calculator and Clotho workflow version identities.

An actual installability check then created a disposable owncloud virtual
environment using already-installed dependencies (`--system-site-packages`) and
installed **only this owned wheel**, with `--no-index --no-deps`. Its generated
`clotho`, `mlcalc`, `groupbench` and `mlcalc-workflow` scripts worked outside the
checkout; the complete CSV reproduction and 29 focused integration tests passed.
No global environment or third-party package was changed or downloaded.

**Uninstall, in-place upgrade and fresh dependency resolution were not run.**
Alternate operating systems/Python versions and minimum dependency versions are
also unverified. This is source, extracted-wheel and isolated owned-wheel
installation evidence, not a claim that every existing user environment has
been migrated successfully.
