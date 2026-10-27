# Installed-wheel portability smoke

This is a **manual, synthetic smoke check**, not the complete regression suite,
a release acceptance gate, or scientific validation. Configuration is not a
successful run. Native Windows remains unverified until the exact workflow
revision has a successful Windows job; link that run before changing any support
claim. It does not change frozen release tags or release assets.

CI builds the authorized **public source checkout**. It does not download b1/b2
release assets, verify remote release bytes, or override any earlier tool-policy
block on a release-asset URL. This is separate source-build evidence, not closure
of a release-asset download/verification blocker.

## Scope and limits

- Builds a new wheel from the dispatched checkout; installs it in a fresh venv;
  invokes it outside the checkout. No editable install or source `PYTHONPATH`.
- Tests all four installed CLI version commands and pinned component integrity.
- Reuses only `examples/measurement-regression/fixtures`: 18 invented devices,
  12 training and 6 prediction, dummy/ridge candidates. A copied config tightens
  `max_fits` to 40; plan must report exactly 40 planned fits and zero performed.
- Runs `prepare`, read-only `plan`, bounded `run`, and separate `score`. Outcomes
  are withheld from the working fixture until prediction finishes. Saved-score
  results must report six devices and zero fits. A second score invocation under
  `-S`, with ML imports forbidden, must produce byte-identical scoring outputs.
- Checks portable measurement receipts on both operating systems, including
  nested slash inventory keys. Paths include spaces to exercise argument handling.
- On Linux, the existing `mlcalc verify` must pass for comparison/prediction.
  On Windows, it **must reject** with exit 2 and the POSIX-unsupported diagnostic;
  package-source observations must say `unavailable/unsupported_source_access`.
  No weaker replacement or mocked Windows success is substituted.
- Does not run the full suite, install/upgrade/uninstall matrix, macOS/ARM/GPU,
  every declared Python/library version, minimum dependency bounds, private data,
  real scientific datasets, model weights, or an external ML service.

The four jobs cover standard x64 `ubuntu-24.04` and `windows-2022`, each with
Python 3.10 (declared minimum) and 3.12 (the previously locally verified line).
They are deliberately not a claim that all Python >=3.10 environments work.
Builds require `setuptools>=77.0.3`; runtime dependencies keep the existing bounds
in `pyproject.toml`. CI resolves compatible binary wheels from public PyPI and
prints the actual Python/library versions. Those dependencies and runner images
are not lockfile-pinned, so later reruns may differ. No new project dependency is
introduced. Automatic Git line-ending conversion is disabled before checkout to
preserve pinned vendor bytes; vendor hashes must never be regenerated just to
make a platform pass.

## Trigger, permissions and resource bounds

The workflow has only `workflow_dispatch`: no push/PR/schedule trigger. Publishing
the file does not start a run. Review the exact revision and authorize dispatch
separately. There are no custom inputs, secrets, PATs, OIDC, environment approvals,
deployments, releases, or artifact/cache uploads. The normal ephemeral GitHub
token has only `contents: read`; checkout does not persist credentials. Both
official actions are pinned to full commit IDs. No external account is needed.

Only `xinhi6716/axiom-clotho` with a literal JSON `false` repository-private flag
may run the jobs; a missing/null flag does not pass the guard.
One workflow run is active at a time, with at most two jobs in parallel, four jobs
total and a 15-minute timeout per job (at most 60 runner-minutes per dispatch,
excluding platform scheduling/cleanup). Subprocess timeouts and one-thread BLAS
limits further bound the small fixture. A newer manual dispatch cancels the older
run. Public logs record synthetic status/version evidence, not uploaded artifacts.

GitHub's official billing documentation checked on 2026-10-27 says standard
GitHub-hosted runners are free for public repositories; larger runners are always
charged. These exact Ubuntu/Windows labels appear in its public standard-runner
table. This is conditional on current GitHub policy and public visibility, not a
perpetual no-cost guarantee. No paid/larger runner or budget change is authorized.
No action enables spending or adds payment details. If visibility, runner labels,
pricing, trigger scope or storage behavior changes, stop and review first.

GitHub's standard limits currently include a six-hour job ceiling and 20
concurrent standard jobs for a Free account. This workflow's own limits are much
smaller; availability/organization policy can still prevent or queue execution.

Sources:
- [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
- [Public standard hosted runner labels](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#standard-github-hosted-runners-for-public-repositories)
- [Actions limits](https://docs.github.com/en/actions/reference/limits)
- [Official checkout v7.0.1 commit](https://github.com/actions/checkout/commit/3d3c42e5aac5ba805825da76410c181273ba90b1)
- [Official setup-python v7.1.0 commit](https://github.com/actions/setup-python/commit/6d4fd22b20465ccc0f7354476588e8df7c9ca0e0)

The action tags were resolved through their official repository tag APIs. Both
use Node 24 and require hosted runner software compatible with that runtime; this
is a hosted-runner configuration, not support for old self-hosted runners.
