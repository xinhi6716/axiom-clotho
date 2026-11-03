# Optional grouped models: fixed synthetic study

**Result:** keep the kernel optional. It helped on the specified smooth IID mechanism, but failed on both extrapolation sides of the linear domain-holdout mechanism. The group-median baseline had mixed or negative results. Defaults were not changed in response.

Read [all findings](findings/FINDINGS.md), [all 360 metric records](findings/ALL_RESULTS.csv), and [the fixed method](METHODS.md). This is a small synthetic method comparison, not evidence of physical-device efficacy or universal model superiority.

## Inspect without fitting

From the repository root, on Linux with NumPy already available:

```bash
python examples/optional_grouped_study_v1/check_saved.py \
  --package examples/optional_grouped_study_v1 --output /tmp/optional-study-qa.json
```

The checker reads saved numeric inputs and predictions only: no model imports, random draws, network calls or fitting. `PUBLIC_PROVENANCE.json` inventories the public package. Checksums detect differences relative to this inventory; they are not signatures, authentic execution proofs or guarantees that a writer was honest.

- `artifacts/outputs/`: all 15 cohorts, raw predictions, full selection/fold metadata and summaries; original bytes retained.
- `artifacts/cases.json`: exact 45-fold train/test roster and 1,125-fit plan.
- `independent_qa.json`: original independent saved-output arithmetic results and every prespecified contrast.
- `archive/driver_r2_used.py`: exact used driver, including its disclosed monitoring defect. **Historical evidence, not a recommended operational runner.** Its original source-review receipt and private execution paths are deliberately not distributed; it cannot be treated as a turnkey portable launcher.
- `generator.py`: portable access to that exact generator, with frozen case mappings. Importing it draws no data and fits no models.

## Reproduction and identity

The fitted source was commit `5033a191e00667e7dbc53055a3bbbda5dc2ebc19`, tree `2653ee2f361a443a0f2a87fa42ae497066e3c736`; publication may be on a later commit. Match the 32 module hashes in `artifacts/source_hashes.json` and runtime versions in `artifacts/runtime.json` before interpreting exact numeric reproduction. Floating-point/platform differences can change results.

The saved inputs are sufficient to repeat fitting using the pinned source's `groupbench.benchmark`. The exact call for each saved case is:

```python
result = benchmark(frame, target="y", group="group", features=["x", "z"],
    baselines=("dummy", "ridge", "extra_trees", "nystroem_ridge", "group_median"),
    outer_splits=3, inner_splits=2, random_state=case["root"], objective="group_mae")
```

Load `data.npz` with `numpy.load(..., allow_pickle=False)` and create `frame` from `x`, `z`, `y`, `group`. Iterate the exact `cases.json` order. To regenerate rather than load inputs, `generator.generate_case(scenario_id, root)` uses the original generator and saved prospective fold mapping. **Both regeneration and fitting are new computation, not verification of saved results.** No new fits were used to prepare this public package. Full reproduction entails 1,125 top-level pipeline fits; use a separately reviewed resource supervisor rather than the historical R2 launcher. Never overwrite the supplied results.

## Resource qualification

The original run completed once: 15 datasets, 1,125 fits, zero fit failures. Terminal values were 15.623958 child CPU seconds, 15.683630268 wall seconds and 136640 KiB maximum RSS, below their numerical limits. However, an unavailable `/proc/.../children` read hit a broad exception handler and skipped live CPU/RSS, wall, retained-size and managed-I/O checks. Sampled RSS zero means **unobserved**, not zero memory. The topology assertion was policy, not verified evidence. Terminal checks and CPU limits remained, but this is **not a full operational-supervision pass**. Managed I/O excludes dependency-loader traffic.

Future supervisor repair is separate software work; no repaired supervisor ran this study, and no original receipts have been retroactively changed.
