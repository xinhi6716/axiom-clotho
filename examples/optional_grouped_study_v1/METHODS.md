# Fixed method, published after the sealed run

This is a sanitized method description, not the original prospective document. Original prospective revision-2 SHA-256: `708227ab3f634ffaed0ab5179e9189e4ea4f6019ea7e8c8e8eb159adcf41556a`. It omits internal review coordination and unrelated study references; it does not alter the experiment.

## Cohorts and draw order

Scenario-major, root-minor order: roots 731903, 731927, 731939, 731951, 731967; scenarios `linear_iid`, `smooth_iid`, `linear_domain_holdout`. Each cohort has groups g00…g17 with sizes [2,3,4] repeated six times: 54 rows. Features are x,z only. No group label, fold, seed, latent intercept or group size is a feature.

One `Generator(PCG64(SeedSequence([root, scenario_id])))` per dataset advances across groups in numeric order, never restarted per group. For each group draw scalar b ~ Normal(0,0.2), vector x, vector z ~ Uniform(-1,1), vector epsilon ~ Normal(0,0.15), in that order. y=f(x,z)+b+epsilon; ordinary float64, no clipping, rounding, rejection or resampling.

- Linear IID: x ~ Uniform(-1,1); f=1.5+2x+0.5z.
- Smooth IID: x ~ Uniform(-1,1); f=1.5+2sin(pi*x)+0.5z².
- Linear domain holdout: same linear f. Preassigned outer fold 0 gives x ~ Uniform(-2,-1), fold 1 Uniform(-0.5,0.5), fold 2 Uniform(1,2).

The package's `GroupKFold(3, random_state=root)` determines group assignments before outcome draws. Fold 0 is left extrapolation, fold 1 interpolation and fold 2 right extrapolation. This intentionally domain-coupled design is not ordinary exchangeable group CV, three extrapolation folds, or a generic OOD benchmark. Exact assignments are saved in `artifacts/cases.json`.

## Models and leakage boundary

Outer 3 / inner 2 group-disjoint folds. Inner selection minimizes equal-group MAE. Exact ties retain declared candidate order. Models: dummy mean; ridge alpha 0.1/1/10; ExtraTrees depth 6/None, 64 trees, minimum leaf 2; Nyström ridge alpha 0.1/1/10, rank cap 64, gamma inverse transformed training width; group-balanced lower median. Source and runtime pins are retained. Native computation threads were one.

Preprocessing, landmarks, model fits and inner selection use current training partitions only. Kernel training/landmark sampling is row-weighted; selecting with group MAE does not make training group-balanced. The median uses equal training-group then equal-row weights and lower observed target tie breaking. Median-versus-dummy changes the location rule and weighting together, not weighting alone.

Each dataset has 60 inner fits and 15 family refits. Selected predictions reuse the chosen family refit without an extra fit: 75 fits per cohort, 1,125 total. All five families and selected output are saved. No full-data refit, retuning or repeated outer evaluation was added.

## Outcomes and limits

Primary: equal-group MAE. Secondary: row MAE; retain RMSE and worst-group MAE. Every model, seed and fold is retained, including negative results. Prespecified group-MAE contrasts: kernel−ridge, kernel−ExtraTrees, median−dummy, selected−ridge. Negative is better; compare unrounded values, report all pairs and improvement/tie/harm counts. Five-root means and ranges are descriptive, not confidence intervals. Domain folds are reported separately and as an explicitly mixed pooled cohort. Rows and overlapping-training folds are not independent replicates.

All 15 outputs were sealed before operator comparison. The benchmark computed metrics internally, so this was operational noninspection, not cryptographic blinding. Independent QA recomputed saved metrics and selected-family consistency without fits. No p-values, pass threshold, seed exclusion, favorable-metric substitution, general superiority claim or real-device claim was specified.

Original budget: 300 CPU seconds total, main 200, independent QA 50, control 50; wall 900 seconds, RSS 512 MiB, retained artifacts 16 MiB, managed I/O 128 MiB. Preliminary deterministic timing controls (not study-seed draws) measured 2.069169 and 1.854570 child CPU seconds for 75-fit software fixtures; both were retained internally. These were estimates, not runtime guarantees. See README for the confirmed live monitoring defect and terminal values; the public package does not claim full enforcement.
