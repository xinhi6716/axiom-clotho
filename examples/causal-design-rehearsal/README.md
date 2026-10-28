# Rehearse a causal design with invented data

[Clotho home](../../README.md) · [All examples](../README.md) · [繁體中文總覽](../../README.zh-TW.md)

Use a small invented causal model to see how adjustment choices change the question you answer. This simulation-only companion illustrates three distinctions:

1. A pre-setting batch context is not a post-setting intermediary.
2. A total intervention effect is not a mediator-adjusted coefficient.
3. A simulator can know intervention truth when observational adjustment fails.

The command generates observations, runs three fixed OLS plans, computes paired interventions and writes a readable comparison. It rejects an intermediary-adjusted request for a total effect, and returns no coefficient for a rank-deficient design. Estimators never receive latent variables or simulator truth.

This is not a real-data causal estimator, physical simulator or calibrated semiconductor model. It rehearses a design whose equations are already known.

## Run locally

Use an existing development environment with NumPy and threadpoolctl. The command accepts no real CSV and downloads nothing. Development checks used Python 3.12.14, NumPy 2.3.5 and threadpoolctl 3.6.0; other runtime combinations have not been verified. From the repository root:

```bash
mkdir -p runs
PYTHONDONTWRITEBYTECODE=1 python examples/causal-design-rehearsal/rehearsal.py \
  --output-dir runs/causal-design-rehearsal \
  --accept-simulation-only
cat runs/causal-design-rehearsal/INTERPRETATION.txt
```

The output directory must not exist. Default seed is 7, with 32 invented batches
and 16 devices per batch (512 labels per scenario). Optional `--seed`, `--batches`
and `--devices-per-batch` change only sampling, not equations, roles, estimands,
analysis plans or intervention levels. Bounds are 2–128 batches, 2–64 devices per
batch and an unsigned 32-bit seed. Every invocation retains all three scenarios
and all three plans. No seed is redrawn to make an adjusted estimate look better.
Small allowed counts are software bounds, not sufficient sample sizes.
Same-seed bytes reproduce in the tested runtime; bitwise equality across NumPy,
linear-algebra libraries or platforms is not promised. Preserve the generated
observations and runtime record when comparing runs.

All computation and serialization precede output creation. New directories use
mode 0700 and new files mode 0600 where POSIX modes apply. Existing directories,
files and live/dangling output symlinks are rejected; parents are not created by
the Python command. Each file is capped at 8 MiB. Exclusive writes prevent normal
clobbering, but a disk/write failure can leave a partial directory: only the
last-written `COMPLETE.json` denotes a completed invocation. It does **not** denote
successful causal identification. No overwrite or resume mode is provided.
Use trusted local storage and do not edit sources during execution. This is not
an atomic directory transaction or a hostile concurrent-filesystem sandbox.

## The invented model

Every numeric variable is a **dimensionless illustrative index**:

- C / `batch_context`: measured before the setting, shared within a batch.
- A / `process_setting`: a fictional adjustable **device-level** setting.
- M / `intermediary`: observed after A and before the readout.
- Y / `readout`: an invented device readout.
- U: a latent device variable used only in the omitted-confounder scenario.

The analogy must not be read as a shared batch anneal that can be independently
changed for each device. Real intervention units, process coupling and interference
need their own model. No synthetic I–V measurements or earlier invented endpoint
fixtures are repurposed as causal outcomes here.

The base equations are:

```text
C = UC                  (one shared draw per batch)
A = C + UA              (device-level setting noise)
M = 2*A + C + UM
Y = A + 3*M + 2*C + UY
```

UC, UA, UM, UY and U are mutually independent standard normal exogenous draws;
the latter four are device-level. U is unused in the base scenario. All three
scenarios reuse the same draws, modifying only their declared equations. The
coefficients 1, 2 and 3 are arbitrary teaching constants, **not measured physical
parameters**. The graph, equations, variable timing, seed and counts are saved
in `SCM.json` on every run.

### Three prespecified observational plans

- `naive_association`: regress Y on intercept and A. Its A coefficient is an
  observational association, not the declared total effect.
- `context_adjusted`: regress Y on intercept, A and C, requesting a total effect.
- `intermediary_adjusted`: regress Y on intercept, A, C and M, requesting a
  controlled direct effect under the exact base SCM. Relative to a total-effect
  question, conditioning on M answers a different question.

These are ordinary least-squares coefficients, not regularized GroupBench
predictions, held-out accuracy scores or competing methods selected by test error.
No p-values, confidence intervals, bootstrap, model selection or automatic winner
are produced. Finite Gaussian samples need not match population coefficients, and
an adjusted estimate is not promised to be closer for every seed.

Substituting M gives `Y = 7*A + 5*C + 3*UM + UY`. Under the **base** unit-variance
SCM, `Cov(A,C)/Var(A) = 1/2`, so the population naive coefficient is
`7 + 5/2 = 9.5`. The C-adjusted coefficient is 7 and the C/M-adjusted coefficient
is 1. These are independently derivable properties of the stipulated model,
not fitted empirical truth. Tests separately use orthogonal ±1 arithmetic
fixtures to check these targets; those fixtures are not described as Gaussian draws.

### Paired intervention oracle, kept separate

For each invented device, `do(A=0)` and `do(A=1)` replace the A equation while
preserving C, UM, UY and U. **M is recomputed** in each arm. Thus ΔM=2 and ΔY=7.
Holding observed M fixed would silently change the question and is tested as an
error. A second, explicitly different oracle sets both A and M: the contrast
between `do(A=1,M=0)` and `do(A=0,M=0)` is 1 under these equations.

The intervention outputs never enter OLS. They are not experimentally measured
counterfactuals. The 7/1 contrasts remain properties of the declared equations in
each fixed scenario because the added confounder or assignment restriction does
not change the A→M→Y and A→Y coefficients. That fact does not make them identifiable
from each scenario's observational data. Runtime checks reject disagreement with
this declared algebra beyond a 1e-9 absolute floating-point tolerance.

### Two failures retained in the same run

**Omitted device confounding:** replace `A = C + UA` with `A = C + UA + U` and add
`2*U` to Y. U is not included in observation CSVs or regression inputs. C adjustment
no longer blocks the unobserved path; neither adjusted coefficient is identified
as its requested causal quantity. The report explicitly labels them descriptive,
even if one happens to be close to the known simulator contrast. Identifiability
fails because of the stipulated omitted path, not because a statistical test
compared an estimate to 7.

**Deterministic assignment:** set `A = C` exactly. Both requested adjusted designs
have aliased columns and return null coefficients. Treatment has no variation
conditional on the required C, even though the simulator still defines the
intervention oracle. This is **not** the claim that any absence of within-batch
variation prevents identification: independently varying batch-level treatment
could be a different valid design. Here the failure is specifically A=C.

## Numerical rank is not causal positivity

The unscaled OLS design includes an intercept. SVD declares rank using singular
values strictly greater than `1e-12 * largest_singular_value`. The report saves
all singular values, the relative and absolute cutoffs, required/observed rank,
and the full-rank condition number. A rank-deficient plan does not call the
least-squares solver or publish a minimum-norm coefficient.

This is an explicit **numerical** tolerance, not a test that assignment noise is
exactly zero. Near-collinearity with nonzero setting variation can also be
rejected at this tolerance. The exact zero-noise statement in the deterministic
scenario comes from its generator equation, not from that numerical result.
Likewise, full rank does not establish positivity, exchangeability, adequate
sample size or stable precision. Conditional Gaussian support is a premise of
the noisy SCM, not an empirical certification.

## Read the outputs and run checks

Each run saves:

- Three `*.observations.csv` files: exact observed roles and invented identities.
  They never include U, latent errors or paired intervention values.
- Three `*.paired_interventions.csv` files: separate paired simulator outputs.
- `SCM.json`: explicit equations, graph, roles, noise assumptions and oracle origin.
- `REPORT.json`: every coefficient or numerical refusal, diagnostics, counts,
  population references where appropriate and assumptions/limits.
- `INTERPRETATION.txt`: readable comparison without empirical or causal validation.
- `COMPLETE.json`: source-byte hash and hashes of the nine other files. These
  identify bytes, not authenticity, scientific validity or historical preregistration.

The functions in this source example are internal rehearsal/test helpers, **not
new validated real-data APIs**. The observed-table helper enforces exact roles,
unique device rows and shared C within each batch, but cannot verify physical
identities, timing, independence, omitted confounding or experimental provenance.
It accepts only the three fixed role/estimand plans. Keep real data and private
research results out of these public examples and run directories.

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -p 'test_causal_design_rehearsal.py' -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

The existing cross-repository walkthrough tests additionally require the exported
`SEMICONDUCTOR_TOOLBOX_SOURCE` variable described in that example; they explicitly
skip without it. The new causal-design tests require no other repository.

## Before using these ideas in research

This tool supplies invented ground truth so design mistakes can be caught locally.
It cannot supply the real intervention definition, causal graph, exposure timing,
absence of unmeasured confounding, positivity, consistency, no interference,
measurement validity or dependence-aware uncertainty. Those require independent
scientific justification. Batch labels, prediction gains and regression adjustment
alone do not make them true.

For background on the distinction between direct and total effects, see
[Pearl, *Direct and Indirect Effects*](https://ftp.cs.ucla.edu/pub/stat_ser/r273-jsm05.pdf).
For identification and positivity conditions, see
[Hernán and Robins, *Causal Inference: What If*, Chapters 2–3](https://miguelhernan.org/whatifbook).
These references motivate the distinction; they do not validate these invented
coefficients or any semiconductor application. No reference code or dataset is
copied into this companion.
