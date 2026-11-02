# Explicit group-balanced median constant

`GroupBalancedMedianRegressor` is an opt-in, standalone Python estimator. It
predicts a lower weighted median of **all current training targets**, with each
training group receiving equal total weight and its rows sharing that weight.
It is **not** the median of group means or group medians.

Also available as explicit `group_median` in `GroupBenchRegressor`, benchmark,
fit_predict and supported measurement/FET workflow baseline lists. Existing
defaults and `dummy`'s unweighted training mean remain unchanged. The selector
routes only the current fit's training groups to this family; query rows need no
groups. See [extended family usage](OPTIONAL_REGRESSION_FAMILIES.md). The fixed
`package_source` inventory includes this module.
Its canonical source digest consequently changes. Historical saved manifests
with a 28-file source record, or no source record, remain readable: saved-bundle
verification does not compare their recorded bytes with the current installation.
This is backward-readable metadata, not proof that an old producer executed
correctly, authentication, or new-family support in an old verifier.

## Direct use

```python
import numpy as np
from groupbench.baselines import GroupBalancedMedianRegressor

X = np.ones((4, 1))
estimator = GroupBalancedMedianRegressor().fit(
    X, [0, 100, 101, 1], groups=["a", "a", "a", "b"]
)
assert estimator.constant_ == 1.0
assert estimator.predict(np.ones((2, 1))).tolist() == [1.0, 1.0]
```

These are invented arbitrary-unit values, not device measurements or a
performance result. The median of the two group medians would instead be 50.5.
One elementary comparator can better match an objective without being generally
better on held-out data, row MAE, RMSE, or other objectives.

## Exact weighting and tie rule

For current training group sizes `n_g`, the constant minimizes training loss

`L(c) = (1/G) * sum_g [(1/n_g) * sum_(i in g) abs(y_i - c)]`.

The implementation sorts the finite converted targets and uses integer weights
`L/n_g`, where `L` is the least common multiple of observed positive group sizes.
Python arbitrary-precision integers make the half-weight decision exact. It
returns the first sorted target reaching at least half of total integer weight;
an exact half selects the **lower observed target**. No floating-point tolerance,
interpolation, target averaging or outcome-dependent tie choice is involved.
Signed zero is represented as positive zero. The target remains a converted
binary64 value: this does not recover precision lost during input conversion.

No future group sizes enter the calculation. Repeating **every** row of one
training group equally leaves its weighted target distribution unchanged.
Repeating only selected rows need not. Neither this property nor adding weights
would make an unrelated preprocessing/tree pipeline duplication-invariant.

## Input and lifecycle contract

- `fit(X, y, *, groups=None, sample_weight=None)` requires one finite numeric
  target and one valid group ID for every supplied training row. Missing,
  masked, blank, NUL-containing, mixed text/numeric, Boolean, nonfinite, nested,
  or length-mismatched group inputs reject through the documented group checks.
  Actual masked target or feature entries also reject before array conversion,
  including retained masked scalars/rows nested inside lists, tuples, object
  arrays, or pandas object cells, and masks on CSR/CSC `data`, `indices`, or
  `indptr` buffers. Already-erased mask information cannot be
  reconstructed. The guard neither fills masks nor mutates supplied inputs.
- Numeric group ID `0` is valid; it is not a zero-weight group. Empty inputs
  reject. One observed group is valid for this constant; cross-validation
  group sufficiency is a separate upstream constraint. There is no predefined
  vocabulary of known group IDs, and IDs do not establish physical independence.
- Every non-`None` `sample_weight` rejects, including uniform or zero arrays.
  Combining caller weights would change or double-weight the declared objective.
- `X` is a finite real numeric 2D dense, CSR or CSC matrix with at least one
  feature and 1–20,000 rows per call. It normally comes from fold-local
  preprocessing. Sparse inputs are not densified. Other sparse formats,
  including COO, are explicitly rejected rather than implicitly converted;
  sparse conversion can otherwise discard retained mask metadata. The constant ignores its
  values, but validates shape and finiteness; raw missing/categorical values
  require preprocessing first. The row limit bounds the exact-integer work,
  not arbitrary input matrix memory or whole-platform resource use.
  Mask inspection rejects cyclic containers, nesting of 64 or more container
  edges, and more than 2,040,001 visited container/scalar entries per input.
  It uses an incremental stack, not recursion or a width-sized work list.
  Ordinary numeric ndarrays and sparse matrices are not expanded into Python
  scalar traversal; arbitrary iterables/generators are not consumed by the mask
  guard. These inspection bounds do not make arbitrary Python objects a sandbox.
- Named DataFrame columns must be distinct strings. After a named fit,
  prediction requires exactly the same names **and order**. There is no implicit
  reordering or unknown-column fallback. After an unnamed fit, feature count is
  checked. No unknown constructor or `set_params` options are supported.
- `predict(X)` takes no groups, targets or weights. New group identities are
  therefore neither inspected nor used at prediction time. It returns a finite
  constant for every valid query row; it does not selectively omit rows.
- Successful fitting stores only the constant, feature-schema/scalar counts and
  rule names, not training IDs, targets or row weights. Validation and computation
  complete before fitted state is published. A failed refit retains the prior
  complete estimator; a failed first fit remains unfitted.
- `score` is the inherited sklearn **row R-squared**, not group MAE. For the
  group-aware evaluation question use explicit group metrics and group-disjoint
  validation; do not reinterpret `.score()` as the fitting objective.

## Pipelines, cloning and persistence

With sklearn's ordinary parameter forwarding (metadata routing disabled), pass
the groups of the current training rows to the final step explicitly:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("impute", SimpleImputer()),
    ("scale", StandardScaler()),
    ("model", GroupBalancedMedianRegressor()),
])
pipeline.fit(X, [0, 100, 101, 1], model__groups=["a", "a", "a", "b"])
```

Do not pass `model__groups` to other estimator families that do not accept it.
The integrated grouped selector slices groups for every inner training fit and
the final family refit. This standalone module does not implement CV splitting,
metadata-routing-enabled dispatch or automatic `GridSearchCV` group delivery.
See the primary [Pipeline parameter-forwarding documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.pipeline.Pipeline.html)
and [estimator interface guide](https://scikit-learn.org/1.8/developers/develop.html).

`sklearn.base.clone` produces an unfitted estimator. A trusted, freshly generated
in-memory pickle roundtrip is software-tested on the recorded runtime, but no
public persistence/export format or model loader is added. **Never load an
untrusted pickle.** Cross-version serialization compatibility is not claimed.

## Verification scope

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_group_median_baseline.py -v
```

The tests use new anonymous software fixtures and an independent Fraction-based
oracle for the exact loss/median, including both median-of-medians counterexamples,
half-weight ties, extreme finite targets, replication/order invariance, sparse
inputs, invalid inputs, failed refits, explicit pipeline group forwarding, cloning,
trusted memory serialization, and the 20,000-row boundary. A 19,900-row case uses
199 different group sizes to exercise large exact denominators.

Executed runtime: Linux, Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0,
scikit-learn 1.8.0. The existing package dependency range is unchanged; this is
not verification of its complete version/platform matrix. No old reference
benchmark was rerun, no new scientific performance study was performed, and no
dataset, model weight, hosted API or dependency download is needed.
