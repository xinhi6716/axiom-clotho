# Frozen observed-history evaluation (no fit)

`groupbench.observed_history.evaluate_observed_history` is a separate Python API.
It does not change `GroupKFold`, which is group-disjoint, not temporal. It accepts
one already-saved model prediction per explicitly selected origin, constructs
two fixed baselines, and reuses the existing regression/grouped metric functions.
It does not train, tune, select, read files, write artifacts, infer units or sort.

```python
import pandas as pd
from groupbench.observed_history import evaluate_observed_history

# Original invented data, irregularly spaced numeric times in caller's units.
observations = pd.DataFrame({
    'trace_id': ['example'] * 4,
    'timestamp': [0, 1, 4, 9],
    'value': [2., 4., 6., 8.],
})
origins = pd.DataFrame({'trace_id': ['example', 'example'],
                        'target_index': [2, 3]})
saved_predictions = origins.assign(prediction=[5., 9.])
result = evaluate_observed_history(observations, origins=origins,
                                   saved_predictions=saved_predictions)
assert result.summary['n_origins'] == 2
assert result.predictions['index_continuation'].tolist() == [6., 8.]
```

## Exact coverage and timing contract

Use exactly the ordered columns shown above. Trace IDs are nonempty strings;
each trace is an explicitly identified sequence boundary, not evidence of an
independent physical device. Input order within each trace defines zero-based
positions. Numeric timestamps must be finite and strictly increasing per trace;
duplicate/decreasing times fail, not silently sort. Interleaved traces are allowed.
No datetime, string, boolean or complex numeric conversion is performed. Numeric
inputs use float64; integers that cannot be represented exactly and NumPy floating
scalars wider than float64 fail. Other accepted real values are converted to
float64 and may lose precision; this is not a complete precision audit. Precision
already lost in caller-created arrays cannot be recovered by this helper. Review
the resolution and units before explicitly converting wider inputs.

The caller must freeze the origin roster and model predictions before inspecting
their scoring outcomes. The helper verifies exact ordered key equality, uniqueness,
increasing positions per trace and valid positions >= 2, but cannot verify that
the caller actually froze them prospectively or respected a model training cutoff.
Every method is scored on every roster entry, with no missing-value filtering or
model-specific intersections. All input observations must be finite. Existing
20,000-row limits and fail-closed metric arithmetic apply. Overflow/underflow
raises, never drops a row or silently repairs the scores. No pass threshold,
winner selection, confidence interval or independent-sample count is supplied.

At target position t, persistence is y[t-1]; index continuation is
2*y[t-1]-y[t-2], both from the same trace. Caller-supplied current/future/cross-trace
history references are not accepted: histories are derived internally. The
continuation formula ignores elapsed-time gaps and is not a time-normalized slope.
Return rows preserve the roster and show both prior positions, target timestamp,
observed target, saved-model prediction and both baseline predictions. Overall
MAE/RMSE/R2/bias are row-weighted. Equal-trace `group_*` metrics and per-trace
scores are reported separately, with no claim of independent traces.

## Interpretation boundary

### Machine-readable statistical scope

The returned Python `result.summary` includes an additive `evaluation_support`
object. This API has no dedicated CLI or artifact writer; callers may serialize
the summary themselves. Existing prediction rows, metrics, baseline definitions,
coverage and provenance fields are unchanged.

- `claim_scope` is `descriptive_observed_history` and
  `statistical_interpretation` is `descriptive_only`.
- `n_scored_traces` counts distinct `trace_id` values in the scored origin roster,
  not observation rows, origins, or unscored traces. `counting_unit` states this.
- `source_group_count` is JSON `null` and `source_group_status` is `not_declared`:
  this input contract has no trace-to-source mapping. A trace count is not an
  independent-source count, and unknown does not mean zero.
- `physical_independence` is `not_established`. `support_reasons` records the
  undeclared source grouping and unestablished physical independence. A single
  scored trace also records `no_between_trace_replication`.

These fields describe limits of statistical interpretation, not whether a model
can predict. More traces do not automatically establish independent replication
or valid population inference. No universal minimum-group threshold, confidence
interval, significance test, or pass/fail rule is introduced. Descriptive scores
remain available even when independent-source support is unknown. Consumers that
enforce exact summary key sets must allow this new additive field; this does not
change the scoring protocol identity or promise an unchanged serialized document.

This is teacher-forced one-step evaluation using observed history, not recursive
multi-step forecasting. An earlier scored target may become a later origin's
observed input. Changing a target/future value cannot change that origin's two
baselines, but may legitimately change later histories and changes scores.
The saved model's features/training/fitting are outside this helper's verification.

Timestamps alone cannot establish whether exported values were smoothed using
future observations or otherwise processed upstream. Upstream causal availability
remains unknown. The comparison establishes only errors on the supplied export
and roster, not physical causality, prospective deployment, sensor read savings,
independent-device generalization or a resolution of unknown physical units.
No unit conversion is attempted. Different transformations or provenance require
separate evidence, not a stronger label from this helper.
