"""No-fit, one-step evaluation on an explicit frozen observed-history roster."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .data import MAX_ROWS
from .metrics import _checked_arithmetic, grouped_metrics


@dataclass
class ObservedHistoryResult:
    """In-memory scores and aligned predictions; no fitting or artifact writes."""

    summary: dict
    predictions: pd.DataFrame


def _table(frame, columns, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if list(frame.columns) != columns:
        raise ValueError(f"{name} requires exactly these ordered columns: {columns}.")
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"{name} must contain 1 to {MAX_ROWS} rows.")


def _numbers(values, name):
    # Reject strings, booleans and complex numbers instead of coercing labels.
    values = list(values)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in values):
        raise ValueError(f"{name} must contain real numeric values, not strings or booleans.")
    try:
        with np.errstate(over='raise', under='raise', invalid='raise'):
            result = np.asarray(values, dtype=np.float64)
    except (OverflowError, ValueError, FloatingPointError) as exc:
        raise ValueError(f"{name} cannot be represented in float64.") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite.")
    if any(v != 0 and f == 0 for v, f in zip(values, result)):
        raise ValueError(f"{name} contains a nonzero value that underflows float64.")
    if any(isinstance(v, Integral) and int(f) != int(v) for v, f in zip(values, result)):
        raise ValueError(f"{name} contains an integer not exactly representable in float64.")
    return result


def _keys(frame, name):
    traces = list(frame['trace_id'])
    if any(not isinstance(t, str) or not t or '\x00' in t for t in traces):
        raise ValueError(f"{name} trace_id must be nonempty strings without NUL.")
    indices = list(frame['target_index'])
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 2 for i in indices):
        raise ValueError(f"{name} target_index must be integer positions >= 2.")
    keys = list(zip(traces, map(int, indices)))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} contains duplicate origins.")
    return keys


def evaluate_observed_history(observations: pd.DataFrame, *, origins: pd.DataFrame,
                              saved_predictions: pd.DataFrame) -> ObservedHistoryResult:
    """Score one saved model and two baselines on exactly the same origins.

    observations: columns trace_id, timestamp, value. Input order within each
    trace defines zero-based positions and must have strictly increasing finite
    numeric timestamps. No sorting, interpolation or unit conversion occurs.
    origins: columns trace_id, target_index, in a caller-frozen evaluation order.
    saved_predictions: same keys in the same order, plus prediction. A missing,
    extra, duplicate or reordered origin fails rather than changing coverage.

    Each target at t uses observed values at t-1 and t-2 in its OWN trace. This
    is teacher-forced one-step scoring, not recursive/open-loop forecasting.
    The second baseline is 2*y[t-1]-y[t-2] in index space, not elapsed-time
    extrapolation. Strict timestamps do not prove upstream causal processing,
    saved-model training provenance or prospective availability of measurements.
    """
    _table(observations, ['trace_id', 'timestamp', 'value'], 'observations')
    _table(origins, ['trace_id', 'target_index'], 'origins')
    _table(saved_predictions, ['trace_id', 'target_index', 'prediction'], 'saved_predictions')
    keys = _keys(origins, 'origins')
    if _keys(saved_predictions, 'saved_predictions') != keys:
        raise ValueError('Saved predictions must match the exact ordered frozen origin roster.')
    traces = list(observations['trace_id'])
    if any(not isinstance(t, str) or not t or '\x00' in t for t in traces):
        raise ValueError('Observation trace_id must be nonempty strings without NUL.')
    times = _numbers(observations['timestamp'], 'timestamps')
    values = _numbers(observations['value'], 'observations')
    model = _numbers(saved_predictions['prediction'], 'saved predictions')
    positions = {}
    for i, trace in enumerate(traces):
        indices = positions.setdefault(trace, [])
        if indices and not times[i] > times[indices[-1]]:
            raise ValueError('Timestamps must be strictly increasing within each trace in input order.')
        indices.append(i)
    selected, previous, previous2, last = [], [], [], {}
    for trace, target in keys:
        if trace not in positions or target >= len(positions[trace]):
            raise ValueError('Origin references an unknown trace or out-of-range target.')
        if target <= last.get(trace, -1):
            raise ValueError('Origin positions must increase within each trace; no sorting is performed.')
        last[trace] = target
        indices = positions[trace]
        selected.append(indices[target])
        previous.append(indices[target - 1])
        previous2.append(indices[target - 2])
    observed = values[selected]
    persistence = values[previous]
    with _checked_arithmetic():
        continuation = 2.0 * persistence - values[previous2]
    if not np.isfinite(continuation).all():
        raise ValueError('Index continuation must be finite.')
    output = origins.reset_index(drop=True).copy()
    output['timestamp'] = times[selected]
    output['previous_index'] = [t - 1 for _, t in keys]
    output['previous2_index'] = [t - 2 for _, t in keys]
    output['observed'] = observed
    scores = {}
    for name, prediction in [('saved_model', model), ('persistence', persistence),
                             ('index_continuation', continuation)]:
        overall, per_trace = grouped_metrics(observed, prediction, [t for t, _ in keys])
        scores[name] = {'overall': overall, 'per_trace': per_trace}
        output[name] = prediction
    return ObservedHistoryResult(summary={
        'protocol': 'observed_history_one_step_v1',
        'evaluation_support': {
            'claim_scope': 'descriptive_observed_history',
            'n_scored_traces': len({trace for trace, _ in keys}),
            'counting_unit': 'distinct trace_id in the scored origin roster',
            'source_group_count': None,
            'source_group_status': 'not_declared',
            'physical_independence': 'not_established',
            'statistical_interpretation': 'descriptive_only',
            'support_reasons': ['source_groups_not_declared',
                                'physical_independence_not_established'] + (
                ['no_between_trace_replication']
                if len({trace for trace, _ in keys}) == 1 else []),
        },
        'n_origins': len(keys), 'coverage': 'exact_frozen_roster_no_drops',
        'weighting': 'overall row metrics and separately reported equal-trace group metrics',
        'models': scores, 'fit_calls': 0, 'selection_performed': False,
        'history': 'past observed values, including earlier scored targets once observed',
        'time_basis': 'index continuation; irregular timestamps allowed',
        'upstream_processing': 'unknown; timestamp order does not establish causal preprocessing',
        'prediction_provenance': 'caller supplied; training cutoff and prediction freeze not independently verified',
        'claim_limit': 'exported-sequence comparison only; no physical, causal or read-saving claim',
    }, predictions=output)
