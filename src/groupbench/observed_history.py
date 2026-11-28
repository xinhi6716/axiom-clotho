"""No-fit evaluation on frozen observed-history rosters and declared cutoffs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Rational, Real

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


def _numbers(values, name, *, exact_clock=False):
    # NumPy timedelta64 can register as Real/Integral but must not be
    # reinterpreted as a clock, target or model score in implicit units.
    values = list(values)
    if any(isinstance(v, (bool, np.bool_, np.timedelta64)) or not isinstance(v, Real) for v in values):
        raise ValueError(f"{name} must contain real numeric values, not strings, booleans or timedeltas.")
    # Do not let conversion erase distinctions in NumPy extended-precision
    # inputs before the metric validator sees them.  Ordinary Python floats and
    # NumPy floating scalars no wider than float64 retain their prior behavior.
    if any(isinstance(v, np.floating) and v.dtype.itemsize > np.dtype(np.float64).itemsize
           for v in values):
        raise ValueError(f"{name} contains floating values wider than float64; explicitly convert after reviewing precision.")
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
    if exact_clock:
        for v, f in zip(values, result):
            if isinstance(v, Rational) and not isinstance(v, Integral):
                if Fraction(v.numerator, v.denominator) != Fraction.from_float(float(f)):
                    raise ValueError(f"{name} clock rational must be exactly representable in float64.")
    return result


def _integer_ticks(values):
    ticks = list(values)
    if any(isinstance(t, (bool, np.bool_, np.timedelta64)) or not isinstance(t, Integral)
           or not -(1 << 63) <= int(t) < (1 << 63) for t in ticks):
        raise ValueError('Integer timestamps must be signed-64-bit integer ticks, not floats, booleans or timedeltas.')
    return np.asarray(ticks, dtype=np.int64)


def _keys(frame, name):
    traces = list(frame['trace_id'])
    if any(not isinstance(t, str) or not t or '\x00' in t for t in traces):
        raise ValueError(f"{name} trace_id must be nonempty strings without NUL.")
    indices = list(frame['target_index'])
    if any(isinstance(i, (bool, np.bool_, np.timedelta64)) or not isinstance(i, Integral) or i < 2 for i in indices):
        raise ValueError(f"{name} target_index must be integer positions >= 2.")
    keys = list(zip(traces, map(int, indices)))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} contains duplicate origins.")
    return keys


def _elapsed_continuation(times, previous, previous2, schedule,
                          values, *, integer_ticks):
    clock = int if integer_ticks else lambda t: Fraction.from_float(float(t))
    # Convert before differencing to avoid int64 wrap or float64 span overflow.
    out = []
    for future, newer, older in zip(schedule, previous, previous2):
        t0, t1, t2 = clock(times[older]), clock(times[newer]), clock(future)
        ratio = Fraction(t2 - t1, t1 - t0)
        y0 = Fraction.from_float(float(values[older]))
        y1 = Fraction.from_float(float(values[newer]))
        exact = y1 + ratio * (y1 - y0)
        try:
            forecast = float(exact)
        except OverflowError as exc:
            raise ValueError("Elapsed-time continuation exceeds float64 range.") from exc
        if forecast == 0.0 and exact:
            raise ValueError("Nonzero elapsed-time continuation underflows float64.")
        out.append(forecast)
    return np.asarray(out, dtype=np.float64)


def _cutoff_continuation(current, older, leads):
    result = []
    for x, y, n in zip(current, older, leads):
        a, b = Fraction.from_float(float(x)), Fraction.from_float(float(y))
        exact = a + n * (a - b)
        try:
            out = float(exact)
        except OverflowError as exc:
            raise ValueError("Index continuation exceeds float64 range.") from exc
        if out == 0.0 and exact:
            raise ValueError("Nonzero index continuation underflows float64.")
        result.append(out)
    return np.asarray(result, dtype=np.float64)


def evaluate_observed_history(observations: pd.DataFrame, *, origins: pd.DataFrame,
                              saved_predictions: pd.DataFrame,
                              timestamp_mode: str = 'float64',
                              scheduled_origins: pd.DataFrame | None = None,
                              history_cutoffs: pd.DataFrame | None = None) -> ObservedHistoryResult:
    """Score one saved model and observed-history baselines on the same targets.

    observations: columns trace_id, timestamp, value. Input order within each
    trace defines zero-based positions and must have strictly increasing finite
    numeric timestamps. Use timestamp_mode='integer_ticks' for exact signed
    64-bit integer timestamps. Timedeltas must be converted explicitly by
    the caller before admission. Nonintegral rational clocks must already be
    exactly representable in float64. No sorting, interpolation or conversion
    of physical time units occurs.
    origins: columns trace_id, target_index, in a caller-frozen evaluation order.
    saved_predictions: same keys in the same order, plus prediction. A missing,
    extra, duplicate or reordered origin fails rather than changing coverage.

    Each target at t uses observed values at t-1 and t-2 in its OWN trace. This
    is teacher-forced one-step scoring, not recursive/open-loop forecasting.
    The second baseline is 2*y[t-1]-y[t-2] in index space, not elapsed-time
    extrapolation. Optionally provide scheduled_origins: columns trace_id,
    target_index, timestamp, with the same ordered frozen roster and exact
    realized clock values. This enables an elapsed-time linear baseline using
    the *supplied* scheduled target clock. A schedule mismatch fails the entire
    evaluation, never subset-selects. The caller must freeze the schedule before
    forecasting; matching clock values cannot verify that provenance.
    Optional history_cutoffs must contain trace_id, target_index, history_index
    with the same frozen roster and order. A history_index of at least 1 and
    strictly before its target identifies the latest admissible read. The
    baselines then use that read and its predecessor, without refreshing on
    intervening observations; linear continuation uses the declared lead.
    Assign the same cutoff to multiple targets for an open-loop forecast and
    independently bind saved model predictions to that information set.
    The caller must establish that the cutoffs and predictions were frozen.
    Strict timestamps do not prove upstream causal processing,
    saved-model training provenance or prospective availability of measurements.
    """
    _table(observations, ['trace_id', 'timestamp', 'value'], 'observations')
    _table(origins, ['trace_id', 'target_index'], 'origins')
    _table(saved_predictions, ['trace_id', 'target_index', 'prediction'], 'saved_predictions')
    keys = _keys(origins, 'origins')
    if _keys(saved_predictions, 'saved_predictions') != keys:
        raise ValueError('Saved predictions must match the exact ordered frozen origin roster.')
    if history_cutoffs is None:
        cutoffs = [target - 1 for _, target in keys]
    else:
        _table(history_cutoffs, ['trace_id', 'target_index', 'history_index'], 'history_cutoffs')
        if _keys(history_cutoffs, 'history_cutoffs') != keys:
            raise ValueError('History cutoffs must match the exact ordered frozen origin roster.')
        cutoffs = []
        for (_, target), value in zip(keys, history_cutoffs['history_index']):
            if (isinstance(value, (bool, np.bool_, np.timedelta64))
                    or not isinstance(value, Integral) or not 1 <= value < target):
                raise ValueError('History indices must be integers from 1 to target_index-1.')
            cutoffs.append(int(value))
    leads = [target - h for (_, target), h in zip(keys, cutoffs)]
    traces = list(observations['trace_id'])
    if any(not isinstance(t, str) or not t or '\x00' in t for t in traces):
        raise ValueError('Observation trace_id must be nonempty strings without NUL.')
    if not isinstance(timestamp_mode, str) or timestamp_mode not in ('float64', 'integer_ticks'):
        raise ValueError("timestamp_mode must be 'float64' or 'integer_ticks'.")
    times = (_integer_ticks(observations['timestamp']) if timestamp_mode == 'integer_ticks'
             else _numbers(observations['timestamp'], 'timestamps', exact_clock=True))
    values = _numbers(observations['value'], 'observations')
    model = _numbers(saved_predictions['prediction'], 'saved predictions')
    positions = {}
    for i, trace in enumerate(traces):
        indices = positions.setdefault(trace, [])
        if indices and not times[i] > times[indices[-1]]:
            raise ValueError('Timestamps must be strictly increasing within each trace in input order.')
        indices.append(i)
    selected, previous, previous2, last = [], [], [], {}
    for (trace, target), cutoff in zip(keys, cutoffs):
        if trace not in positions or target >= len(positions[trace]):
            raise ValueError('Origin references an unknown trace or out-of-range target.')
        if target <= last.get(trace, -1):
            raise ValueError('Origin positions must increase within each trace; no sorting is performed.')
        last[trace] = target
        indices = positions[trace]
        selected.append(indices[target])
        previous.append(indices[cutoff])
        previous2.append(indices[cutoff - 1])
    elapsed = None
    if scheduled_origins is not None:
        _table(scheduled_origins, ['trace_id', 'target_index', 'timestamp'], 'scheduled_origins')
        if _keys(scheduled_origins, 'scheduled_origins') != keys:
            raise ValueError('Scheduled origins must match the exact ordered frozen origin roster.')
        schedule = (_integer_ticks(scheduled_origins['timestamp']) if timestamp_mode == 'integer_ticks'
                    else _numbers(scheduled_origins['timestamp'], 'scheduled timestamps', exact_clock=True))
        if not np.array_equal(schedule, times[selected]):
            raise ValueError('Scheduled timestamps must exactly match the realized target clocks.')
        elapsed = _elapsed_continuation(times, previous, previous2, schedule, values,
                                        integer_ticks=timestamp_mode == 'integer_ticks')
    observed = values[selected]
    persistence = values[previous]
    older = values[previous2]
    if history_cutoffs is not None:
        continuation = _cutoff_continuation(persistence, older, leads)
    else:
        large = np.abs(persistence) > np.finfo(np.float64).max / 2.0
        with _checked_arithmetic():
            if np.any(large):
                continuation = np.empty_like(persistence)
                ordinary = ~large
                continuation[ordinary] = 2.0 * persistence[ordinary] - older[ordinary]
                # Halve the older value before subtracting when doubling persistence
                # alone would overflow. A sufficiently large same-sign older value
                # halves exactly; genuinely out-of-range results still fail.
                continuation[large] = 2.0 * (persistence[large] - 0.5 * older[large])
            else:
                continuation = 2.0 * persistence - older
    if not np.isfinite(continuation).all():
        raise ValueError('Index continuation must be finite.')
    output = origins.reset_index(drop=True).copy()
    output['timestamp'] = times[selected]
    output['previous_index'] = cutoffs
    output['previous2_index'] = [h - 1 for h in cutoffs]
    if history_cutoffs is not None:
        output['lead'] = leads
    output['observed'] = observed
    scores = {}
    forecasts = [('saved_model', model), ('persistence', persistence),
                 ('index_continuation', continuation)]
    if elapsed is not None:
        forecasts.append(('elapsed_continuation', elapsed))
    for name, prediction in forecasts:
        overall, per_trace = grouped_metrics(observed, prediction, [t for t, _ in keys])
        scores[name] = {'overall': overall, 'per_trace': per_trace}
        output[name] = prediction
    return ObservedHistoryResult(summary={
        'protocol': ('observed_history_one_step_v1' if history_cutoffs is None
                     else 'observed_history_declared_cutoff_v1'),
        **({'timestamp_mode': timestamp_mode} if timestamp_mode == 'integer_ticks' else {}),
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
        'history': ('past observed values, including earlier scored targets once observed'
                    if history_cutoffs is None else 'caller-declared cutoff; no intervening observations used by baselines'),
        **({'cutoff_provenance': 'caller supplied; saved-model information set not independently verified'}
           if history_cutoffs is not None else {}),
        'time_basis': ('index and scheduled elapsed-time continuation; irregular timestamps allowed'
                       if elapsed is not None else 'index continuation; irregular timestamps allowed'),
        **({'schedule_provenance': 'caller supplied; prospective availability not independently verified'}
           if elapsed is not None else {}),
        'upstream_processing': 'unknown; timestamp order does not establish causal preprocessing',
        'prediction_provenance': 'caller supplied; training cutoff and prediction freeze not independently verified',
        'claim_limit': 'exported-sequence comparison only; no physical, causal or read-saving claim',
    }, predictions=output)
