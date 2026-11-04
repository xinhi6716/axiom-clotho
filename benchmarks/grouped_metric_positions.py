"""Paired synthetic public fit_predict timings; no downloads or file writes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import platform
import statistics
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import sklearn

from groupbench import fit_predict, plan_prediction
from groupbench import metrics
from groupbench.data import validate_groups


CASES = {'few_groups': (2000, 20), 'many_groups': (2000, 500),
         'larger_many_groups': (5000, 1250)}
OPTIONS = dict(target='y', group='g', baselines=('dummy', 'ridge'),
               inner_splits=3, random_state=7)


def labeled_index_reference(y_true, y_pred, groups):
    """Frozen prior grouping path; retain the current checked metric arithmetic."""
    y, pred = metrics._metric_inputs(y_true, y_pred)
    g = validate_groups(groups, len(y))
    per_group = []
    for group, positions in pd.Series(np.arange(len(g))).groupby(g, sort=False).groups.items():
        idx = np.asarray(positions, dtype=int)
        per_group.append({'group': str(group), **metrics.regression_metrics(y[idx], pred[idx])})
    overall = metrics.regression_metrics(y, pred)
    with metrics._checked_arithmetic():
        overall.update(n_groups=len(per_group),
                       group_mae=float(np.mean([row['mae'] for row in per_group])),
                       group_rmse=float(np.mean([row['rmse'] for row in per_group])),
                       worst_group_mae=float(max(row['mae'] for row in per_group)))
    return overall, per_group


def stable_result(result):
    manifest = deepcopy(result.manifest)
    for field in ('created_at_utc', 'elapsed_seconds'):
        manifest.pop(field, None)
    # Keep fingerprints, package provenance, selection records and row order.
    return json.dumps({'manifest': manifest,
                       'predictions': result.predictions.to_dict(orient='list'),
                       'selection': result.estimator.selection_results_,
                       'audit': result.estimator.inner_audit_},
                      sort_keys=True, allow_nan=False)


def timed_call(source, query, scorer):
    # Scope the reference substitution to the estimator's imported scorer.
    # Patch setup/restoration and comparison serialization are outside timing.
    with patch('groupbench.model.grouped_metrics', scorer):
        start = time.perf_counter()
        result = fit_predict(source, query, **OPTIONS)
        elapsed = time.perf_counter() - start
    return elapsed, stable_result(result)


def run_case(name, repeats):
    if name not in CASES or type(repeats) is not int or not 1 <= repeats <= 21:
        raise ValueError('Choose a named case and 1 to 21 paired repetitions.')
    rows, count = CASES[name]
    rng = np.random.default_rng(991)
    source = pd.DataFrame(rng.normal(size=(rows, 10)), columns=[f'x{i}' for i in range(10)])
    query = source.iloc[:100].copy()
    source['y'] = source.x0.to_numpy() + rng.normal(size=rows)
    source['g'] = [f'g{i}' for i in np.arange(rows) % count]
    plan = plan_prediction(source, query, **OPTIONS)
    implementations = [('labeled_index_reference', labeled_index_reference),
                       ('positional_indices', metrics.grouped_metrics)]
    controls = [timed_call(source, query, scorer)[1] for _, scorer in implementations]
    if controls[0] != controls[1]:
        raise RuntimeError('Warmup results differ; no timing comparison is valid.')
    samples = {key: [] for key, _ in implementations}
    for repetition in range(repeats):
        order = implementations if repetition % 2 == 0 else implementations[::-1]
        for key, scorer in order:
            elapsed, result = timed_call(source, query, scorer)
            if result != controls[0]:
                raise RuntimeError('Repeated results differ; no timing comparison is valid.')
            samples[key].append(elapsed)
    ratios = [new / old for old, new in zip(samples['labeled_index_reference'],
                                           samples['positional_indices'])]
    return {'case': name, 'rows': rows, 'groups': count, 'features': 10,
            'query_rows': len(query), 'paired_repetitions': repeats,
            'warmup_calls': 2, 'timed_calls': 2 * repeats,
            'planned_pipeline_fits_per_call': plan['fit_budget']['total_fits'],
            'samples_seconds': samples,
            'median_seconds': {key: statistics.median(values) for key, values in samples.items()},
            'paired_positional_to_reference_ratios': ratios,
            'median_paired_ratio': statistics.median(ratios),
            'positional_faster_pairs': sum(value < 1 for value in ratios),
            'all_results_exact_excluding_time': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', *CASES], default='all')
    parser.add_argument('--repeats', type=int, choices=range(1, 22), default=9)
    args = parser.parse_args(argv)
    names = list(CASES) if args.case == 'all' else [args.case]
    result = {'scope': 'Synthetic regression fit_predict only; this performs model fits but writes no files.',
              'limits': 'Host-specific warm timings, not real-data accuracy, a timing gate, a resource bound, or a speed guarantee. Paired ratios do not eliminate shared-host noise.',
              'python': platform.python_version(), 'numpy': np.__version__,
              'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
              'cases': [run_case(name, args.repeats) for name in names]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
