"""Independent saved-output QA. No product/model imports, RNG calls or refits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np

STUDY_ID = 'CLOTHO-OPTIONAL-GROUPED-SYNTH-V1'
PROTOCOL_SHA256 = '708227ab3f634ffaed0ab5179e9189e4ea4f6019ea7e8c8e8eb159adcf41556a'
ROOTS = [731903, 731927, 731939, 731951, 731967]
SCENARIOS = ['linear_iid', 'smooth_iid', 'linear_domain_holdout']
FAMILIES = ['dummy', 'ridge', 'extra_trees', 'nystroem_ridge', 'group_median']
ENTRIES = FAMILIES + ['selected']
CONTRASTS = [('kernel_minus_ridge', 'nystroem_ridge', 'ridge'),
             ('kernel_minus_extra_trees', 'nystroem_ridge', 'extra_trees'),
             ('median_minus_dummy', 'group_median', 'dummy'),
             ('selected_minus_ridge', 'selected', 'ridge')]
EXPECTED_GROUPS = [f'g{i:02d}' for i, size in enumerate([2, 3, 4] * 6) for _ in range(size)]
METRICS = ['mae', 'rmse', 'r2', 'bias', 'group_mae', 'group_rmse', 'worst_group_mae']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, maximum=2 * 1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Expected regular non-symlink artifact')
    require(path.stat().st_size <= maximum, 'Artifact exceeds bounded size')
    raw = path.read_bytes()
    require(len(raw) <= maximum, 'Artifact changed beyond bounded size')
    return raw


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate artifact JSON key')
            result[key] = value
        return result
    def constant(_token):
        raise ValueError('Nonfinite artifact JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def close(actual, expected, context):
    if expected is None:
        require(actual is None, context + ': null mismatch')
    else:
        require(type(actual) in (int, float) and math.isfinite(actual), context + ': nonfinite/non-numeric')
        require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12),
                context + ': independent metric mismatch')


def scalar_metrics(y, prediction):
    """Independent Python-fsum arithmetic, not product metric helpers."""
    require(len(y) == len(prediction) and len(y) > 0, 'Empty/mismatched metric rows')
    require(all(math.isfinite(v) for v in y + prediction), 'Nonfinite metric source')
    residual = [p - v for v, p in zip(y, prediction)]
    n = len(y)
    # Anchoring avoids throwing away variance at a large common offset.
    anchored = [v - y[0] for v in y]
    center = math.fsum(anchored) / n
    total = math.fsum((v - center) ** 2 for v in anchored)
    squared = math.fsum(r * r for r in residual)
    return {'n_rows': n, 'mae': math.fsum(abs(r) for r in residual) / n,
            'rmse': math.sqrt(squared / n), 'bias': math.fsum(residual) / n,
            'r2': 1 - squared / total if n > 1 and total > 0 else None}


def metrics(y, prediction, groups):
    result = scalar_metrics(y, prediction)
    positions = {}
    for i, group in enumerate(groups):
        positions.setdefault(group, []).append(i)
    rows = [{'group': group, **scalar_metrics([y[i] for i in ids], [prediction[i] for i in ids])}
            for group, ids in positions.items()]
    result.update(n_groups=len(rows), group_mae=math.fsum(r['mae'] for r in rows) / len(rows),
                  group_rmse=math.fsum(r['rmse'] for r in rows) / len(rows),
                  worst_group_mae=max(r['mae'] for r in rows))
    return result, rows


def compare_metrics(saved, recomputed, context):
    for count in ['n_rows', 'n_groups']:
        if count in recomputed:
            require(type(saved.get(count)) is int and saved[count] == recomputed[count], context + ': denominator mismatch')
    for name in METRICS:
        if name in recomputed:
            close(saved.get(name), recomputed[name], context + '/' + name)


def compare_group_rows(saved, recomputed, context):
    require(len(saved) == len(recomputed), context + ': group count mismatch')
    mapping = {r['group']: r for r in saved}
    require(len(mapping) == len(saved) and set(mapping) == {r['group'] for r in recomputed}, context + ': group identity mismatch')
    for row in recomputed:
        compare_metrics(mapping[row['group']], row, context + '/' + row['group'])


def read_case(raws, case):
    """Decode exact verified in-memory bytes; never reopen a scored payload."""
    require(set(raws) == {'data.npz', 'manifest.json', 'summary.json', 'predictions.csv'}, 'Unexpected case artifact inventory')
    with zipfile.ZipFile(io.BytesIO(raws['data.npz'])) as archive:
        require(sum(i.file_size for i in archive.infolist()) <= 1024**2, 'NPZ expanded size exceeds cap')
        require(set(archive.namelist()) == {'x.npy', 'z.npy', 'y.npy', 'group.npy'}, 'Unexpected NPZ array inventory')
    with np.load(io.BytesIO(raws['data.npz']), allow_pickle=False) as arrays:
        values = {key: arrays[key] for key in arrays.files}
    for name in ['x', 'z', 'y']:
        require(values[name].shape == (54,) and values[name].dtype == np.dtype('float64'), 'Wrong numeric array contract')
        require(np.isfinite(values[name]).all(), 'Nonfinite generated source')
    require(values['group'].shape == (54,) and values['group'].dtype == np.dtype('U3'), 'Wrong group array contract')
    require(values['group'].tolist() == EXPECTED_GROUPS, 'Generated group roster mismatch')
    require(np.all((values['z'] >= -1) & (values['z'] <= 1)), 'z outside frozen domain')
    for i, x in enumerate(values['x']):
        bounds = [(-2, -1), (-.5, .5), (1, 2)][case['fold_ids'][i]] if case['scenario_id'] == 2 else (-1, 1)
        require(bounds[0] <= x <= bounds[1], 'x outside prospective fold domain')
    reader = csv.DictReader(io.StringIO(raws['predictions.csv'].decode('utf-8'), newline=''), strict=True)
    expected_columns = ['row_position', 'group', 'outer_fold', 'observed'] + ['prediction_' + n for n in ENTRIES]
    require(reader.fieldnames == expected_columns, 'Prediction columns/order differ')
    rows = list(reader)
    require(len(rows) == 54, 'Prediction row roster incomplete')
    for i, row in enumerate(rows):
        require(set(row) == set(expected_columns) and None not in row.values(), 'Malformed prediction row')
        require(row['row_position'] == str(i) and row['group'] == EXPECTED_GROUPS[i], 'Prediction row identity differs')
        require(row['outer_fold'] == str(case['fold_ids'][i]), 'Prediction fold assignment differs')
        require(float(row['observed']) == float(values['y'][i]), 'Saved outcome differs from generated source')
    predictions = {name: [float(row['prediction_' + name]) for row in rows] for name in ENTRIES}
    require(all(math.isfinite(x) for values_ in predictions.values() for x in values_), 'Nonfinite saved prediction')
    return values['y'].tolist(), EXPECTED_GROUPS, predictions, parse_json(raws['manifest.json']), parse_json(raws['summary.json'])


def check_splits_and_selection(manifest, case, predictions):
    require(manifest['n_rows'] == 54 and manifest['n_groups'] == 18, 'Manifest roster differs')
    config = manifest['config']
    require(config['features'] == ['x', 'z'] and config['baselines'] == FAMILIES, 'Feature/family contract differs')
    require(config['outer_splits'] == 3 and config['inner_splits'] == 2 and config['random_state'] == case['root'], 'Split config differs')
    require(config['selection_objective'] == 'group_mae' and config['threads'] == 1, 'Selection/thread contract differs')
    require(len(manifest['outer_folds']) == 3, 'Outer audit incomplete')
    for fold, audit in enumerate(manifest['outer_folds']):
        test = [i for i, f in enumerate(case['fold_ids']) if f == fold]
        train = [i for i, f in enumerate(case['fold_ids']) if f != fold]
        train_groups, test_groups = {EXPECTED_GROUPS[i] for i in train}, {EXPECTED_GROUPS[i] for i in test}
        require(audit['fold'] == fold and audit['overlap_count'] == 0, 'Outer fold identity/overlap differs')
        require(set(audit['train_groups']) == train_groups and set(audit['test_groups']) == test_groups, 'Outer group audit differs')
        require(audit['train_rows'] == len(train) and audit['test_rows'] == len(test), 'Outer row denominator differs')
        require(audit['train_group_count'] == len(train_groups) and audit['test_group_count'] == len(test_groups), 'Outer group denominator differs')
        require(audit['inner_seed'] == case['root'] + fold + 1, 'Inner seed differs')
        inner = audit['inner_folds']; require(len(inner) == 2, 'Inner audit incomplete')
        held_out = []
        for j, entry in enumerate(inner):
            a, b = set(entry['train_groups']), set(entry['test_groups'])
            require(entry['fold'] == j and not (a & b) and a | b == train_groups, 'Inner groups overlap or escape outer training')
            require(entry['overlap_count'] == 0, 'Inner overlap declared')
            require(entry['train_group_count'] == len(a) and entry['test_group_count'] == len(b), 'Inner group denominator differs')
            require(entry['train_rows'] == sum(g in a for g in [EXPECTED_GROUPS[i] for i in train]), 'Inner train rows differ')
            require(entry['test_rows'] == sum(g in b for g in [EXPECTED_GROUPS[i] for i in train]), 'Inner test rows differ')
            held_out.extend(entry['test_groups'])
        require(len(held_out) == len(train_groups) and set(held_out) == train_groups, 'Inner OOF group coverage differs')
        records = audit['inner_selection']; require(len(records) == 10, 'Candidate selection table incomplete')
        require([r['candidate'] for r in records] == [c['name'] for c in config['candidates']], 'Candidate order differs from frozen manifest')
        require({family: sum(r['family'] == family for r in records) for family in FAMILIES} == {'dummy': 1, 'ridge': 3, 'extra_trees': 2, 'nystroem_ridge': 3, 'group_median': 1}, 'Candidate family counts differ')
        for record in records:
            require(len(record['inner_fold_scores']) == 2 and all(math.isfinite(v) for v in record['inner_fold_scores']), 'Inner fold scores incomplete')
            require(record['objective'] == 'group_mae' and math.isfinite(record['selection_score']), 'Invalid inner score')
        winner = min(range(len(records)), key=lambda i: records[i]['selection_score'])
        require(audit['selected_family'] == records[winner]['family'] and audit['selected_candidate'] == records[winner]['candidate'], 'Recorded selected winner/tie order differs')
        for family in FAMILIES:
            eligible = [i for i, r in enumerate(records) if r['family'] == family]
            require(bool(eligible), 'Missing family candidates')
            chosen = min(eligible, key=lambda i: records[i]['selection_score'])
            require(audit['family_candidates'][family] == records[chosen]['candidate'], 'Recorded family winner differs')
        family = audit['selected_family']
        for i in test:
            require(predictions['selected'][i] == predictions[family][i], 'Selected prediction does not match its saved chosen family')


def analyze_case(raws, case):
    y, groups, predictions, manifest, summary = read_case(raws, case)
    check_splits_and_selection(manifest, case, predictions)
    require(set(summary['models']) == set(ENTRIES), 'Saved score entry inventory differs')
    models = {}
    for name in ENTRIES:
        overall, rows = metrics(y, predictions[name], groups)
        saved = summary['models'][name]
        compare_metrics(saved['overall'], overall, case['case_id'] + '/' + name + '/overall')
        compare_group_rows(saved['per_group'], rows, case['case_id'] + '/' + name + '/groups')
        require(len(saved['per_fold']) == 3, 'Saved fold scores incomplete')
        folds = []
        for fold in range(3):
            positions = [i for i, value in enumerate(case['fold_ids']) if value == fold]
            fold_metrics, _ = metrics([y[i] for i in positions], [predictions[name][i] for i in positions], [groups[i] for i in positions])
            saved_fold = saved['per_fold'][fold]
            require(saved_fold['fold'] == fold, 'Saved fold ordering differs')
            compare_metrics(saved_fold, fold_metrics, case['case_id'] + '/' + name + '/fold' + str(fold))
            folds.append({'fold': fold, 'role': case['fold_roles'][fold], **fold_metrics})
        models[name] = {'overall': overall, 'per_group': rows, 'per_fold': folds}
    contrasts = {}
    for label, a, b in CONTRASTS:
        contrasts[label] = {'overall': models[a]['overall']['group_mae'] - models[b]['overall']['group_mae'],
                            'per_fold': [{'fold': f, 'role': case['fold_roles'][f],
                                          'difference': models[a]['per_fold'][f]['group_mae'] - models[b]['per_fold'][f]['group_mae']} for f in range(3)]}
    return {'case_id': case['case_id'], 'scenario': case['scenario'], 'root': case['root'], 'models': models, 'contrasts': contrasts}


def aggregate(cases):
    results = {}
    for scenario in SCENARIOS:
        items = [c for c in cases if c['scenario'] == scenario]
        require([c['root'] for c in items] == ROOTS, 'Scenario root roster differs')
        summaries = {}
        for label, _, _ in CONTRASTS:
            scopes = {'mixed_whole_cohort' if scenario == SCENARIOS[2] else 'whole_cohort': [c['contrasts'][label]['overall'] for c in items]}
            if scenario == SCENARIOS[2]:
                for fold, role in enumerate(['left_extrapolation', 'interpolation', 'right_extrapolation']):
                    scopes[role] = [c['contrasts'][label]['per_fold'][fold]['difference'] for c in items]
            summaries[label] = {scope: {'roots': ROOTS, 'paired_differences': values,
                                        'mean': math.fsum(values) / len(values), 'minimum': min(values), 'maximum': max(values),
                                        'improvement_count': sum(v < 0 for v in values), 'tie_count': sum(v == 0 for v in values),
                                        'harm_count': sum(v > 0 for v in values), 'denominator': 5}
                               for scope, values in scopes.items()}
        model_ranges = {name: {metric: {'mean': math.fsum(c['models'][name]['overall'][metric] for c in items) / 5,
                                        'minimum': min(c['models'][name]['overall'][metric] for c in items),
                                        'maximum': max(c['models'][name]['overall'][metric] for c in items)}
                               for metric in ['group_mae', 'mae', 'rmse', 'worst_group_mae']} for name in ENTRIES}
        results[scenario] = {'model_summaries': model_ranges, 'contrasts': summaries}
    return results


def evaluate(package, expected_provenance_sha=None):
    """Check declared byte linkage, not authorship or historical authenticity."""
    package = Path(package)
    raw = read(package / 'PUBLIC_PROVENANCE.json')
    provenance_sha = sha(raw)
    if expected_provenance_sha is not None:
        require(provenance_sha == expected_provenance_sha, 'Public external pin differs')
    provenance = parse_json(raw)
    require(provenance['study_id'] == STUDY_ID and provenance['protocol_sha256'] == PROTOCOL_SHA256, 'Study/protocol pin differs')
    inventory = provenance['files']
    require(type(inventory) is dict and 81 <= len(inventory) <= 200, 'Public inventory size differs')
    required = {'artifacts/' + name for name in ['config.json', 'cases.json', 'runtime.json', 'source_hashes.json', 'ALL_CASES_SEALED.json', 'SUPERVISOR.json']}
    required.update('artifacts/outputs/' + str(i) + '_' + str(seed) + '/' + name
                    for i in range(3) for seed in ROOTS
                    for name in ['data.npz', 'manifest.json', 'summary.json', 'predictions.csv', 'SEALED.json'])
    require(required <= set(inventory), 'Missing required public artifact pin')
    verified = {}
    for name, digest in inventory.items():
        require(type(name) is str and name and not name.startswith('/') and
                all(p not in ('', '.', '..') for p in name.split('/')) and '\\' not in name,
                'Unsafe public artifact path')
        target = package / name
        require(not any(p.is_symlink() for p in [target, *target.parents] if p != package.parent), 'Symlink public artifact path')
        payload = read(target)
        require(sha(payload) == digest, 'Public artifact bytes changed')
        verified[name] = payload
    root = package / 'artifacts'
    files = {name: parse_json(verified['artifacts/' + name])
             for name in ['config.json', 'cases.json', 'runtime.json', 'source_hashes.json']}
    cases = files['cases.json']
    require([c['case_id'] for c in cases] == [f'{i}_{seed}' for i in range(3) for seed in ROOTS], 'Complete prospective case roster differs')
    require(files['config.json']['roots'] == ROOTS and files['config.json']['scenarios'] == SCENARIOS and files['config.json']['families'] == FAMILIES, 'Configuration roster differs')
    supervisor = parse_json(read(root / 'SUPERVISOR.json'))
    require(supervisor['status'] == 'SEALED' and supervisor['returncode'] == 0 and supervisor['stop_reason'] is None, 'Run did not terminate successfully')
    require(supervisor['child_cpu_seconds'] <= 200 and supervisor['wall_seconds'] <= 900, 'Main resource budget exceeded')
    require(supervisor['sampled_peak_rss_bytes'] <= 512 * 1024**2 and supervisor['child_rusage_max_rss_kib'] * 1024 <= 512 * 1024**2, 'Main RSS budget exceeded')
    require(not (root / 'FAILURE.json').exists(), 'Failure artifact exists; do not score survivors as complete')
    sealed = parse_json(read(root / 'ALL_CASES_SEALED.json'))
    require(sealed['status'] == 'ALL_15_SEALED_BEFORE_OPERATOR_COMPARISON', 'Complete sealing receipt absent')
    stats = sealed['stats']
    require(all(stats[key] == value for key, value in {'attempted_fits': 1125, 'completed_fits': 1125, 'failed_fits': 0, 'datasets_drawn': 15, 'cases_sealed': 15}.items()), 'Main execution counts differ')
    require([c['case_id'] for c in sealed['cases']] == [c['case_id'] for c in cases], 'Sealed case roster differs')
    analyzed = []
    for case, seal in zip(cases, sealed['cases']):
        require(len(case['fold_ids']) == 54 and set(case['fold_ids']) == {0, 1, 2}, 'Prospective fold map invalid')
        require(case['scenario'] == SCENARIOS[case['scenario_id']] and case['root'] in ROOTS and case['case_id'] == f"{case['scenario_id']}_{case['root']}", 'Case identity differs')
        require(case['plan']['fit_budget']['total_fits'] == 75, 'Prospective fit plan differs')
        for group in set(EXPECTED_GROUPS):
            require(len({case['fold_ids'][i] for i, g in enumerate(EXPECTED_GROUPS) if g == group}) == 1, 'Prospective group is split')
        expected_roles = ['left_extrapolation', 'interpolation', 'right_extrapolation'] if case['scenario_id'] == 2 else ['iid_group_holdout'] * 3
        require(case['fold_roles'] == expected_roles, 'Fold role labels differ')
        folder = root / 'outputs' / case['case_id']
        require(parse_json(read(folder / 'SEALED.json')) == seal and seal['fits_completed'] == 75, 'Case seal differs')
        require(set(seal['files']) == {'data.npz', 'manifest.json', 'summary.json', 'predictions.csv'}, 'Sealed file inventory differs')
        raws = {}
        for name, digest in seal['files'].items():
            payload = read(folder / name); require(sha(payload) == digest, 'Sealed case bytes changed'); raws[name] = payload
        observed_source = parse_json(raws['manifest.json'])['package_source']
        expected_source = {name.removeprefix('src/groupbench/'): digest for name, digest in files['source_hashes.json'].items()}
        require(observed_source['status'] == 'available' and observed_source['files'] == expected_source, 'Observed source files differ from prospective source pins')
        analyzed.append(analyze_case(raws, case))
    return {'study_id': STUDY_ID, 'status': 'PASS_INDEPENDENT_NO_REFIT_QA', 'public_provenance_sha256': provenance_sha,
            'protocol_sha256': PROTOCOL_SHA256, 'cases_checked': 15, 'models_per_case': 6,
            'case_results': analyzed, 'scenario_summaries': aggregate(analyzed),
            'numeric_comparison': {'method': 'independent Python math.fsum arithmetic', 'relative_tolerance': 1e-10, 'absolute_tolerance': 1e-12,
                                   'contrast_comparison': 'signed recomputed float values before display rounding'},
            'fits_performed': 0, 'study_rng_calls': 0,
            'limitations': ['No product model or metric imports; no refitting or reconstruction of random draws.',
                            'Checksums establish declared byte linkage, not authorship or historical authenticity.',
                            'A zero sampled RSS receipt is a monitoring gap, not zero memory use or verified enforcement.',
                            'Inner winner checks use saved selection scores, not independently regenerated inner predictions.',
                            'Five seed repeats are descriptive; rows and overlapping-training folds are not independent repeats.',
                            'Scenario2 pooled results mix two extrapolation folds and one interpolation fold.',
                            'Median versus dummy changes both location rule and weighting, not isolated evidence for weighting.',
                            'Synthetic registered results do not establish real material/device or causal efficacy.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--provenance-sha256', help='Optional independently obtained public manifest pin')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(not any(name == 'groupbench' or name.startswith(('groupbench.', 'sklearn.', 'pandas.')) for name in sys.modules), 'Unexpected product/model/dataframe import')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU, (50, 51))
    started = time.monotonic()
    result = evaluate(args.package, args.provenance_sha256)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result['qa_resources'] = {'cpu_seconds_since_process_start': usage.ru_utime + usage.ru_stime,
                              'wall_seconds_evaluation': time.monotonic() - started, 'max_rss_kib': usage.ru_maxrss}
    require(result['qa_resources']['cpu_seconds_since_process_start'] <= 50, 'QA CPU reservation exceeded')
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with Path(args.output).open('x') as handle:
        handle.write(payload)
    print(json.dumps({'status': result['status'], 'cases_checked': 15, 'fits_performed': 0}))


if __name__ == '__main__':
    main()
