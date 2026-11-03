"""Deterministic fake artifacts only: no study RNG, fitted models or results."""
import copy
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import check_saved as qa


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def fixture(scenario=0, root=731903):
    groups = qa.EXPECTED_GROUPS
    folds = [int(g[1:]) % 3 for g in groups]
    case = {'case_id': f'{scenario}_{root}', 'scenario_id': scenario, 'scenario': qa.SCENARIOS[scenario],
            'root': root, 'fold_ids': folds,
            'fold_roles': ['left_extrapolation', 'interpolation', 'right_extrapolation'] if scenario == 2 else ['iid_group_holdout'] * 3,
            'plan': {'fit_budget': {'total_fits': 75}}}
    x = np.asarray([[-1.5, 0., 1.5][f] + .01 * (i % 2) if scenario == 2 else (i - 27) / 100 for i, f in enumerate(folds)], dtype=np.float64)
    z = np.zeros(54, dtype=np.float64)
    y = np.asarray([1.5 + 2 * float(v) for v in x], dtype=np.float64)
    predictions = {name: [float(v) + offset for v in y] for name, offset in zip(qa.FAMILIES, [.5, .2, .1, -.1, .4])}
    predictions['selected'] = predictions['extra_trees'].copy()
    families = ['dummy'] + ['ridge'] * 3 + ['extra_trees'] * 2 + ['nystroem_ridge'] * 3 + ['group_median']
    records = [{'candidate': f'fake_candidate_{i}', 'family': family, 'selection_score': score,
                'objective': 'group_mae', 'inner_fold_scores': [score, score]} for i, (family, score) in enumerate(zip(families, [1., .5, .6, .7, .2, .3, .4, .5, .6, .9]))]
    outer = []
    for fold in range(3):
        train = [i for i, f in enumerate(folds) if f != fold]; test = [i for i, f in enumerate(folds) if f == fold]
        tg = sorted({groups[i] for i in train}); vg = sorted({groups[i] for i in test})
        inners = []
        for j in range(2):
            b = tg[j::2]; a = [g for g in tg if g not in b]
            inners.append({'fold': j, 'train_groups': a, 'test_groups': b, 'overlap_count': 0,
                           'train_rows': sum(g in a for g in groups), 'test_rows': sum(g in b for g in groups),
                           'train_group_count': len(a), 'test_group_count': len(b)})
        outer.append({'fold': fold, 'train_rows': len(train), 'test_rows': len(test),
                      'train_group_count': len(tg), 'test_group_count': len(vg), 'train_groups': tg, 'test_groups': vg,
                      'overlap_count': 0, 'inner_seed': root + fold + 1, 'inner_folds': inners,
                      'inner_selection': copy.deepcopy(records), 'selected_family': 'extra_trees',
                      'selected_candidate': 'fake_candidate_4',
                      'family_candidates': {f: min([r for r in records if r['family'] == f], key=lambda r:r['selection_score'])['candidate'] for f in qa.FAMILIES}})
    manifest = {'n_rows': 54, 'n_groups': 18, 'package_source': {'status': 'available', 'files': {'fake.py': '0' * 64}},
                'config': {'features': ['x', 'z'], 'baselines': qa.FAMILIES, 'outer_splits': 3, 'inner_splits': 2,
                           'random_state': root, 'selection_objective': 'group_mae', 'threads': 1,
                           'candidates': [{'name': r['candidate']} for r in records]}, 'outer_folds': outer}
    summary = {'models': {}}
    for name in qa.ENTRIES:
        overall, per_group = qa.metrics(y.tolist(), predictions[name], groups)
        per_fold = []
        for f in range(3):
            idx = [i for i, k in enumerate(folds) if k == f]
            scores, _ = qa.metrics([float(y[i]) for i in idx], [predictions[name][i] for i in idx], [groups[i] for i in idx])
            per_fold.append({'fold': f, **scores})
        summary['models'][name] = {'overall': overall, 'per_group': per_group, 'per_fold': per_fold}
    array_bytes = io.BytesIO(); np.savez(array_bytes, x=x, z=z, y=y, group=np.asarray(groups, dtype='U3'))
    stream = io.StringIO(newline=''); writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['row_position', 'group', 'outer_fold', 'observed'] + ['prediction_' + n for n in qa.ENTRIES])
    for i in range(54): writer.writerow([i, groups[i], folds[i], float(y[i])] + [predictions[n][i] for n in qa.ENTRIES])
    raws = {'data.npz': array_bytes.getvalue(), 'predictions.csv': stream.getvalue().encode(),
            'manifest.json': encoded(manifest), 'summary.json': encoded(summary)}
    return case, raws


def fake_sealed(package):
    root = package / "artifacts"
    root.mkdir()
    cases, seals = [], []
    for scenario in range(3):
        for seed in qa.ROOTS:
            case, raws = fixture(scenario, seed); cases.append(case)
            folder = root/'outputs'/case['case_id']; folder.mkdir(parents=True)
            for name, raw in raws.items(): (folder/name).write_bytes(raw)
            seal = {'case_id': case['case_id'], 'files': {name: qa.sha(raw) for name, raw in raws.items()}, 'fits_completed': 75}
            (folder/'SEALED.json').write_bytes(encoded(seal)); seals.append(seal)
    prepared_files = {'config.json': {'roots': qa.ROOTS, 'scenarios': qa.SCENARIOS, 'families': qa.FAMILIES},
                      'cases.json': cases, 'runtime.json': {}, 'source_hashes.json': {'src/groupbench/fake.py': '0'*64}}
    for name, value in prepared_files.items(): (root/name).write_bytes(encoded(value))
    prep = {'study_id': qa.STUDY_ID, 'protocol_sha256': qa.PROTOCOL_SHA256,
            'files': {name: qa.sha((root/name).read_bytes()) for name in prepared_files}}
    (root/'PREPARED.json').write_bytes(encoded(prep))
    (root/'SUPERVISOR.json').write_bytes(encoded({'status': 'SEALED', 'returncode': 0, 'stop_reason': None,
                                               'child_cpu_seconds': 1., 'wall_seconds': 1.,
                                               'sampled_peak_rss_bytes': 1000, 'child_rusage_max_rss_kib': 1}))
    (root/'ALL_CASES_SEALED.json').write_bytes(encoded({'status': 'ALL_15_SEALED_BEFORE_OPERATOR_COMPARISON', 'cases': seals,
                                                     'stats': {'attempted_fits': 1125, 'completed_fits': 1125, 'failed_fits': 0,
                                                               'datasets_drawn': 15, 'cases_sealed': 15}}))
    (root/'PREPARED.json').unlink()
    provenance = {'study_id': qa.STUDY_ID, 'protocol_sha256': qa.PROTOCOL_SHA256,
                  'files': {str(p.relative_to(package)): qa.sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()}}
    (package/'PUBLIC_PROVENANCE.json').write_bytes(encoded(provenance))
    return qa.sha(encoded(provenance))


class EvaluatorTests(unittest.TestCase):
    def test_hand_computed_oracle(self):
        result, groups = qa.metrics([0., 2., 4.], [1., 2., 2.], ['a', 'a', 'b'])
        self.assertEqual(result['mae'], 1.)
        self.assertEqual(result['rmse'], math.sqrt(5/3))
        self.assertEqual(result['r2'], .375)
        self.assertEqual(result['group_mae'], 1.25)
        self.assertEqual(result['worst_group_mae'], 2.)
        self.assertIsNone(groups[1]['r2'])

    def test_fake_case_metrics_and_selected_identity(self):
        case, raws = fixture(); result = qa.analyze_case(raws, case)
        self.assertEqual(set(result['models']), set(qa.ENTRIES))
        self.assertLess(result['contrasts']['kernel_minus_ridge']['overall'], 0)

    def test_bad_selected_prediction_rejects(self):
        case, raws = fixture(); rows = list(csv.reader(io.StringIO(raws['predictions.csv'].decode())))
        rows[1][-1] = str(float(rows[1][-1]) + 1.)
        stream = io.StringIO(); csv.writer(stream).writerows(rows); raws['predictions.csv'] = stream.getvalue().encode()
        with self.assertRaisesRegex(ValueError, 'Selected prediction'): qa.analyze_case(raws, case)

    def test_incorrect_stored_score_and_denominator_reject(self):
        for field, value in [('group_mae', 999), ('n_groups', 999)]:
            case, raws = fixture(); saved = json.loads(raws['summary.json']); saved['models']['ridge']['overall'][field] = value; raws['summary.json'] = encoded(saved)
            with self.assertRaises(ValueError): qa.analyze_case(raws, case)

    def test_inner_selection_and_leakage_reject(self):
        for corruption in ['selected', 'leakage', 'candidate_order']:
            case, raws = fixture(); m = json.loads(raws['manifest.json'])
            if corruption == 'selected': m['outer_folds'][0]['selected_candidate'] = 'fake_candidate_0'
            if corruption == 'leakage': m['outer_folds'][0]['inner_folds'][0]['test_groups'].append(m['outer_folds'][0]['test_groups'][0])
            if corruption == 'candidate_order': m['outer_folds'][0]['inner_selection'].reverse()
            raws['manifest.json'] = encoded(m)
            with self.assertRaises(ValueError): qa.analyze_case(raws, case)

    def test_data_and_csv_identity_not_just_metrics(self):
        case, raws = fixture(); rows = list(csv.reader(io.StringIO(raws['predictions.csv'].decode())))
        rows[1][3] = str(float(rows[1][3]) + 1.)
        stream = io.StringIO(); csv.writer(stream).writerows(rows); raws['predictions.csv'] = stream.getvalue().encode()
        with self.assertRaisesRegex(ValueError, 'Saved outcome'): qa.analyze_case(raws, case)

    def test_all_fifteen_fake_seals_and_missing_tamper_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); expected = fake_sealed(root)
            result = qa.evaluate(root, expected)
            self.assertEqual(result['cases_checked'], 15)
            self.assertEqual(result['fits_performed'], 0)
            self.assertEqual(result['study_rng_calls'], 0)
            scenario = result['scenario_summaries']['linear_domain_holdout']['contrasts']['kernel_minus_ridge']
            self.assertEqual(set(scenario), {'mixed_whole_cohort', 'left_extrapolation', 'interpolation', 'right_extrapolation'})
            for scope in scenario.values():
                self.assertEqual(scope['denominator'], 5)
                self.assertEqual(scope['improvement_count'] + scope['tie_count'] + scope['harm_count'], 5)
            (root/'artifacts'/'FAILURE.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Failure artifact'): qa.evaluate(root, expected)
            (root/'artifacts'/'FAILURE.json').unlink()
            target = root/'artifacts'/'outputs'/'0_731903'/'predictions.csv'; target.write_bytes(target.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'bytes changed'): qa.evaluate(root, expected)

    def test_public_manifest_pin_inventory_and_traversal_reject(self):
        for corruption in ['external_pin', 'missing', 'traversal']:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); expected = fake_sealed(root)
                if corruption == 'external_pin':
                    with self.assertRaisesRegex(ValueError, 'external pin'): qa.evaluate(root, '0' * 64)
                    continue
                path = root / 'PUBLIC_PROVENANCE.json'; doc = json.loads(path.read_bytes())
                if corruption == 'missing': del doc['files']['artifacts/config.json']
                else: doc['files']['../escape'] = '0' * 64
                path.write_bytes(encoded(doc))
                with self.assertRaises(ValueError): qa.evaluate(root)

    def test_symlink_parent_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fake_sealed(root)
            (root/'artifacts').rename(root/'relocated')
            (root/'artifacts').symlink_to('relocated', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Symlink'): qa.evaluate(root)

    def test_no_product_model_modules_imported(self):
        self.assertFalse(any(name == 'groupbench' or name.startswith(('groupbench.', 'sklearn.', 'pandas.')) for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
