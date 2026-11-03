"""Saved declaration checks only; hand-authored fixtures, no study draws."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from groupbench import benchmark, fit_predict
from groupbench.evaluation_schema import _check
from groupbench.prediction_schema import _metadata, _DOMAIN_FIELDS, _DOMAIN_AUDIT_FIELDS
from test_domain_inner import fixture, ARGS


class DomainSchemaContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = fixture()
        cls.pred = fit_predict(f, f[['x']], **ARGS).manifest
        cls.evaluation = benchmark(f, outer_splits=3, **ARGS)
        legacy = ARGS | dict(domain=None, inner_strategy='group_kfold', inner_splits=2,
                             categorical_encoding='distinct-v2')
        cls.legacy_pred = fit_predict(f.drop(columns='d'), f[['x']], **legacy).manifest
        cls.legacy_eval = benchmark(f.drop(columns='d'), outer_splits=3, **legacy)

    def check_eval(self, manifest):
        r = self.evaluation
        return _check(manifest, r.summary, r.predictions.to_csv(index=False).encode())

    def test_all_explicit_legacy_audit_markers_rejected(self):
        for marker in _DOMAIN_AUDIT_FIELDS:
            with self.subTest(marker=marker):
                m = deepcopy(self.legacy_pred)
                m['inner_folds'][0][marker] = None
                with self.assertRaisesRegex(ValueError, 'domain audit'):
                    _metadata(m)
                r = self.legacy_eval
                m = deepcopy(r.manifest)
                m['outer_folds'][0]['inner_folds'][0][marker] = None
                with self.assertRaisesRegex(ValueError, 'domain audit'):
                    _check(m, r.summary, r.predictions.to_csv(index=False).encode())

    def test_config_stripped_downgrade_rejected(self):
        m = deepcopy(self.pred)
        for key in _DOMAIN_FIELDS:
            m['config'].pop(key, None)
        m.update(schema_version='1.2', protocol='group-fit-predict-distinct-v2')
        m['config']['categorical_encoding'] = 'distinct-v2'
        with self.assertRaisesRegex(ValueError, 'domain audit'):
            _metadata(m)
        r = self.evaluation
        m, s = deepcopy(r.manifest), deepcopy(r.summary)
        for key in _DOMAIN_FIELDS:
            m['config'].pop(key, None)
        m['config'].update(inner_splits=3, categorical_encoding='distinct-v2')
        m.update(schema_version='1.1', protocol='nested-group-kfold-distinct-v2')
        s.update(schema_version='1.1', protocol=m['protocol'])
        with self.assertRaisesRegex(ValueError, 'domain audit'):
            _check(m, s, r.predictions.to_csv(index=False).encode())

    def test_unrelated_legacy_metadata_retained(self):
        m = deepcopy(self.legacy_pred)
        m['inner_folds'][0]['user_note'] = {'domain': 'unrelated nested annotation'}
        self.assertEqual(_metadata(m)[0], '1.2')
        r = self.legacy_eval
        m = deepcopy(r.manifest)
        m['outer_folds'][0]['inner_folds'][0]['user_note'] = 'annotation'
        self.assertEqual(_check(m, r.summary, r.predictions.to_csv(index=False).encode())[0], '1.1')

    def test_domain_record_tampering_both_schemas(self):
        changes = [
            lambda a: a.update(inner_selection=a['inner_selection'][:1]),
            lambda a: a['inner_selection'].append(deepcopy(a['inner_selection'][0])),
            lambda a: a['inner_selection'].reverse(),
            lambda a: a['inner_selection'][0].update(candidate='fabricated'),
            lambda a: a['inner_selection'][0].update(family='fabricated'),
            lambda a: a['inner_selection'][1].update(params={'alpha': True}),
            lambda a: a['inner_selection'][0].update(selection_score=-10),
            lambda a: a['inner_selection'][0].update(selection_score=float('inf')),
            lambda a: a['inner_selection'][0].update(selection_score=True),
            lambda a: a['inner_selection'][0]['inner_fold_scores'].__setitem__(0, -1),
            lambda a: a['inner_selection'][0].update(objective='rmse'),
            lambda a: a.update(selected_candidate='fabricated'),
            lambda a: a.update(selected_family='fabricated'),
        ]
        for i, change in enumerate(changes):
            with self.subTest(change=i):
                m = deepcopy(self.pred); change(m)
                with self.assertRaises(ValueError):
                    _metadata(m)
                m = deepcopy(self.evaluation.manifest); change(m['outer_folds'][0])
                with self.assertRaises(ValueError):
                    self.check_eval(m)

    def test_eval_grid_and_family_winner_tampering(self):
        for change in [lambda m: m['config'].update(candidates=m['config']['candidates'][:1]),
                       lambda m: m['config']['candidates'][0].update(name='fabricated'),
                       lambda m: m['outer_folds'][0]['family_candidates'].update(dummy='fabricated')]:
            m = deepcopy(self.evaluation.manifest); change(m)
            with self.assertRaises(ValueError):
                self.check_eval(m)

    def test_frozen_contract_independent_of_installed_grid(self):
        with patch('groupbench.model.candidates_for', side_effect=AssertionError('runtime grid read')):
            self.assertEqual(_metadata(self.pred)[0], '1.4')
            self.assertEqual(self.check_eval(self.evaluation.manifest)[0], '1.3')

    def test_tie_order_declared_scores(self):
        m = deepcopy(self.pred)
        for record in m['inner_selection']:
            record['selection_score'] = 0
        m.update(selected_candidate='dummy_mean', selected_family='dummy')
        self.assertEqual(_metadata(m)[0], '1.4')
        m.update(selected_candidate='ridge_alpha_0.1', selected_family='ridge')
        with self.assertRaises(ValueError):
            _metadata(m)

    def test_extra_trees_frozen_grid(self):
        f = fixture()
        r = fit_predict(f, f[['x']], **(ARGS | dict(baselines=('extra_trees', 'dummy'))))
        self.assertEqual(_metadata(r.manifest)[0], '1.4')


if __name__ == '__main__':
    unittest.main()
