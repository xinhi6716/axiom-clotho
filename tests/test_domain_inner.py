"""Hand-authored software fixtures only; not a scientific simulation study."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from groupbench import benchmark, fit_predict, plan_benchmark, plan_prediction, GroupBenchRegressor
from groupbench.model import selection_plan, _fit_pipeline
from groupbench.evaluation_schema import _check
from groupbench.prediction_schema import _metadata


def fixture():
    return pd.DataFrame({'g':['a','b','c','d','e','f'], 'd':['A','A','B','B','C','C'],
                         'x':[0.,1.,2.,3.,4.,5.], 'y':[0.,2.,3.,4.,5.,6.]})

ARGS=dict(target='y',group='g',domain='d',inner_strategy='leave_one_domain_out',
          inner_splits=None,baselines=('dummy','ridge'))

class DomainInnerTests(unittest.TestCase):
    def test_plan_counts_and_actual_fits(self):
        f=fixture(); p=plan_benchmark(f,outer_splits=3,**ARGS)
        self.assertEqual(p['config']['inner_splits_per_outer'],[3,3,2])
        self.assertEqual(p['fit_budget']['total_fits'],38)
        with patch('groupbench.model._fit_pipeline', wraps=_fit_pipeline) as call:
            r=benchmark(f,outer_splits=3,**ARGS)
        self.assertEqual(call.call_count,38)
        self.assertEqual(_check(r.manifest,r.summary,r.predictions.to_csv(index=False).encode())[0],'1.3')
        self.assertIsNone(r.manifest['config']['inner_splits'])
        self.assertIn('Outer evaluation still holds out groups', ' '.join(r.manifest['limitations']))

    def test_prediction_counts_and_schema(self):
        f=fixture(); p=plan_prediction(f,f[['x']],**ARGS)
        with patch('groupbench.model._fit_pipeline', wraps=_fit_pipeline) as call:
            r=fit_predict(f,f[['x']],**ARGS)
        self.assertEqual(call.call_count,p['fit_budget']['total_fits'])
        self.assertEqual(_metadata(r.manifest)[0],'1.4')
        self.assertEqual(r.estimator.inner_strategy,'leave_one_domain_out')
        self.assertIsNone(r.estimator.inner_splits)
        self.assertEqual(r.manifest['config']['features'],['x'])

    def test_all_outer_folds_preflight_before_any_fit(self):
        f=fixture(); f['d']=['A','A','A','A','A','B']
        with patch('groupbench.model._fit_pipeline', side_effect=AssertionError('fit happened')):
            with self.assertRaisesRegex(ValueError,'at least two'):
                benchmark(f,outer_splits=3,**ARGS)

    def test_invalid_declarations_before_fit(self):
        f=fixture()
        changes=[dict(inner_splits=3),dict(inner_strategy='auto'),dict(domain=None),
                 dict(features=['x','d']),dict(domain='y'),dict(domain='g')]
        with patch('groupbench.model._fit_pipeline', side_effect=AssertionError('fit happened')):
            for change in changes:
                with self.subTest(change=change),self.assertRaises(ValueError):
                    fit_predict(f,f[['x']],**(ARGS|change))
        for vals in [['A']*6,['A','A','B','B','C',None],['A','A','B','B','C',' C']]:
            bad=f.copy();bad['d']=vals
            with self.assertRaises(ValueError):plan_prediction(bad,bad[['x']],**ARGS)
        bad=f.copy();bad.loc[2,'g']='a'
        with self.assertRaisesRegex(ValueError,'nested'):plan_benchmark(bad,outer_splits=3,**ARGS)

    def test_low_level_alignment_and_masks(self):
        f=fixture(); model=GroupBenchRegressor(baselines=('dummy',),inner_strategy='leave_one_domain_out',inner_splits=None)
        for d in [f.d.iloc[::-1], np.ma.array(f.d,mask=[0,0,0,0,0,1]), ['A']*5]:
            with self.assertRaises(ValueError):model.fit(f[['x']],f.y,groups=f.g,domains=d)
        with self.assertRaisesRegex(ValueError,'predictor'):
            model.fit(f[['x','d']],f.y,groups=f.g,domains=f.d)

    def test_target_independent_schedule_and_exact_oof(self):
        f=fixture(); a=plan_benchmark(f,outer_splits=3,**ARGS)
        f.y=[100.,-900.,44.,2.,0.,8.]
        b=plan_benchmark(f,outer_splits=3,**ARGS)
        self.assertEqual(a,b)
        p=selection_plan(f[['x']],f.g,baselines=('dummy',),inner_splits=None,
                         random_state=0,objective='group_mae',inner_strategy='leave_one_domain_out',domains=f.d)
        np.testing.assert_array_equal(np.sort(np.concatenate([v for _,v in p.splits])),np.arange(len(f)))
        for t,v in p.splits:
            self.assertFalse(set(f.g.iloc[t])&set(f.g.iloc[v]))
            self.assertFalse(set(f.d.iloc[t])&set(f.d.iloc[v]))

    def test_legacy_explicit_default_parity(self):
        f=fixture().drop(columns='d'); args=dict(target='y',group='g',baselines=('dummy','ridge'),outer_splits=3,inner_splits=2)
        a=benchmark(f,**args);b=benchmark(f,**args,inner_strategy='group_kfold')
        pd.testing.assert_frame_equal(a.predictions,b.predictions)
        self.assertEqual(a.summary,b.summary)
        self.assertEqual(a.manifest['config'],b.manifest['config'])
        self.assertEqual(a.manifest['schema_version'],'1.0')
        self.assertNotIn('inner_strategy',a.manifest['config'])

    def test_pooled_group_weighting_not_equal_domain_weighting(self):
        f=fixture().iloc[[0,0,0,1,2,3,4,5]].reset_index(drop=True)
        f['d']=['A','A','A','B','B','B','C','C']
        r=fit_predict(f,f[['x']],**(ARGS|dict(baselines=('dummy',))))
        group_errors=[]; domain_errors=[]
        for domain in sorted(set(f.d)):
            train=f[f.d != domain]; valid=f[f.d == domain]
            pred=train.y.mean()
            errors=[float(abs(part.y-pred).mean()) for _,part in valid.groupby('g')]
            domain_errors.append(sum(errors)/len(errors));group_errors.extend(errors)
        score=r.manifest['inner_selection'][0]['selection_score']
        self.assertAlmostEqual(score,sum(group_errors)/len(group_errors))
        self.assertNotAlmostEqual(score,sum(domain_errors)/len(domain_errors))

    def test_optional_families_and_distinct_schema(self):
        f=fixture(); args=ARGS|dict(baselines=('dummy','group_median','nystroem_ridge'),categorical_encoding='distinct-v2')
        r=fit_predict(f,f[['x']],**args)
        self.assertEqual(_metadata(r.manifest)[0],'1.4')
        r=benchmark(f,outer_splits=3,**args)
        self.assertEqual(_check(r.manifest,r.summary,r.predictions.to_csv(index=False).encode())[0],'1.3')

    def test_schema_tampering_and_downgrade(self):
        f=fixture(); r=fit_predict(f,f[['x']],**ARGS)
        changes=[lambda m:m['config'].update(inner_strategy='group_kfold'),
                 lambda m:m['config'].update(n_domains=2),
                 lambda m:m['config'].update(domain='x'),
                 lambda m:m['inner_folds'][0].update(domain_overlap_count=1),
                 lambda m:m['inner_folds'][0].update(test_domains=['B']),
                 lambda m:m.update(schema_version='1.1',protocol='group-fit-predict-v1')]
        for mutate in changes:
            m=deepcopy(r.manifest); mutate(m)
            with self.assertRaises(ValueError):_metadata(m)
        r=benchmark(f,outer_splits=3,**ARGS)
        m=deepcopy(r.manifest);m['config']['inner_splits_per_outer'][0]=2
        with self.assertRaises(ValueError):_check(m,r.summary,r.predictions.to_csv(index=False).encode())

if __name__=='__main__':unittest.main()
