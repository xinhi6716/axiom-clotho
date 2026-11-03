import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from groupbench.input_support import outer_fold_input_support, prediction_input_support
from groupbench.splitting import GroupKFold
from groupbench.data import validate_features
from groupbench.evaluation import _prepare_benchmark

class OuterSupportTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({'x':np.arange(18,dtype=float),'kind':['a','b','c']*6}), np.repeat(['a','b','c','d','e','f'],3)

    def test_exact_benchmark_schedule_and_no_fit(self):
        X,g=self.fixture();frame=X.assign(group=g,y=np.arange(18))
        _,_,_,_,schedule=_prepare_benchmark(frame,target='y',group='group',features=['x','kind'],outer_splits=3,inner_splits=2,random_state=9,baselines=('ridge',),objective='group_mae')
        with patch('groupbench.model.make_pipeline',side_effect=AssertionError('no model construction')):
            out=outer_fold_input_support(X,g,outer_splits=3,random_state=9)
        for fold,(train,test,_) in zip(out['folds'],schedule):
            expected=prediction_input_support(validate_features(X).iloc[train],X.iloc[test])
            expected['reference']='outer_training_features_before_imputation'
            self.assertEqual(fold['input_support'],expected)
            self.assertEqual(fold['flagged_row_fraction'],expected['rows_with_any_flag']/len(test))
        self.assertEqual(sum(f['input_support']['n_prediction_rows'] for f in out['folds']),18)

    def test_extremes_local_reference_no_query_expansion(self):
        # Derive prospective domains from the actual splitter, not target values.
        g=np.repeat(['a','b','c','d','e','f'],2);X=pd.DataFrame({'x':np.zeros(12)})
        for fold,(_,test) in enumerate(GroupKFold(3,4).split(X,groups=g)):X.loc[test,'x']=(-2.,0.,2.)[fold]
        out=outer_fold_input_support(X,g,outer_splits=3,random_state=4)
        self.assertEqual([f['input_support']['row_flag_counts']['numeric_outside_training_range'] for f in out['folds']],[4,0,4])
        self.assertEqual(out['folds'][0]['input_support']['features'][0]['below_training_min_count'],4)
        self.assertEqual(out['folds'][2]['input_support']['features'][0]['above_training_max_count'],4)

    def test_missing_unseen_and_unavailable_reference(self):
        g=np.repeat(['a','b','c','d','e','f'],2);X=pd.DataFrame({'x':np.full(12,np.nan),'kind':['known']*12})
        test=next(GroupKFold(3,4).split(X,groups=g))[1];X.loc[test,'x']=3.;X.loc[test,'kind']='new'
        support=outer_fold_input_support(X,g,outer_splits=3,random_state=4)['folds'][0]['input_support']
        self.assertIsNone(support['features'][0]['below_training_min_count'])
        self.assertEqual(support['features'][0]['prediction_non_missing_without_training_values_count'],4)
        self.assertEqual(support['features'][1]['unseen_category_count'],4)

    def test_boundary_equality_determinism_detachment_and_no_mutation(self):
        X,g=self.fixture();X['x']=1.;before=X.copy(deep=True)
        a=outer_fold_input_support(X,g,outer_splits=3);b=outer_fold_input_support(X,g,outer_splits=3)
        self.assertEqual(json.dumps(a,allow_nan=False),json.dumps(b,allow_nan=False))
        self.assertTrue(all(f['input_support']['row_flag_counts']['numeric_outside_training_range']==0 for f in a['folds']))
        a['folds'][0]['input_support']['features'][0]['name']='mutated'
        self.assertEqual(b['folds'][0]['input_support']['features'][0]['name'],'x')
        pd.testing.assert_frame_equal(X,before)

    def test_invalid_input_rejected(self):
        X,g=self.fixture()
        for kwargs in [{'outer_splits':True},{'outer_splits':7},{'random_state':True}]:
            with self.assertRaises(ValueError):outer_fold_input_support(X,g,**kwargs)
        with self.assertRaises(ValueError):outer_fold_input_support(X,g[:-1])

if __name__=='__main__':unittest.main()
