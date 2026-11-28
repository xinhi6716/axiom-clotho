import unittest

import pandas as pd

from groupbench.observed_history import evaluate_observed_history


class HistoryCutoffsTest(unittest.TestCase):
    def setUp(self):
        self.obs = pd.DataFrame(dict(trace_id=['x']*6, timestamp=[2, 3, 5, 8, 13, 20],
                                     value=[1, 3, 5, 11, 17, 31]))
        self.origins = pd.DataFrame(dict(trace_id=['x']*3, target_index=[3, 4, 5]))
        self.model = self.origins.assign(prediction=[4, 5, 6])
        self.schedule = self.origins.assign(timestamp=[8, 13, 20])
        self.cut = self.origins.assign(history_index=[1, 1, 1])

    def run_cutoff(self, cut, obs=None):
        return evaluate_observed_history(self.obs if obs is None else obs, origins=self.origins,
                                         saved_predictions=self.model, scheduled_origins=self.schedule,
                                         history_cutoffs=cut, timestamp_mode='integer_ticks')

    def test_open_loop_baselines_ignore_intervening_reads(self):
        r = self.run_cutoff(self.cut)
        self.assertEqual(r.summary['protocol'], 'observed_history_declared_cutoff_v1')
        self.assertEqual(r.predictions['lead'].tolist(), [2, 3, 4])
        self.assertEqual(r.predictions['index_continuation'].tolist(), [7., 9., 11.])
        self.assertEqual(r.predictions['elapsed_continuation'].tolist(), [13., 23., 37.])
        self.assertEqual(r.predictions['persistence'].tolist(), [3., 3., 3.])
        revised = self.obs.copy()
        revised.loc[2:4, 'value'] = -77
        changed = self.run_cutoff(self.cut, revised)
        for col in ('persistence', 'index_continuation', 'elapsed_continuation'):
            self.assertEqual(r.predictions[col].tolist(), changed.predictions[col].tolist())
        self.assertNotEqual(r.predictions['observed'].tolist(), changed.predictions['observed'].tolist())
        regular = self.run_cutoff(None)
        self.assertNotEqual(regular.predictions['persistence'].tolist(), r.predictions['persistence'].tolist())
        self.assertNotIn('lead', regular.predictions)

    def test_reject_unknown_or_noncausal_history(self):
        bad = [self.cut.iloc[:-1], self.cut.iloc[::-1], self.cut.assign(history_index=[1,3,5]),
               self.cut.assign(history_index=[0, 1, 1]), self.cut.assign(history_index=[1.,1.,1.]),
               self.cut.assign(history_index=[True,1,1])]
        for cut in bad:
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                self.run_cutoff(cut)
