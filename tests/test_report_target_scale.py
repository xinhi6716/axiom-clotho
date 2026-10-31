"""Synthetic reading-path checks, not unit inference or physical validation."""
import copy
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from groupbench import benchmark
from groupbench.evaluation_schema import validate_evaluation_bundle
from groupbench.report import render_html
from test_report_reading_path import fixture, language_html


class TargetScaleReportTests(unittest.TestCase):
    def test_bilingual_scope_appears_once_before_metrics(self):
        text = render_html(fixture())
        self.assertEqual(text.count('id="target-scale-heading"'), 1)
        self.assertLess(text.index('id="target-scale-heading"'), text.index('<caption id="comparison-table">'))
        for lang, heading, scale, caveat in (
            ('en', 'Target scale is not verified', 'input target scale', 'No automatic target-unit conversion or inverse transformation is performed.'),
            ('zh', '目標尺度未經核實', '輸入目標尺度', '不會自動換算目標單位或反轉換。'),
        ):
            rendered = language_html(text, lang)
            for expected in (heading, scale, caveat):
                self.assertIn(expected, rendered)
        self.assertNotIn('Lower is better · target units', text)

    def test_endpoint_name_does_not_certify_units_and_remains_escaped(self):
        result = fixture()
        result.manifest['config']['target'] = 'cv_percent<unknown>&ratio'
        text = render_html(result)
        self.assertIn('cv_percent&lt;unknown&gt;&amp;ratio', text)
        self.assertNotIn('cv_percent<unknown>', text)
        self.assertIn('Target scale is not verified', text)
        self.assertIn('ratio versus percent', text)
        self.assertIn('physical comparability or model readiness', text)

    def test_render_does_not_change_saved_inputs_or_require_new_fields(self):
        result = fixture()
        before = copy.deepcopy(result)
        render_html(result)
        self.assertEqual(result.summary, before.summary)
        self.assertEqual(result.manifest, before.manifest)
        self.assertNotIn('target_unit', result.manifest['config'])
        self.assertNotIn('target_scale', result.manifest['config'])


class SyntheticScaleWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        frame = pd.DataFrame({'batch': np.repeat(list('abcdef'), 2),
                              'x': np.tile([0., 1.], 6),
                              'cv': np.arange(12) / 100 + .05})
        for scale in ('ratio', 'percent', 'mixed'):
            data = frame.copy()
            if scale == 'percent':
                data['cv'] *= 100
            elif scale == 'mixed':
                data.loc[data['batch'].isin(list('abc')), 'cv'] *= 100
            cls.results[scale] = benchmark(data, target='cv', group='batch', features=['x'],
                outer_splits=3, inner_splits=2, baselines=('dummy',))

    def test_known_ratio_percent_conversion_changes_errors_not_r2(self):
        ratio = self.results['ratio'].summary['models']['selected']['overall']
        percent = self.results['percent'].summary['models']['selected']['overall']
        for key in ('mae', 'rmse', 'group_mae', 'worst_group_mae'):
            self.assertAlmostEqual(percent[key], ratio[key] * 100)
        self.assertAlmostEqual(percent['r2'], ratio['r2'])

    def test_mixed_numeric_scales_are_not_silently_converted_or_certified(self):
        result = self.results['mixed']
        observed = result.predictions['observed'].to_numpy()
        self.assertGreater(observed[0], 1)
        self.assertLess(observed[-1], 1)
        self.assertTrue(all(f['overlap_count'] == 0 for f in result.manifest['outer_folds']))
        self.assertIn('Target scale is not verified', render_html(result))
        self.assertIn('same physical endpoint and a common scale', render_html(result))

    def test_writer_keeps_schema_metrics_and_numeric_predictions_unchanged(self):
        for scale, result in self.results.items():
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as directory:
                before_summary = copy.deepcopy(result.summary)
                before_manifest = copy.deepcopy(result.manifest)
                before_predictions = result.predictions.copy(deep=True)
                result.write(Path(directory) / 'run')
                checked = validate_evaluation_bundle(Path(directory) / 'run')
                self.assertEqual((checked.n_rows, checked.n_groups), (12, 6))
                self.assertEqual(result.summary, before_summary)
                self.assertEqual(result.manifest, before_manifest)
                pd.testing.assert_frame_equal(result.predictions, before_predictions)
                self.assertIn('Target scale is not verified', (Path(directory) / 'run/report.html').read_text())


if __name__ == '__main__':
    unittest.main()
