"""Synthetic precision disclosure checks; not a measurement-resolution audit."""
import copy
import unittest

import numpy as np
import pandas as pd

from groupbench.data import validate_features
from groupbench.model import Candidate, make_pipeline
from groupbench.report import render_html
from test_report_reading_path import fixture, language_html


class InputPrecisionReportTests(unittest.TestCase):
    def test_bilingual_disclosure_is_visible_once(self):
        text = render_html(fixture())
        self.assertEqual(text.count('id="input-precision-heading"'), 1)
        self.assertLess(text.index('id="input-precision-heading"'), text.index('id="split-audit"'))
        for lang, phrases in (
            ('en', ('Input precision is not audited', 'float64 input does not guarantee',
                    'ExtraTrees uses float32', 'does not count lost distinctions',
                    'scaling cannot recover distinctions already lost')),
            ('zh', ('輸入精度未經稽核', 'float64 輸入不保證', '使用 float32 特徵值',
                    '不計算遺失的數值區別', '縮放無法恢復已遺失的數值區別')),
        ):
            for phrase in phrases:
                self.assertIn(phrase, language_html(text, lang))

    def test_disclosure_does_not_mutate_results_or_claim_a_diagnosis(self):
        result = fixture()
        before = copy.deepcopy(result)
        render_html(result)
        self.assertEqual(result.manifest, before.manifest)
        self.assertEqual(result.summary, before.summary)
        self.assertNotIn('input_precision', result.manifest)

    def test_large_integer_distinction_lost_upstream_cannot_be_recovered(self):
        source = pd.DataFrame({'x': np.array([2**53, 2**53 + 1], dtype=np.int64)})
        self.assertEqual(source.x.nunique(), 2)
        with self.assertRaisesRegex(ValueError, "representable in float64"):
            validate_features(source)
        # Precision lost by the caller before validation remains undetectable.
        normalized = validate_features(source.astype(float))
        self.assertEqual(normalized.x.nunique(), 1)
        pipeline = make_pipeline(normalized, Candidate('probe', 'dummy', {}), 0)
        transformed = pipeline.named_steps['preprocess'].fit_transform(normalized)
        self.assertEqual(len(np.unique(transformed)), 1)

    def test_float32_can_merge_distinctions_after_existing_preprocessing(self):
        source = pd.DataFrame({'x': [-3., -1., 1., 1. + 1e-8, 3.]})
        pipeline = make_pipeline(source, Candidate('probe', 'extra_trees',
            {'n_estimators': 1, 'min_samples_leaf': 1}), 0)
        transformed = pipeline.named_steps['preprocess'].fit_transform(source)
        self.assertEqual(len(np.unique(transformed)), 5)
        self.assertEqual(len(np.unique(transformed.astype(np.float32))), 4)


if __name__ == '__main__':
    unittest.main()
