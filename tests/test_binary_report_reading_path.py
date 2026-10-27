"""Renderer structure and copy contracts; not browser/assistive-technology tests."""
import copy
from html.parser import HTMLParser
from types import SimpleNamespace
import unittest

from groupbench.binary.report import render_binary_html


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def fixture():
    model = {
        'overall': dict(group_brier=0.2, brier=0.3, group_log_loss_clipped=0.4,
                        accuracy=0.5, confusion=dict(fn=1, fp=2), n_true_0=4, n_true_1=2),
        'per_group': [
            dict(group='no-positive', n_rows=4, n_true_0=4, n_true_1=0, brier=0.2,
                 log_loss_clipped=0.4, confusion=dict(fn=0, fp=2), recall_1=None, specificity_0=0.5),
            dict(group='no-negative', n_rows=2, n_true_0=0, n_true_1=2, brier=0.1,
                 log_loss_clipped=0.3, confusion=dict(fn=1, fp=0), recall_1=0.5, specificity_0=None),
        ],
    }
    return SimpleNamespace(summary={
        'models': {'selected': copy.deepcopy(model), 'dummy': copy.deepcopy(model)},
        'adverse_comparison_to_dummy': dict(available=True, group_brier_delta=-0.1,
                                          groups_better=1, groups_tied=1, groups_worse=0),
    }, manifest={
        'config': dict(target='fail', group='lot', random_state=0),
        'n_rows': 6, 'n_groups': 2,
        'outer_folds': [dict(fold=0, train_group_count=1, test_group_count=1,
                             selected_candidate='dummy', overlap_count=0)],
        'limitations': ['Synthetic fixture only.', 'Retain this recorded diagnostic.'],
        'protocol': 'renderer-only-test', 'fit_budget': {'total_fits': 0},
        'run_key': 'test-run', 'created_at_utc': 'test-time',
    })


class BinaryReportReadingPathTests(unittest.TestCase):
    def setUp(self):
        self.result = fixture()
        self.before = copy.deepcopy(self.result)
        self.text = render_binary_html(self.result)
        self.document = Document(self.text)

    def test_split_scope_precedes_scores_in_both_languages(self):
        self.assertLess(self.text.index('id="split-scope-heading"'),
                        self.text.index('<caption id="comparison-table">'))
        for expected in ('Whole-group holdout on: ', '<code>lot</code>',
                         'Zero group-ID overlap checks only the supplied IDs',
                         'other domain/site axes are not separately audited',
                         'If that column defines the intended domain boundary',
                         'Not run. The seed assigns whole groups to folds',
                         '此切分可回答的問題', '未另稽核其他領域／場域軸',
                         '若該欄位定義預定的領域邊界',
                         '未執行。種子將完整群組分配至各折',
                         '不同群組識別碼仍可能共享場域、時段或其他相依軸'):
            self.assertIn(expected, self.text)

    def test_singleton_note_requires_positive_integer_equal_counts(self):
        for rows, groups, expected in ((6, 6, True), (6, 2, False), (6, 4, False),
                                       (0, 0, False), (True, True, False), (6.0, 6.0, False)):
            with self.subTest(rows=rows, groups=groups):
                self.result.manifest.update(n_rows=rows, n_groups=groups)
                text = render_binary_html(self.result)
                self.assertEqual('Recorded rows = groups:' in text, expected)
                self.assertEqual('記錄列數等於群組數' in text, expected)
                if expected:
                    self.assertIn('every supplied group has one row', text)
                    self.assertIn('請檢查相關列是否被指定為不同識別碼', text)
                self.assertNotIn('equivalent to a random', text)

    def test_domain_named_group_is_displayed_not_inferred_or_certified(self):
        self.result.manifest['config']['group'] = 'domain'
        text = render_binary_html(self.result)
        self.assertIn('<code>domain</code>', text)
        self.assertIn('the scientific meaning of the IDs is not verified', text)
        self.assertIn('識別碼的科學意義並未經驗證', text)
        self.assertNotIn('This is not domain-heldout', text)

    def test_old_manifest_scope_escapes_long_unicode_group_without_mutation(self):
        import html
        name = '群組<&"' + '域' * 160
        self.result.manifest['config']['group'] = name
        self.result.manifest.pop('package_source', None)
        before = copy.deepcopy(self.result)
        text = render_binary_html(self.result)
        self.assertIn('<code>' + html.escape(name, quote=True) + '</code>', text)
        self.assertNotIn(name, text)
        self.assertIn('未記錄', text)
        self.assertEqual(self.result.manifest, before.manifest)
        self.assertEqual(self.result.summary, before.summary)

    def test_ids_and_all_accessible_references_resolve(self):
        ids = [attrs['id'] for _, attrs in self.document.tags if 'id' in attrs]
        self.assertEqual(len(ids), len(set(ids)))
        for tag, attrs in self.document.tags:
            for name in ('aria-labelledby', 'aria-describedby'):
                for reference in attrs.get(name, '').split():
                    self.assertIn(reference, ids)
            if tag == 'a' and attrs['href'].startswith('#'):
                self.assertIn(attrs['href'][1:], ids)
        anchors = [a['href'] for tag, a in self.document.tags if tag == 'a' and a['href'].startswith('#')]
        self.assertEqual(anchors, ['#overview', '#overview', '#comparison', '#difficult-groups', '#split-audit', '#run-details'])

    def test_native_language_group_defaults_to_english_without_scripts(self):
        controls = [attrs for tag, attrs in self.document.tags if tag == 'input']
        self.assertEqual([c['type'] for c in controls], ['radio', 'radio'])
        self.assertEqual({c['name'] for c in controls}, {'report-language'})
        self.assertEqual([c['id'] for c in controls if 'checked' in c], ['report-en'])
        labels = {attrs['for'] for tag, attrs in self.document.tags if tag == 'label'}
        self.assertEqual(labels, {c['id'] for c in controls})
        self.assertIn('.copy-zh{display:none}', self.text)
        self.assertIn('#report-zh:checked ~ .report-content .copy-en{display:none}', self.text)
        self.assertIn('#report-zh:checked ~ .report-content .copy-zh{display:inline}', self.text)
        self.assertTrue(any(a.get('lang') == 'zh-Hant' for _, a in self.document.tags))
        self.assertIn('does not change scores, saved data or the CLI language', self.text)

    def test_focusable_tables_have_captions_and_header_scopes(self):
        captions = {a['id'] for tag, a in self.document.tags if tag == 'caption'}
        regions = [a for _, a in self.document.tags if a.get('role') == 'region']
        self.assertEqual(len(regions), 3)
        self.assertEqual({r['aria-labelledby'] for r in regions}, captions)
        self.assertTrue(all(r['tabindex'] == '0' for r in regions))
        self.assertTrue(all(a.get('scope') in ('row', 'col') for tag, a in self.document.tags if tag == 'th'))
        self.assertIn('input:focus-visible', self.text)
        self.assertIn('outline:3px solid', self.text)

    def test_narrow_screen_rules_contain_tables_not_the_whole_page(self):
        self.assertIn('.scroll{overflow:auto;', self.text)
        self.assertIn('grid-template-columns:repeat(3,minmax(0,1fr))', self.text)
        self.assertIn('@media(max-width:700px)', self.text)
        self.assertIn('.cards,.evidence-links{grid-template-columns:1fr}', self.text)
        self.assertIn('keyboard users can focus each table', self.text)
        self.assertIn('overflow-wrap:anywhere', self.text)

    def test_no_class_denominator_is_explicit_not_zero_or_missing_prediction(self):
        self.assertIn('undefined: no label 1', self.text)
        self.assertIn('undefined: no label 0', self.text)
        self.assertIn('未定義：沒有標籤 1', self.text)
        self.assertIn('未定義：沒有標籤 0', self.text)
        self.assertIn('not zero performance or a missing prediction', self.text)
        self.assertIn('no-positive', self.text)
        self.assertIn('no-negative', self.text)
        self.result.summary['models']['selected']['per_group'][0]['recall_1'] = 0.0
        changed = render_binary_html(self.result)
        self.assertNotIn('undefined: no label 1', changed)
        self.assertIn('<td>0</td>', changed)

    def test_calibration_and_selection_boundaries_in_both_languages(self):
        for text in ('Held-out evaluation · not probability calibration',
                     '保留群組評估 · 並非機率校準', 'Selection procedure', '模型選擇程序',
                     'chosen only inside training folds', '僅在訓練折內選擇',
                     'not another tuning step', '不是另一輪調參',
                     'not established calibrated failure risks', '尚未證實為已校準的失敗風險',
                     'p1 ≥ 0.5', 'Descriptive, not significance.',
                     'not real-world accuracy', '不代表真實世界準確度'):
            self.assertIn(text, self.text)
        self.assertNotIn('winner', self.text.lower())

    def test_fixed_bundle_links_no_active_or_external_assets(self):
        urls = [a['href'] for tag, a in self.document.tags if tag == 'a' and not a['href'].startswith('#')]
        self.assertEqual(urls, ['summary.json', 'manifest.json', 'predictions.csv', 'checksums.json'])
        self.assertFalse(any(tag in {'script', 'iframe', 'link', 'img', 'object', 'form'} for tag, _ in self.document.tags))
        self.assertFalse(any(any(key.startswith('on') for key in a) for _, a in self.document.tags))
        self.assertNotIn('url(', self.text)
        self.assertIn("default-src 'none'", self.text)

    def test_source_names_and_diagnostics_are_not_translated_or_hidden(self):
        self.assertIn('Names and recorded diagnostics below are kept as supplied', self.text)
        self.assertIn('<li>Retain this recorded diagnostic.</li>', self.text)
        self.assertIn('<th scope="row">no-positive</th>', self.text)
        self.assertIn('-0.1', self.text)
        self.assertEqual(self.result.summary, self.before.summary)
        self.assertEqual(self.result.manifest, self.before.manifest)

    def test_absent_dummy_comparison_retains_reason_without_invented_score(self):
        self.result.summary['adverse_comparison_to_dummy'] = dict(available=False, reason='Dummy was not requested.')
        text = render_binary_html(self.result)
        self.assertIn('Comparison unavailable; recorded reason:', text)
        self.assertIn('Dummy was not requested.', text)
        self.assertNotIn('Selected minus Dummy group Brier:', text)

    def test_user_strings_cannot_add_controls_links_or_markup(self):
        attack = '<input id="report-en"><script src="https://evil.invalid">&</script>'
        self.result.manifest['config']['target'] = attack
        self.result.manifest['config']['group'] = attack
        self.result.manifest['limitations'] = [attack]
        self.result.manifest['outer_folds'][0]['selected_candidate'] = attack
        self.result.summary['models']['selected']['per_group'][0]['group'] = attack
        self.result.summary['models'][attack] = self.result.summary['models'].pop('dummy')
        self.result.summary['adverse_comparison_to_dummy'] = dict(available=False, reason=attack)
        text = render_binary_html(self.result)
        self.assertIn('&lt;script', text)
        self.assertNotIn('<script', text)
        doc = Document(text)
        self.assertEqual([a for tag, a in doc.tags if tag in ('input', 'a')],
                         [a for tag, a in self.document.tags if tag in ('input', 'a')])


if __name__ == '__main__':
    unittest.main()
