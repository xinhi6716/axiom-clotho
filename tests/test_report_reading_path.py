"""Renderer-only structure/copy tests; not browser or assistive-technology checks."""
import copy
import html
import math
from pathlib import Path
import sys
import tempfile
import re
from html.parser import HTMLParser
from types import SimpleNamespace
import unittest

from groupbench.report import _number, render_html, write_report


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def language_html(text, language):
    """Project the renderer's escaped flat copy spans, not execute its CSS."""
    return re.sub(r'<span class="copy-(en|zh)" lang="(?:en|zh-Hant)">([^<]*)</span>',
                  lambda match: match[2] if match[1] == language else '', text)


class Cells(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.cells = []
        self.current = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag in ('td', 'th'):
            self.current = [tag, '']

    def handle_data(self, data):
        if self.current is not None:
            self.current[1] += data

    def handle_endtag(self, tag):
        if self.current is not None and tag == self.current[0]:
            self.cells.append(tuple(self.current))
            self.current = None


def fixture():
    metrics = dict(group_mae=1.5, mae=1.6, rmse=2.0, r2=-0.75, worst_group_mae=2.5)
    model = dict(overall=metrics, per_group=[dict(group='batch <one>', n_rows=2, mae=2.5, rmse=3., bias=-1.)])
    return SimpleNamespace(summary={'models': {'selected': copy.deepcopy(model), 'dummy': copy.deepcopy(model)}}, manifest={
        'config': dict(target='response', group='batch', outer_splits=2, inner_splits=2, random_state=0, selection_objective='group_mae'),
        'n_rows': 4, 'n_groups': 2, 'outer_folds': [dict(fold=0, train_group_count=1, test_group_count=1, train_rows=2, test_rows=2, selected_candidate='dummy', overlap_count=0)],
        'limitations': ['Candidate failure: keep this diagnostic visible.'], 'protocol': 'test-only', 'groupbench_version': 'test', 'elapsed_seconds': 0.0, 'data_sha256': 'test', 'run_key': 'test', 'created_at_utc': 'test'})


class ReadingPathTests(unittest.TestCase):
    def setUp(self):
        self.result = fixture()
        self.before = copy.deepcopy(self.result)
        self.text = render_html(self.result)
        self.document = Document(self.text)

    def _set_group_count(self, count):
        template = self.result.summary['models']['selected']['per_group'][0]
        self.result.summary['models']['selected']['per_group'] = [
            dict(copy.deepcopy(template), group=f'g-{i:05d}', mae=float(i % 3),
                 rmse=float(i % 3) + .5, bias=-i / 10, n_rows=i + 1)
            for i in reversed(range(count))
        ]

    @staticmethod
    def _body(text, caption):
        return text.split(f'<caption id="{caption}">', 1)[1].split('<tbody>', 1)[1].split('</tbody>', 1)[0]

    def test_all_group_toggle_uses_record_count_not_manifest_count(self):
        for count in (0, 1, 15, 16, 20000):
            with self.subTest(count=count):
                self.result = fixture()
                self._set_group_count(count)
                self.result.manifest['n_groups'] = 2 if count > 15 else 999
                text = render_html(self.result)
                self.assertEqual('<details class="all-groups">' in text, count > 15)
                self.assertEqual(self._body(text, 'groups-table').count('<tr>'), min(count, 15))
                if count > 15:
                    self.assertIn(f'<summary>View all {count} groups</summary>', language_html(text, 'en'))
                    self.assertIn(f'<summary>查看全部 {count} 個群組</summary>', language_html(text, 'zh'))
                    self.assertEqual(self._body(text, 'all-groups-table').count('<tr>'), count)
                    self.assertNotIn('<details class="all-groups" open', text)

    def test_all_groups_id_order_escaped_values_and_worst_order_are_exact(self):
        self._set_group_count(16)
        groups = self.result.summary['models']['selected']['per_group']
        names = ['2', '10', '群組<&"', 'é', 'e\u0301', '</th><script>alert(1)</script>']
        for row, name in zip(groups, names):
            row['group'] = name
        groups[-1]['group'] = '長' * 300
        text = render_html(self.result)
        ids = lambda body: re.findall(r'<th scope="row">(.*?)</th>', body)
        expected_all = sorted(groups, key=lambda row: row['group'])
        self.assertEqual(ids(self._body(text, 'all-groups-table')),
                         [html.escape(row['group'], quote=True) for row in expected_all])
        self.assertEqual(ids(self._body(text, 'groups-table')),
                         [html.escape(row['group'], quote=True) for row in sorted(groups, key=lambda r: -r['mae'])[:15]])
        for row in groups:
            cells = (f'<td>{row["n_rows"]}</td><td>{row["mae"]:,.4f}</td>'
                     f'<td>{row["rmse"]:,.4f}</td><td>{row["bias"]:,.4f}</td>')
            self.assertIn(cells, self._body(text, 'all-groups-table'))
        self.assertNotIn('<script', text)

    def test_all_groups_native_structure_and_repeat_render_do_not_mutate(self):
        self._set_group_count(16)
        before = copy.deepcopy(self.result)
        text = render_html(self.result)
        self.assertEqual(text, render_html(self.result))
        self.assertEqual(self.result.summary, before.summary)
        self.assertEqual(self.result.manifest, before.manifest)
        doc = Document(text)
        ids = [attrs['id'] for _, attrs in doc.tags if 'id' in attrs]
        self.assertEqual(len(ids), len(set(ids)))
        region = [attrs for _, attrs in doc.tags if attrs.get('aria-labelledby') == 'all-groups-table']
        self.assertEqual(len(region), 1)
        self.assertEqual(region[0]['role'], 'region')
        self.assertEqual(region[0]['tabindex'], '0')
        self.assertEqual(sum(tag == 'summary' for tag, _ in doc.tags), 1)
        self.assertIn('Selected procedure: all groups (ID order)', text)
        self.assertIn('summary:focus-visible', text)
        self.assertFalse(any(tag in ('script', 'iframe', 'form') for tag, _ in doc.tags))
        self.assertEqual([a['href'] for t, a in doc.tags if t == 'a'],
                         [a['href'] for t, a in self.document.tags if t == 'a'])

    def test_unique_ids_and_resolved_anchors(self):
        ids = [a['id'] for _, a in self.document.tags if 'id' in a]
        self.assertEqual(len(ids), len(set(ids)))
        anchors = [a['href'][1:] for t, a in self.document.tags if t == 'a' and a.get('href', '').startswith('#')]
        self.assertEqual(anchors, ['overview', 'overview', 'comparison', 'difficult-groups', 'split-audit', 'run-details'])
        self.assertTrue(set(anchors) <= set(ids))

    def test_tables_have_focusable_caption_associated_regions(self):
        captions = {a['id'] for t, a in self.document.tags if t == 'caption'}
        regions = [a for t, a in self.document.tags if a.get('role') == 'region']
        self.assertEqual(len(regions), 3)
        self.assertEqual({r['aria-labelledby'] for r in regions}, captions)
        self.assertTrue(all(r['tabindex'] == '0' for r in regions))
        self.assertIn(':focus-visible', self.text)
        self.assertIn('outline:3px solid', self.text)
        self.assertIn('.table-wrap{overflow:auto}', self.text)
        self.assertIn('@media(max-width:700px)', self.text)

    def test_fixed_local_evidence_links_only_and_no_active_assets(self):
        urls = [a['href'] for t, a in self.document.tags if t == 'a' and not a.get('href', '').startswith('#')]
        self.assertEqual(urls, ['summary.json', 'manifest.json', 'predictions.csv', 'checksums.json'])
        self.assertFalse(any(t in {'script', 'iframe', 'link', 'img', 'object', 'form'} for t, _ in self.document.tags))
        self.assertNotIn('url(', self.text)
        self.assertIn("default-src 'none'", self.text)

    def test_negative_undefined_and_failure_text_remain_visible(self):
        self.assertIn('-0.7500', self.text)
        self.assertIn('Candidate failure: keep this diagnostic visible.', self.text)
        self.result.summary['models']['dummy']['overall']['r2'] = None
        self.assertIn('undefined', render_html(self.result))

    def test_selection_and_limitations_are_not_promoted_to_winner(self):
        self.assertIn('Selection procedure', self.text)
        self.assertIn('chosen only on inner folds', self.text)
        self.assertIn('outer leaderboard is for evaluation, not another tuning set', self.text)
        self.assertIn('No confidence intervals are implied', self.text)
        self.assertNotIn('winner', self.text.lower())
        self.assertIn('Checksums record file bytes, not scientific validity', self.text)

    def test_input_labels_remain_escaped_and_cannot_change_links(self):
        attack = '<script src="https://evil.invalid">&../../x</script>'
        self.result.manifest['config']['target'] = attack
        self.result.manifest['config']['group'] = attack
        self.result.manifest['limitations'] = [attack]
        self.result.manifest['outer_folds'][0]['selected_candidate'] = attack
        self.result.summary['models'][attack] = self.result.summary['models'].pop('dummy')
        self.result.summary['models']['selected']['per_group'][0]['group'] = attack
        text = render_html(self.result)
        self.assertNotIn('<script', text)
        self.assertIn('&lt;script', text)
        self.assertFalse(any(t == 'script' or any(k.startswith('on') for k in a) for t, a in Document(text).tags))
        self.assertEqual([a['href'] for t, a in Document(text).tags if t == 'a'], [a['href'] for t, a in self.document.tags if t == 'a'])

    def test_empty_group_and_fold_lists_render_without_invented_rows(self):
        self.result.summary['models']['selected']['per_group'] = []
        self.result.manifest['outer_folds'] = []
        text = render_html(self.result)
        self.assertEqual(text.count('<tbody></tbody>'), 2)
        self.assertNotIn('PASS', text)

    def test_render_does_not_mutate_inputs(self):
        self.assertEqual(self.result.summary, self.before.summary)
        self.assertEqual(self.result.manifest, self.before.manifest)

    def test_split_scope_precedes_comparison_scores_without_claiming_domain_validity(self):
        scope = self.text.index('id="split-scope-heading"')
        self.assertLess(scope, self.text.index('class="bar-row"'))
        self.assertLess(scope, self.text.index('<caption id="comparison-table">'))
        for expected in ('Whole-group holdout on: <code>batch</code>',
                         'Zero group-ID overlap checks only the supplied IDs',
                         'other domain/site axes are not separately audited',
                         'If that column defines the intended domain boundary',
                         'Random-row comparison?',
                         'Not run. The seed assigns whole groups to folds',
                         'may still share a site, time period or another dependence axis'):
            self.assertIn(expected, language_html(self.text, 'en'))
        self.assertNotIn('batch it has never seen', self.text)

    def test_singleton_note_requires_positive_integer_equal_counts(self):
        for rows, groups, expected in ((4, 4, True), (4, 2, False), (4, 3, False),
                                       (0, 0, False), (True, True, False), (4.0, 4.0, False)):
            with self.subTest(rows=rows, groups=groups):
                self.result.manifest.update(n_rows=rows, n_groups=groups)
                text = render_html(self.result)
                self.assertEqual('Recorded rows = groups:' in text, expected)
                if expected:
                    self.assertIn('every supplied group has one row', text)
                    self.assertIn('check that related rows have not been assigned different IDs', text)
                self.assertNotIn('equivalent to a random', text)

    def test_domain_named_group_is_displayed_not_inferred_or_certified(self):
        self.result.manifest['config']['group'] = 'domain'
        text = render_html(self.result)
        self.assertIn('Whole-group holdout on: <code>domain</code>', language_html(text, 'en'))
        self.assertIn('the scientific meaning of the IDs is not verified', text)
        self.assertIn('If that column defines the intended domain boundary', text)
        self.assertNotIn('This is not domain-heldout', text)

    def test_old_manifest_scope_escapes_long_unicode_group_without_mutation(self):
        import html
        name = '群組<&"' + '域' * 160
        self.result.manifest['config']['group'] = name
        self.result.manifest.pop('package_source', None)
        before = copy.deepcopy(self.result)
        text = render_html(self.result)
        self.assertIn('<code>' + html.escape(name, quote=True) + '</code>', text)
        self.assertNotIn(name, text)
        self.assertIn('not recorded', text)
        self.assertEqual(self.result.manifest, before.manifest)
        self.assertEqual(self.result.summary, before.summary)

    def test_native_language_controls_and_static_css_contract(self):
        radios = [a for tag, a in self.document.tags if tag == 'input']
        self.assertEqual(len(radios), 2)
        self.assertTrue(all(a['type'] == 'radio' and a['name'] == 'report-language' for a in radios))
        self.assertEqual([a['value'] for a in radios], ['en', 'zh-Hant'])
        self.assertEqual(['checked' in a for a in radios], [True, False])
        labels = {a['for']: a['lang'] for tag, a in self.document.tags if tag == 'label'}
        self.assertEqual(labels, {'report-en': 'en', 'report-zh': 'zh-Hant'})
        self.assertTrue(all(a['aria-describedby'] == 'language-help' for a in radios))
        # Two class selectors outrank the existing .evidence-links span rule.
        for expected in ('.report-content .copy-zh{display:none}',
                         '#report-zh:checked ~ .report-content .copy-en{display:none}',
                         '#report-zh:checked ~ .report-content .copy-zh{display:inline}',
                         'input:focus-visible', '.language-title,.language-choice,.language-label{display:none}'):
            self.assertIn(expected, self.text)
        self.assertLess(self.text.index('id="report-zh"'), self.text.index('<div class="report-content">'))

    def test_reading_path_and_limits_have_both_language_copies(self):
        english, chinese = [language_html(self.text, language) for language in ('en', 'zh')]
        for copy_text in ('Baseline comparison', 'Where it struggles', 'Split audit',
                          'Reproduce this run', 'Scope &amp; limitations', 'Keep the evidence close'):
            self.assertIn(copy_text, english)
        for copy_text in ('基準模型比較', '表現較差的群組', '切分稽核', '重做此執行',
                          '範圍與限制', '保留完整證據', '群組等權 MAE',
                          '僅在內層折中選擇', '不表示提供信賴區間', '未另稽核其他領域／場域軸',
                          '檢查碼記錄檔案位元組，不代表科學有效性'):
            self.assertIn(copy_text, chinese)
        self.assertNotIn('Baseline comparison', chinese)
        self.assertNotIn('基準模型比較', english)
        self.assertIn('Candidate failure: keep this diagnostic visible.', chinese)

    def test_both_languages_share_exact_numeric_cells_and_group_order(self):
        self._set_group_count(16)
        self.result.summary['models']['dummy']['overall']['r2'] = None
        text = render_html(self.result)
        for caption in ('comparison-table', 'groups-table', 'all-groups-table', 'audit-table'):
            with self.subTest(caption=caption):
                en = Cells(self._body(language_html(text, 'en'), caption)).cells
                zh = Cells(self._body(language_html(text, 'zh'), caption)).cells
                self.assertEqual(en, [(tag, 'undefined' if tag == 'td' and value == '未定義' else value)
                                      for tag, value in zh])
        self.assertIn(('td', 'undefined'), Cells(self._body(language_html(text, 'en'), 'comparison-table')).cells)
        self.assertIn(('td', '未定義'), Cells(self._body(language_html(text, 'zh'), 'comparison-table')).cells)

    def test_names_and_recorded_diagnostics_remain_verbatim_in_both_views(self):
        name = 'Rows 群組<&"'
        self.result.manifest['config']['target'] = name
        self.result.manifest['config']['group'] = name
        self.result.manifest['limitations'] = [name]
        self.result.summary['models'][name] = self.result.summary['models'].pop('dummy')
        self.result.summary['models']['selected']['per_group'][0]['group'] = name
        text = render_html(self.result)
        for language in ('en', 'zh'):
            view = language_html(text, language)
            self.assertIn('<strong>' + html.escape(name, quote=True) + '</strong>', view)
            self.assertIn('<li>' + html.escape(name, quote=True) + '</li>', view)
            self.assertIn('<th scope="row">' + html.escape(name, quote=True) + '</th>', view)
        self.assertNotIn(name, text)

    def test_old_and_unavailable_source_records_are_bilingual(self):
        for source, english, chinese in ((None, 'not recorded', '未記錄'),
                                          ({'status': 'unavailable'}, 'unavailable', '無法取得')):
            with self.subTest(source=source):
                self.result.manifest.pop('package_source', None)
                if source is not None:
                    self.result.manifest['package_source'] = source
                text = render_html(self.result)
                self.assertIn('<code>' + english + '</code>', language_html(text, 'en'))
                self.assertIn('<code>' + chinese + '</code>', language_html(text, 'zh'))


    def test_small_number_format_preserves_zero_threshold_and_normal_range(self):
        smallest = math.nextafter(0.0, 1.0)
        below = math.nextafter(0.0001, 0.0)
        for value, expected in ((smallest, "4.9407e-324"),
                                (-smallest, "-4.9407e-324"),
                                (sys.float_info.min, "2.2251e-308"),
                                (-sys.float_info.min, "-2.2251e-308"),
                                (below, "1.0000e-04"), (-below, "-1.0000e-04"),
                                (1e-10, "1.0000e-10"), (-1e-10, "-1.0000e-10")):
            with self.subTest(value=value):
                self.assertEqual(_number(value), expected)
                self.assertNotEqual(float(_number(value)), 0.0)
        for value in (0.0, -0.0, 0.0001, -0.0001,
                      math.nextafter(0.0001, math.inf),
                      -math.nextafter(0.0001, math.inf),
                      0.25, -0.75, 1234.56789, sys.float_info.max):
            with self.subTest(value=value):
                self.assertEqual(_number(value), f"{value:,.4f}")
        self.assertEqual(language_html(_number(None), 'en'), 'undefined')
        self.assertEqual(language_html(_number(None), 'zh'), '未定義')

    def test_small_metrics_and_negative_r2_share_bilingual_number_text(self):
        self._set_group_count(16)
        self.result.summary['models']['selected']['overall'].update(
            group_mae=1e-9, mae=1e-9, rmse=2e-9, worst_group_mae=1e-9, r2=-1e-12)
        for row in self.result.summary['models']['selected']['per_group']:
            row.update(mae=1e-9, rmse=2e-9, bias=-3e-9)
        before = copy.deepcopy(self.result)
        text = render_html(self.result)
        en, zh = [language_html(text, language) for language in ('en', 'zh')]
        self.assertIn('<span class="value">1.0000e-09</span>', text)
        self.assertIn('<td>-1.0000e-12</td>', self._body(text, 'comparison-table'))
        self.assertIn('scientific notation', en)
        self.assertIn('非零數值以科學記號顯示', zh)
        for caption in ('comparison-table', 'groups-table', 'all-groups-table', 'audit-table'):
            self.assertEqual(Cells(self._body(en, caption)).cells,
                             Cells(self._body(zh, caption)).cells)
        self.assertIn('<td>-3.0000e-09</td>', self._body(text, 'all-groups-table'))
        self.assertEqual(self.result.summary, before.summary)
        self.assertEqual(self.result.manifest, before.manifest)

    def test_nonfinite_metrics_still_fail_before_bundle_write(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                self.result.summary['models']['selected']['overall']['r2'] = value
                output = Path(directory) / 'new'
                with self.assertRaises(ValueError):
                    write_report(self.result, output)
                self.assertFalse(output.exists())

if __name__ == '__main__':
    unittest.main()
