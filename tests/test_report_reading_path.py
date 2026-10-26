"""Renderer-only tests: hand-authored values, no estimator fitting."""
import copy
from html.parser import HTMLParser
from types import SimpleNamespace
import unittest

from groupbench.report import render_html


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


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


if __name__ == '__main__':
    unittest.main()
