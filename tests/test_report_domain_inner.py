"""No-fit compatibility tests for opt-in inner-domain report wording."""
import copy
import hashlib
import html
import unittest

from groupbench.report import render_html
from test_report_reading_path import fixture, language_html, Document


def domain_fixture():
    result = fixture()
    result.manifest.update(schema_version='1.3', protocol='nested-group-outer-domain-inner-v1')
    result.manifest['config'].update(inner_strategy='leave_one_domain_out', domain='site',
                                    inner_splits=None, inner_splits_per_outer=[2, 3])
    result.manifest.update(n_rows=12, n_groups=6)
    result.manifest['config'].update(outer_splits=3, inner_splits_per_outer=[2, 3, 3])
    template = result.manifest['outer_folds'][0]
    result.manifest['outer_folds'] = []
    for index, domains in enumerate((['b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c'])):
        outer = copy.deepcopy(template)
        outer.update(fold=index, train_group_count=4, test_group_count=2, train_rows=8, test_rows=4)
        outer['inner_folds'] = [dict(train_domains=[d for d in domains if d != held],
                                     test_domains=[held], train_domain_count=len(domains)-1,
                                     test_domain_count=1, domain_overlap_count=0) for held in domains]
        result.manifest['outer_folds'].append(outer)
    return result


class DomainInnerReportTests(unittest.TestCase):
    def test_default_bytes_unchanged(self):
        expected = '11f4e8e3639001022d2c61215cdd517d77671defa5642a28cd049b5987770189'
        self.assertEqual(hashlib.sha256(render_html(fixture()).encode()).hexdigest(), expected)
        result = fixture()
        result.manifest['config']['inner_strategy'] = 'group_kfold'
        self.assertEqual(hashlib.sha256(render_html(result).encode()).hexdigest(), expected)

    def test_bilingual_scope_and_nullable_global_count(self):
        text = render_html(domain_fixture())
        en = language_html(text, 'en')
        zh = language_html(text, 'zh')
        for phrase in ('Outer group folds / inner domain folds', '3 / see split audit',
                       'within each outer-training partition', 'Outer evaluation still holds out groups, not domains',
                       'does not establish generalization to unseen domains'):
            self.assertIn(phrase, en)
        for phrase in ('外層群組折數 / 內層領域折數', '3 / 見切分稽核',
                       '在各外層訓練分割內', '外層評估仍保留群組，而非領域', '不證明對未見領域的泛化能力'):
            self.assertIn(phrase, zh)
        self.assertNotIn(' / None', text)

    def test_per_outer_counts_are_not_global_count(self):
        text = render_html(domain_fixture())
        self.assertIn('<td>0</td><td>2</td></tr>', text)
        self.assertIn('<td>0</td><td>3</td></tr>', text)
        self.assertIn('<th scope="col">', text)
        self.assertIn('Inner domain folds', text)

    def test_escaping_purity_and_offline(self):
        result = domain_fixture()
        attack = '</strong><script src="https://evil.invalid">&</script>'
        result.manifest['config']['domain'] = attack
        before = copy.deepcopy(result)
        text = render_html(result)
        self.assertEqual(before.manifest, result.manifest)
        self.assertEqual(before.summary, result.summary)
        self.assertIn(html.escape(attack, quote=True), text)
        self.assertNotIn('<script', text)
        self.assertFalse(any(tag in ('script', 'link', 'iframe', 'object') for tag, _ in Document(text).tags))
