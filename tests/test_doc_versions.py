"""No model imports, network access or fits; only documentation contract tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('doc_versions', ROOT / 'tools/check_doc_versions.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class DocVersionsTests(unittest.TestCase):
    def setUp(self):
        self.contract = CHECK.load_contract(ROOT / 'tools/doc_version_contract.json')
        self.text = (ROOT / 'docs/USAGE_REFERENCE.md').read_text(encoding='utf-8')

    def reject(self, text):
        with self.assertRaises(ValueError):
            CHECK.check_text(text, self.contract)

    def test_checkout(self):
        self.assertEqual(CHECK.check(ROOT)['status'], 'pass')

    def test_source_feature_wrongly_given_to_download(self):
        self.reject(self.text.replace('| absent | present |', '| present | present |', 1))

    def test_scope_changed_to_frozen(self):
        self.reject(self.text.replace('| current-source |', '| v0.2.0a2 |'))

    def test_frozen_revision_changed_to_current(self):
        self.reject(self.text.replace(self.contract['scope']['frozen_source'], self.contract['scope']['fixes_snapshot']))

    def test_missing_marker(self):
        self.reject(self.text.replace(CHECK.START, ''))

    def test_duplicate_marker(self):
        self.reject(self.text + CHECK.START)

    def test_reversed_markers(self):
        self.reject(self.text.replace(CHECK.START, 'SWAP').replace(CHECK.END, CHECK.START).replace('SWAP', CHECK.END))

    def test_missing_capability(self):
        row = '| reject-bare-string-api-features | absent | present |\n'
        self.reject(self.text.replace(row, ''))

    def test_duplicate_capability(self):
        row = '| reject-bare-string-api-features | absent | present |\n'
        self.reject(self.text.replace(row, row + row))

    def test_unknown_field(self):
        self.reject(self.text.replace('| frozen_tag |', '| package_version |'))

    def test_legacy_status_header_reinserted(self):
        self.reject('**Status:** v0.2.0a2 includes all current fixes.\n' + self.text)

    def test_legal_history_and_other_revisions(self):
        CHECK.check_text(self.text + '\n## History\n**Status:** v0.2.0a1 was released.\nCurrent HEAD: ' + 'a' * 40, self.contract)

    def test_negative_example_in_code_is_not_status(self):
        CHECK.check_text('```markdown\n**Status:** old incorrect text\n' + CHECK.START + '\n```\n' + self.text, self.contract)

    def test_quoted_history_is_not_status(self):
        CHECK.check_text('> **Status:** historical quote\n' + self.text, self.contract)

    def test_tables_only_in_code_do_not_count(self):
        self.reject('```markdown\n' + self.text + '\n```')

    def test_spacing_inline_code_row_order(self):
        text = self.text.replace('| absent | present |', '|  `absent`  |  `present`  |')
        row1 = '| frozen_tag | v0.2.0a2 |'
        row2 = '| documentation_scope | current-source |'
        text = text.replace(row1, 'SWAP').replace(row2, row1).replace('SWAP', row2)
        CHECK.check_text(text, self.contract)

    def test_contract_rejects_duplicate_key(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'bad.json'
            p.write_text('{"schema": 1, "schema": 1}')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
                CHECK.load_contract(p)

    def test_contract_rejects_unknown_state_and_bad_revision(self):
        for key in ('state', 'revision', 'schema'):
            data = copy.deepcopy(self.contract)
            if key == 'state':
                next(iter(data['features'].values()))['source'] = 'maybe'
            elif key == 'revision':
                data['scope']['frozen_source'] = 'main'
            else:
                data['schema'] = True
            with tempfile.TemporaryDirectory() as directory:
                p = Path(directory) / 'bad.json'
                p.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    CHECK.load_contract(p)


if __name__ == '__main__':
    unittest.main()
