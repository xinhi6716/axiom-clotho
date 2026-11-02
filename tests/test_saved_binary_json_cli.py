"""No-fit strict JSON and stdout-only saved-binary command checks."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from groupbench.binary.saved_evaluation import (
    MAX_JSON_BYTES, evaluate_saved_binary_probabilities, load_saved_binary_document, main,
)
from test_saved_binary_evaluation import document


def source_with(field, token):
    doc = document(); doc['rows'][0][field] = 'TOKEN_PLACEHOLDER'
    return json.dumps(doc).replace('"TOKEN_PLACEHOLDER"', token)


class SavedBinaryJsonCliTests(unittest.TestCase):
    def test_standard_json_matches_plain_document_and_no_type_inference(self):
        doc = document()
        loaded = load_saved_binary_document(json.dumps(doc))
        self.assertEqual(loaded, doc)
        self.assertEqual(evaluate_saved_binary_probabilities(loaded), evaluate_saved_binary_probabilities(doc))
        self.assertEqual(loaded['rows'][0]['group'], 'A')

    def test_probability_source_domain_and_endpoint_collapse(self):
        for token in ['-1e-400', '1.0000000000000000000001', '1e-400',
                      '0.9999999999999999999999', '1e309']:
            with self.subTest(token=token), self.assertRaises(ValueError):
                load_saved_binary_document(source_with('probability_1', token))
        for token, expected in [('0.1', .1), ('1e-100', 1e-100), ('0e-400', 0.), ('1.0', 1.)]:
            loaded = load_saved_binary_document(source_with('probability_1', token))
            self.assertEqual(loaded['rows'][0]['probability_1'], expected)
            self.assertIs(type(loaded['rows'][0]['probability_1']), float)

    def test_labels_checked_before_decimal_rounding(self):
        for token in ['0.0000000000000000000001', '0.9999999999999999999999', '1.0000000000000000000001']:
            with self.assertRaises(ValueError): load_saved_binary_document(source_with('observed_label', token))
        for token, expected in [('0.0', 0), ('1e0', 1), ('-0.0', 0)]:
            loaded = load_saved_binary_document(source_with('observed_label', token))
            self.assertIs(type(loaded['rows'][0]['observed_label']), int)
            self.assertEqual(loaded['rows'][0]['observed_label'], expected)

    def test_numeric_group_identity_preserved(self):
        for token in ['0.1', '0.10000000000000000001', '1e-400', '1e100']:
            with self.assertRaises(ValueError): load_saved_binary_document(source_with('group', token))
        for token, expected in [('9007199254740993.0', 9007199254740993), ('1.5', 1.5), ('1e3', 1000)]:
            loaded = load_saved_binary_document(source_with('group', token))
            self.assertEqual(loaded['rows'][0]['group'], expected)
        loaded = load_saved_binary_document(source_with('group', '"0.10000000000000000001"'))
        self.assertEqual(loaded['rows'][0]['group'], '0.10000000000000000001')

    def test_duplicates_rejected_at_every_object_level(self):
        text = json.dumps(document())
        cases = [text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'),
                 text.replace('"group": "A"', '"group": "A", "group": "A"', 1),
                 text.replace('"target_definition":', '"target_definition": "duplicate", "target_definition":', 1)]
        for source in cases:
            with self.assertRaisesRegex(ValueError, 'Duplicate'): load_saved_binary_document(source)

    def test_numeric_tokens_and_nonfinite_constants_bounded(self):
        for token in ['1' * 101, '0.' + '1' * 64, '1e-401', '1e9999999999999999999999', 'NaN', 'Infinity', '-Infinity']:
            with self.subTest(token=token[:20]), self.assertRaises(ValueError):
                load_saved_binary_document(source_with('probability_1', token))
        for token in ['true', 'null', '"0.5"']:
            parsed = load_saved_binary_document(source_with('probability_1', token))
            with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(parsed)

    def test_schema_decimal_and_numeric_row_ids_not_coerced(self):
        text = json.dumps(document()).replace('"schema_version": 1', '"schema_version": 1.0')
        with self.assertRaises(ValueError): evaluate_saved_binary_probabilities(load_saved_binary_document(text))
        with self.assertRaises(ValueError):
            evaluate_saved_binary_probabilities(load_saved_binary_document(source_with('row_id', '1')))

    def test_depth_limit_ignores_escaped_string_brackets(self):
        for text in ['[' * 17 + '0' + ']' * 17, '{"a":' * 17 + '0' + '}' * 17]:
            with self.assertRaisesRegex(ValueError, 'nesting'): load_saved_binary_document(text)
        doc = document(); doc['metadata']['prediction_definition'] = '[{\\"}]' * 30
        loaded = load_saved_binary_document(json.dumps(doc))
        self.assertEqual(loaded['metadata'], doc['metadata'])

    def test_utf8_payload_and_row_bounds(self):
        for text in [' ' * (MAX_JSON_BYTES + 1), '界' * (MAX_JSON_BYTES // 3 + 1), '\ud800']:
            with self.assertRaises(ValueError): load_saved_binary_document(text)
        doc = document(); doc['rows'] = []
        with self.assertRaises(ValueError): load_saved_binary_document(json.dumps(doc))
        doc['rows'] = [None] * 20001
        with self.assertRaises(ValueError): load_saved_binary_document(json.dumps(doc))

    def test_malformed_json_error_does_not_echo_source(self):
        for text in ['{"PRIVATE_SECRET":', '{', ']', 'true', '', '{"x": "PRIVATE_SECRET\n"}']:
            with self.assertRaises(ValueError) as caught: load_saved_binary_document(text)
            self.assertNotIn('PRIVATE_SECRET', str(caught.exception))

    def test_main_complete_stdout_only_and_input_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'input.json'; raw = json.dumps(document()).encode(); path.write_bytes(raw)
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                code = main([str(path)])
            self.assertEqual(code, 0); self.assertEqual(err.getvalue(), '')
            result = json.loads(out.getvalue())
            self.assertEqual(result['cohort']['scored_rows'], 4)
            self.assertEqual(list(root.iterdir()), [path]); self.assertEqual(path.read_bytes(), raw)

    def test_main_failure_has_no_partial_stdout_or_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'PRIVATE_PATH.json'
            for raw in ['PRIVATE_SECRET', source_with('probability_1', '1e-400')]:
                path.write_text(raw)
                with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                    code = main([str(path)])
                self.assertEqual(code, 2); self.assertEqual(out.getvalue(), '')
                self.assertNotIn('PRIVATE', err.getvalue())
            path.unlink()
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main([str(path)]), 2)
            self.assertEqual(out.getvalue(), ''); self.assertNotIn('PRIVATE', err.getvalue())

    def test_main_regular_file_alias_and_directory_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.json'; source.write_text(json.dumps(document()))
            alias = root/'alias.json'; alias.symlink_to(source)
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(alias)]), 0)
            self.assertEqual(json.loads(out.getvalue())['cohort']['scored_rows'], 4)
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(root)]), 2)
            self.assertEqual(out.getvalue(), '')

    def test_reader_reuse_and_all_scoring_completes_before_print(self):
        with patch('groupbench.binary.saved_evaluation._read_input', return_value=json.dumps(document())) as reader, \
             patch('groupbench.binary.saved_evaluation.evaluate_saved_binary_probabilities', side_effect=ValueError('PRIVATE')):
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main(['unused.json']), 2)
            reader.assert_called_once_with(Path('unused.json'))
            self.assertEqual(out.getvalue(), ''); self.assertNotIn('PRIVATE', err.getvalue())

    def test_real_module_subprocess_and_argument_privacy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.json'; path.write_text(json.dumps(document()))
            command = [sys.executable, '-m', 'groupbench.binary.saved_evaluation']
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run(command+[str(path)], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['cohort']['scored_rows'], 4)
            result = subprocess.run(command+['--PRIVATE_VALUE'], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2); self.assertEqual(result.stdout, '')
            self.assertNotIn('PRIVATE_VALUE', result.stderr)


if __name__ == '__main__':
    unittest.main()
