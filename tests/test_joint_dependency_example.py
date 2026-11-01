"""Original tiny synthetic handoff checks; standard library only, no model fits."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1] / 'examples/joint-dependency-tabular'
spec = importlib.util.spec_from_file_location('joint_example', HERE / 'validate.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class JointDependencyExampleTests(unittest.TestCase):
    def setUp(self):
        self.rows = example.read_table(HERE / 'specimens.csv', example.FIELDS)
        self.mapping = example.read_table(HERE / 'expected_components.csv', ('specimen_id', 'component_id'))

    def test_exact_join_keeps_order_and_tokens(self):
        result = example.validate_and_join(self.rows, self.mapping[::-1])
        self.assertEqual([r['specimen_id'] for r in result], [r['specimen_id'] for r in self.rows])
        self.assertEqual([{k: v for k, v in r.items() if k != 'component_id'} for r in result], self.rows)
        self.assertEqual([r['component_id'] for r in result], ['C1'] * 4 + ['C2'] * 2 + ['C3', 'C4'])

    def test_transitive_chain_not_concatenated_pairs(self):
        a, b, c, d = self.rows[:4]
        self.assertEqual(a['job_id'], b['job_id'])
        self.assertEqual(tuple(b[k] for k in example.SETTING), tuple(c[k] for k in example.SETTING))
        self.assertEqual(c['job_id'], d['job_id'])
        self.assertNotEqual(a['job_id'], d['job_id'])
        self.assertNotEqual(tuple(a[k] for k in example.SETTING), tuple(d[k] for k in example.SETTING))
        wrong = copy.deepcopy(self.mapping); wrong[3]['component_id'] = 'C5'
        with self.assertRaises(ValueError): example.validate_and_join(self.rows, wrong)

    def test_exact_text_identity_and_no_coordinate_only_merge(self):
        self.assertEqual((self.rows[0]['job_id'], self.rows[6]['job_id']), ('01', '1'))
        self.assertEqual((self.rows[0]['power'], self.rows[6]['power']), ('200', '200.0'))
        self.assertEqual(len({r['hatch'] for r in self.rows}), 1)
        result = example.validate_and_join(self.rows, self.mapping)
        self.assertEqual(len({r['component_id'] for r in result}), 4)

    def test_missing_extra_duplicate_either_side_rejected(self):
        for side in ('rows', 'mapping'):
            original = getattr(self, side)
            extra = dict(original[0], specimen_id='unknown')
            for changed in (original[:-1], original + [extra], original + [original[0]]):
                with self.subTest(side=side, count=len(changed)), self.assertRaises(ValueError):
                    example.validate_and_join(changed if side == 'rows' else self.rows,
                                              changed if side == 'mapping' else self.mapping)

    def test_matching_but_wrong_roster_rejected(self):
        for rows, mapping in ((self.rows[:-1], self.mapping[:-1]), ([], [])):
            with self.assertRaises(ValueError): example.validate_and_join(rows, mapping)

    def test_overmerge_and_inconsistent_source_rejected(self):
        merged = [dict(r, component_id='C1') for r in self.mapping]
        with self.assertRaises(ValueError): example.validate_and_join(self.rows, merged)
        altered = copy.deepcopy(self.rows); altered[-1]['job_id'] = '01'
        with self.assertRaises(ValueError): example.validate_and_join(altered, self.mapping)

    def test_parser_rejects_bad_header_shape_empty_nul_utf8(self):
        for raw in (b'a,a\nx,y\n', b'specimen_id,component_id\ns01\n',
                    b'specimen_id,component_id\ns01,C1,extra\n', b'specimen_id,component_id\ns01,\n',
                    b'\x00', b'\xff', b'x' * 65537):
            with self.subTest(size=len(raw)), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / 'bad.csv'; p.write_bytes(raw)
                with self.assertRaises((ValueError, UnicodeError)):
                    example.read_table(p, ('specimen_id', 'component_id'))

    def test_output_exclusive_and_reparse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'joined.csv'
            example.main(['--output', str(p)])
            before = p.read_bytes()
            self.assertEqual(len(example.read_table(p, (*example.FIELDS, 'component_id'))), 8)
            with self.assertRaises(FileExistsError): example.main(['--output', str(p)])
            self.assertEqual(p.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
