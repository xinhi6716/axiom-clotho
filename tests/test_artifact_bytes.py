"""Exact-byte output tests; CRLF default emulation is not a Windows OS test."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from groupbench.report import write_payloads
from groupbench.verification import _SAFE_OPEN_SUPPORTED, verify_bundle


_REAL_OPEN = Path.open


def crlf_default(path, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
    if 'w' in mode and 'b' not in mode and newline is None:
        newline = '\r\n'
    return _REAL_OPEN(path, mode, buffering, encoding, errors, newline)


class ArtifactByteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.payloads = {'manifest.json': '{"label": "晶片 μ 🧪"}\n',
                         'predictions.csv': 'x\n1\r\n2\r3\n'}

    def check_bytes(self, payloads):
        expected = {name: hashlib.sha256(content.encode('utf-8')).hexdigest()
                    for name, content in payloads.items()}
        for name, content in payloads.items():
            self.assertEqual((self.out/name).read_bytes(), content.encode('utf-8'))
        checksum = {'algorithm': 'sha256', 'files': expected}
        self.assertEqual((self.out/'checksums.json').read_bytes(),
                         (json.dumps(checksum, indent=2)+'\n').encode('utf-8'))
        if _SAFE_OPEN_SUPPORTED:
            self.assertEqual(verify_bundle(self.out).bundle_type, 'prediction')

    def test_native_default_exact_bytes(self):
        write_payloads(self.payloads, self.out)
        self.check_bytes(self.payloads)

    def test_crlf_default_exact_bytes(self):
        with patch.object(Path, 'open', crlf_default):
            write_payloads(self.payloads, self.out)
        self.check_bytes(self.payloads)

    def test_empty_and_no_final_newline(self):
        payloads = {'manifest.json': '{}', 'predictions.csv': ''}
        with patch.object(Path, 'open', crlf_default):
            write_payloads(payloads, self.out)
        self.check_bytes(payloads)

    def test_overwrite_exact_bytes(self):
        write_payloads({'manifest.json':'{}', 'predictions.csv':'old'}, self.out)
        with patch.object(Path, 'open', crlf_default):
            write_payloads(self.payloads, self.out, overwrite=True)
        self.check_bytes(self.payloads)

    def test_no_clobber_preserves_original_bytes(self):
        write_payloads(self.payloads, self.out)
        before = {p.name:p.read_bytes() for p in self.out.iterdir()}
        with patch.object(Path, 'open', crlf_default):
            with self.assertRaises(FileExistsError):
                write_payloads({'manifest.json':'{}', 'predictions.csv':'new'}, self.out)
        self.assertEqual({p.name:p.read_bytes() for p in self.out.iterdir()}, before)

    def test_staging_failure_leaves_no_committed_payloads(self):
        real = Path.write_bytes
        def fail(path, data):
            if path.name == 'predictions.csv':
                raise OSError('synthetic staging write failure')
            return real(path, data)
        with patch.object(Path, 'write_bytes', fail):
            with self.assertRaisesRegex(OSError, 'synthetic staging'):
                write_payloads(self.payloads, self.out)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_failed_overwrite_staging_keeps_originals(self):
        write_payloads(self.payloads, self.out)
        before = {p.name:p.read_bytes() for p in self.out.iterdir()}
        with patch.object(Path, 'write_bytes', side_effect=OSError('synthetic')):
            with self.assertRaises(OSError):
                write_payloads({'manifest.json':'{}', 'predictions.csv':'new'}, self.out,
                               overwrite=True)
        self.assertEqual({p.name:p.read_bytes() for p in self.out.iterdir()}, before)

    def test_invalid_unicode_never_commits(self):
        with self.assertRaises(UnicodeEncodeError):
            write_payloads({'manifest.json':'{}', 'predictions.csv':'\ud800'}, self.out)
        self.assertEqual(list(self.out.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
