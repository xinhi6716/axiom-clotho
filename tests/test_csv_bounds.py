"""Regular-file and exact-byte guards for all shared CSV input paths."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from groupbench import cli


class CSVBoundsTests(unittest.TestCase):
    def test_exact_limit_and_multibyte_bom_blank_bytes(self):
        content = '\ufeffgroup,x,y\r\n"測試\n組",1,2\r\n \t\r\n'.encode()
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp, 'data.csv'); p.write_bytes(content)
            with patch.object(cli, 'MAX_CSV_BYTES', len(content)):
                frame, digest = cli._read_csv_with_hash(p, group='group')
            self.assertEqual(digest, hashlib.sha256(content).hexdigest())
            self.assertEqual(frame.group.tolist(), ['測試\n組'])
            with patch.object(cli, 'MAX_CSV_BYTES', len(content)-1), patch.object(cli.os, 'open') as opening:
                with self.assertRaisesRegex(ValueError, 'byte input limit'):
                    cli._read_csv_with_hash(p)
                opening.assert_not_called()

    def test_regular_symlink_and_hardlink_preserve_same_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp, 'data.csv'); p.write_bytes(b'group,x,y\ng,1,2\n')
            symlink=Path(temp,'link.csv'); symlink.symlink_to(p)
            hardlink=Path(temp,'hard.csv'); os.link(p,hardlink)
            for q in (p,symlink,hardlink):
                self.assertEqual(cli._read_csv_with_hash(q)[1], hashlib.sha256(p.read_bytes()).hexdigest())

    def test_special_modes_rejected_before_open(self):
        for mode in (stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFDIR):
            with self.subTest(mode=mode), patch.object(Path,'stat',return_value=SimpleNamespace(st_mode=mode,st_size=0)), patch.object(cli.os,'open') as opening:
                with self.assertRaisesRegex(ValueError,'regular local file'):
                    cli._read_csv_with_hash(Path('not-opened.csv'))
                opening.assert_not_called()

    def test_missing_and_broken_symlink_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp,'missing.csv'); q=Path(temp,'broken.csv'); q.symlink_to(p)
            for path in (p,q):
                with self.assertRaises(FileNotFoundError): cli._read_csv_with_hash(path)

    def test_descriptor_recheck_rejects_changed_mode_and_size_and_closes(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp,'data.csv'); p.write_bytes(b'x\n1\n')
            for metadata in (SimpleNamespace(st_mode=stat.S_IFIFO,st_size=0), SimpleNamespace(st_mode=stat.S_IFREG,st_size=cli.MAX_CSV_BYTES+1)):
                seen=[]; real_open=os.open; real_close=os.close
                def capture(*args,**kwargs):
                    fd=real_open(*args,**kwargs); seen.append(fd); return fd
                with patch.object(cli.os,'open',side_effect=capture), patch.object(cli.os,'fstat',return_value=metadata), patch.object(cli.os,'close',wraps=real_close) as close:
                    with self.assertRaises(ValueError): cli._read_csv_with_hash(p)
                    close.assert_called_once_with(seen[0])
                with self.assertRaises(OSError): os.fstat(seen[0])

    def test_fdopen_and_fstat_failures_release_owned_descriptor(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp,'data.csv'); p.write_bytes(b'x\n1\n')
            for target in ('fdopen','fstat'):
                seen=[]; real_open=os.open
                def capture(*args,**kwargs):
                    fd=real_open(*args,**kwargs);seen.append(fd);return fd
                with patch.object(cli.os,'open',side_effect=capture), patch.object(cli.os,target,side_effect=OSError('synthetic failure')):
                    with self.assertRaisesRegex(OSError,'synthetic failure'): cli._read_csv_with_hash(p)
                with self.assertRaises(OSError): os.fstat(seen[0])

    def test_open_flags_nonblocking_where_available(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp,'data.csv');p.write_bytes(b'x\n1\n')
            with patch.object(cli.os,'open',wraps=os.open) as opening:
                cli._read_csv_with_hash(p)
            flags=opening.call_args.args[1]
            for name in ('O_NONBLOCK','O_CLOEXEC','O_BINARY'):
                if getattr(os,name,0):self.assertTrue(flags & getattr(os,name))

    def test_stream_cap_catches_growth_and_stops_after_one_excess_byte(self):
        class Stream(io.BytesIO):
            def __init__(self,data):super().__init__(data);self.total=0;self.requests=[]
            def readinto(self,buf):
                self.requests.append(len(buf));n=super().readinto(buf);self.total+=n;return n
        raw=b'x\n1\n'+b' \n'*500
        stream=Stream(raw)
        with patch.object(cli,'_open_csv_source',return_value=stream),patch.object(cli,'MAX_CSV_BYTES',31),patch.object(cli.pd,'read_csv') as parsing:
            with self.assertRaisesRegex(ValueError,'31-byte input limit'):cli._read_csv_with_hash(Path('synthetic'))
        self.assertEqual(stream.total,32);self.assertTrue(stream.closed);parsing.assert_not_called()
        self.assertLessEqual(max(stream.requests),32)

    def test_stream_exact_limit_requires_eof_and_preserves_hash(self):
        content=b'x\n1\n'+b' '*10
        with patch.object(cli,'_open_csv_source',return_value=io.BytesIO(content)),patch.object(cli,'MAX_CSV_BYTES',len(content)):
            frame,digest=cli._read_csv_with_hash(Path('synthetic'))
        self.assertEqual(len(frame),1);self.assertEqual(digest,hashlib.sha256(content).hexdigest())

    def test_stream_chunks_bounded_even_for_large_caller_buffer(self):
        stream=io.BytesIO(b'x'*100_000);reader=cli._HashingReader(stream,hashlib.sha256())
        self.assertEqual(reader.readinto(bytearray(200_000)),cli.CSV_READ_CHUNK_BYTES)
        self.assertEqual(reader.bytes_read,cli.CSV_READ_CHUNK_BYTES)

    def test_all_cli_paths_reject_before_model_or_output(self):
        actions=('groupbench.cli.benchmark','groupbench.cli.fit_predict','groupbench.cli.plan_benchmark','groupbench.cli.plan_prediction',
                 'groupbench.binary.cli.benchmark_binary','groupbench.binary.cli.fit_predict_binary','groupbench.binary.cli.plan_binary_benchmark','groupbench.binary.cli.plan_binary_prediction')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);good=root/'good.csv';bad=root/'bad.csv';new=root/'new.csv';out=root/'output'
            good.write_text('group,x,y\na,1,0\nb,2,1\n');new.write_text('x\n1\n');bad.write_bytes(b'x\n'+b' '*200)
            for binary in (False,True):
                for planning in (False,True):
                    for mode in ('run','predict'):
                        for role in (('training','prediction') if mode=='predict' else ('training',)):
                            argv=(['binary'] if binary else [])+(['plan'] if planning else [])+[mode,str(bad if role=='training' else good)]
                            if mode=='predict':argv+=[str(bad if role=='prediction' else new)]
                            argv+=['--target','y','--group','group']
                            if not planning:argv+=['--out',str(out)]
                            stdout,stderr=io.StringIO(),io.StringIO()
                            with self.subTest(argv=argv), ExitStack() as stack:
                                stack.enter_context(patch.object(cli,'MAX_CSV_BYTES',100))
                                for action in actions:stack.enter_context(patch(action,side_effect=AssertionError('model or plan called')))
                                stack.enter_context(redirect_stdout(stdout));stack.enter_context(redirect_stderr(stderr))
                                self.assertEqual(cli.main(argv),2)
                            self.assertEqual(stdout.getvalue(),'');self.assertIn('byte input limit',stderr.getvalue());self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
