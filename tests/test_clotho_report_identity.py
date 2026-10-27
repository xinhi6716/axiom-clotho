"""Static presentation contract; does not certify browser rendering."""
from pathlib import Path
import unittest
from groupbench import report
from groupbench.binary import report as binary_report
from test_report_reading_path import fixture

class ClothoReportIdentityTests(unittest.TestCase):
    def test_regression_identity_and_static_containment(self):
        html = report.render_html(fixture())
        self.assertIn('Axiom Clotho', html)
        self.assertNotIn('<h1>ML Super Calculator</h1>', html)
        self.assertIn('repeat(3,minmax(0,1fr))', html)
        self.assertIn('repeat(2,minmax(0,1fr))', html)
        self.assertIn('overflow-wrap:anywhere', html)
        self.assertIn('@media(prefers-color-scheme:dark)', html)
        self.assertIn('@media print', html)
        self.assertIn('focus-visible', html)
    def test_both_renderers_keep_offline_identity(self):
        for module in (report,binary_report):
            source=Path(module.__file__).read_text()
            self.assertIn('Axiom Clotho |',source)
            self.assertIn('clotho-mark',source)
            self.assertIn('repeat(2,minmax(0,1fr))',source)
            self.assertNotIn('@import',source)
            self.assertNotIn('fonts.googleapis.com',source)

if __name__=='__main__': unittest.main()
