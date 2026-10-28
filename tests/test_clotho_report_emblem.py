"""Decorative coil contract; static checks, not browser or screen-reader QA."""

import copy
import html
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

from groupbench.report import render_html
from groupbench.binary.report import render_binary_html
from test_report_reading_path import fixture as regression_fixture
from test_binary_report_reading_path import fixture as binary_fixture


def emblem(text):
    matches = re.findall(r'<svg class="clotho-mark".*?</svg>', text)
    if len(matches) != 1:
        raise AssertionError("Each report must have exactly one decorative Clotho mark")
    return matches[0]


class ClothoReportEmblemTests(unittest.TestCase):
    renderers = ((render_html, regression_fixture),
                 (render_binary_html, binary_fixture))

    def test_regression_and_binary_share_exact_decorative_svg(self):
        self.assertEqual(emblem(render_html(regression_fixture())),
                         emblem(render_binary_html(binary_fixture())))

    def test_coil_geometry_matches_all_current_hero_variants(self):
        assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
        ns = {"svg": "http://www.w3.org/2000/svg"}
        for render, fixture in self.renderers:
            actual = ET.fromstring(emblem(render(fixture()))).find("g/path")
            self.assertIsNotNone(actual)
            for suffix in ("light", "dark", "light-mobile", "dark-mobile"):
                with self.subTest(renderer=render.__name__, hero=suffix):
                    hero = ET.parse(assets / f"clotho-ritual-hero-{suffix}.svg")
                    coil = hero.find('.//svg:g[@id="continuous-axial-thread"]/svg:path', ns)
                    self.assertIsNotNone(coil)
                    self.assertEqual(actual.attrib["d"], coil.attrib["d"])

    def test_mark_is_only_a_noninteractive_coil_in_the_existing_canvas(self):
        for render, fixture in self.renderers:
            with self.subTest(renderer=render.__name__):
                svg = ET.fromstring(emblem(render(fixture())))
                self.assertEqual(svg.attrib, {
                    "class": "clotho-mark", "viewBox": "0 0 32 32",
                    "aria-hidden": "true", "focusable": "false",
                })
                self.assertEqual([node.tag for node in svg.iter()], ["svg", "g", "path"])
                self.assertEqual(svg[0].attrib, {
                    "transform": "translate(-1.92 -.59) scale(.07)",
                })
                self.assertEqual(set(svg[0][0].attrib), {"d", "fill"})
                self.assertEqual(svg[0][0].attrib["fill"], "var(--accent)")
                self.assertFalse(any((node.text or "").strip() for node in svg.iter()))
                self.assertFalse(any((node.tail or "").strip() for node in svg.iter()))

    def test_existing_header_size_accessible_name_and_theme_rules_remain(self):
        for render, fixture in self.renderers:
            with self.subTest(renderer=render.__name__):
                text = render(fixture())
                self.assertIn('<h1>' + emblem(text) + 'Axiom Clotho</h1>', text)
                self.assertIn('.clotho-mark{width:38px;height:38px;vertical-align:middle;'
                              'margin:0 14px 3px 0;overflow:visible}', text)
                self.assertIn('.clotho-mark{width:30px;height:30px;margin-right:9px}', text)
                self.assertIn('@media(prefers-color-scheme:dark)', text)
                self.assertIn('--accent:#D4B77B', text)
                self.assertIn('@media print', text)
                self.assertIn("default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
                              "base-uri 'none'; form-action 'none'", text)

    def test_untrusted_data_never_enters_emblem_and_render_does_not_mutate(self):
        attack = '<svg onload="alert(1)">untrusted & 群組</svg>'
        for render, fixture in self.renderers:
            with self.subTest(renderer=render.__name__):
                result = fixture()
                trusted_emblem = emblem(render(result))
                result.manifest["config"].update(target=attack, group=attack)
                result.manifest["limitations"] = [attack]
                before = copy.deepcopy(result)
                text = render(result)
                self.assertEqual(emblem(text), trusted_emblem)
                self.assertNotIn(attack, text)
                self.assertIn(html.escape(attack, quote=True), text)
                self.assertEqual(text, render(result))
                self.assertEqual(result.summary, before.summary)
                self.assertEqual(result.manifest, before.manifest)


if __name__ == "__main__":
    unittest.main()
