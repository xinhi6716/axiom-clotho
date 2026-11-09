"""Offline contracts for the bilingual, responsive README illustrations."""

import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_readme_workflow", ROOT / "tools" / "render_readme_workflow.py"
)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)
SVG = "{http://www.w3.org/2000/svg}"


def contrast(a, b):
    def luminance(value):
        channels = [int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
                  for v in channels]
        return sum(v * weight for v, weight in zip(linear, (0.2126, 0.7152, 0.0722)))
    light, dark = sorted((luminance(a), luminance(b)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


class ReadmeWorkflowAssetsTest(unittest.TestCase):
    def test_generated_assets_are_current_and_self_contained(self):
        assets = list(RENDERER.assets())
        self.assertEqual(len(assets), 8)
        for path, expected in assets:
            with self.subTest(path=path.name):
                self.assertEqual(path.read_text(encoding="utf-8"), expected)
                root = ET.fromstring(expected)
                self.assertEqual(root.tag, SVG + "svg")
                self.assertEqual(root.attrib["role"], "img")
                self.assertEqual(root.attrib["aria-labelledby"], "title desc")
                self.assertTrue(root.find(SVG + "title").text)
                self.assertTrue(root.find(SVG + "desc").text)
                for element in root.iter():
                    self.assertNotIn(element.tag, {
                        SVG + "script", SVG + "image", SVG + "foreignObject",
                        SVG + "animate", SVG + "style",
                    })
                    for key, value in element.attrib.items():
                        self.assertFalse(key.lower().startswith("on"))
                        self.assertFalse(key.endswith("href"))
                        self.assertNotIn("url(", value.lower())

    def test_variants_preserve_stage_order_and_outcome_boundary(self):
        for path, content in RENDERER.assets():
            with self.subTest(path=path.name):
                root = ET.fromstring(content)
                language = root.attrib["{http://www.w3.org/XML/1998/namespace}lang"]
                copy = RENDERER.COPY[language]
                labels = [node.text for node in root.iter(SVG + "text")]
                positions = []
                for i, step in enumerate(copy["steps"]):
                    self.assertEqual(labels.count(f"0{i + 1}"), 1)
                    for value in step:
                        self.assertIn(value, labels)
                    positions.append(labels.index(step[0]))
                self.assertEqual(positions, sorted(positions))
                self.assertIn(copy["boundary"], labels)
                if "-mobile" in path.name:
                    self.assertEqual(root.attrib["viewBox"], "0 0 480 906")
                    for note in copy["mobile_note"]:
                        self.assertIn(note, labels)
                else:
                    self.assertEqual(root.attrib["viewBox"], "0 0 1200 454")
                    self.assertIn(copy["note"], labels)

    def test_text_palette_meets_aa_contrast_on_both_surfaces(self):
        for theme, colors in RENDERER.PALETTES.items():
            for text_color in ("ink", "muted", "gold"):
                for background in ("bg", "panel"):
                    with self.subTest(theme=theme, text=text_color, surface=background):
                        self.assertGreaterEqual(contrast(colors[text_color], colors[background]), 4.5)

    def test_readmes_select_localized_mobile_and_color_variants(self):
        for language, readme in (("en", "README.md"), ("zh-TW", "README.zh-TW.md")):
            content = (ROOT / readme).read_text(encoding="utf-8")
            expected = (
                f'<source media="(prefers-color-scheme: dark) and (max-width: 600px)" '
                f'srcset="docs/assets/clotho-workflow-{language}-dark-mobile.svg">',
                f'<source media="(max-width: 600px)" '
                f'srcset="docs/assets/clotho-workflow-{language}-light-mobile.svg">',
                f'<source media="(prefers-color-scheme: dark)" '
                f'srcset="docs/assets/clotho-workflow-{language}-dark.svg">',
                f'<img src="docs/assets/clotho-workflow-{language}-light.svg" '
                f'width="100%" alt="{RENDERER.COPY[language]["desc"]}">',
            )
            positions = [content.index(fragment) for fragment in expected]
            self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
