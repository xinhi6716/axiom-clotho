"""The deep-linkable native saved-series guide uses runnable root commands."""
from pathlib import Path
import re
import shlex
import unittest


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "RUST_BASELINE_SKILL.md"


class NativeSavedCommandGuideTests(unittest.TestCase):
    def test_each_cargo_example_declares_root_manifest_and_cli_package(self):
        blocks = re.findall(r"```sh\n(.*?)\n```", GUIDE.read_text(encoding="utf-8"), re.S)
        examples = []
        for block in blocks:
            joined = block.replace("\\\n", "")
            for line in joined.splitlines():
                if not line.lstrip().startswith("cargo run "):
                    continue
                tokens = shlex.split(line)
                separator = tokens.index("--")
                cargo_options = tokens[2:separator]
                command = tokens[separator + 1]
                examples.append(command)
                with self.subTest(command=command):
                    self.assertIn("--manifest-path", cargo_options)
                    manifest = cargo_options[cargo_options.index("--manifest-path") + 1]
                    self.assertEqual(manifest, "rust/Cargo.toml")
                    self.assertTrue((ROOT / manifest).is_file())
                    self.assertEqual(cargo_options[cargo_options.index("-p") + 1], "clotho-cli")
                    self.assertIn("--locked", cargo_options)
        self.assertTrue({"compare-saved", "predict", "prepare-observed", "pool-saved"} <= set(examples))


if __name__ == "__main__":
    unittest.main()
