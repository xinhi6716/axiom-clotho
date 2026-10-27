"""Read and verify pinned resources without loading executable or ML modules."""
import hashlib
import json
from pathlib import Path
import re

RESOURCE_ROOT = Path(__file__).resolve().parent / "_vendor"

# The versioned inventory is independent of the manifest it validates. An empty
# or shortened manifest must never turn integrity checking into a vacuous pass.
EXPECTED_COMPONENTS = {
    "semiconductor": ("semiconductor-machine-learning-toolbox", "4c0ec2ae3eaa4e0b7afc9aaf8f9bada544e7c0de", {
        "LICENSE", "examples/measurement-to-features/README.md",
        "examples/measurement-to-features/build_features.py",
        "examples/measurement-to-features/generate_synthetic.py",
        "tools/iv-readpoint/iv_readpoint.py", "tools/rram-window/rram_window.py",
        "tools/physical-holdout-audit/physical_holdout_audit.py"}),
    "calculator": ("ml-super-calculator", "1f26d265411c86b0c6b30a01bc04d3605f312997", {
        "LICENSE", "examples/semiconductor-heldout-walkthrough/README.md",
        "examples/semiconductor-heldout-walkthrough/run.py",
        "examples/semiconductor-heldout-walkthrough/fixtures/training_endpoints.csv",
        "examples/semiconductor-heldout-walkthrough/fixtures/heldout_outcomes.csv",
        "examples/environment-heldout-comparison/probe.py"}),
}


class ComponentError(ValueError):
    """Categorical resource failure; never include local paths or user data."""


def verify_components(root=None):
    """Integrity/parity check, not source authentication or scientific validation."""
    root = RESOURCE_ROOT if root is None else Path(root)
    try:
        manifest = json.loads((root / "manifest.json").read_bytes())
        if (not isinstance(manifest, dict)
                or set(manifest) != {"schema_version", "policy", "components"}
                or manifest["schema_version"] != "ml-workflow-components-v1"
                or not isinstance(manifest["policy"], str) or not manifest["policy"]
                or not isinstance(manifest["components"], list)
                or len(manifest["components"]) != len(EXPECTED_COMPONENTS)):
            raise ValueError
        seen = set()
        for component in manifest["components"]:
            if not isinstance(component, dict) or set(component) != {"component", "repository", "commit", "license", "license_file", "files"}:
                raise ValueError
            label = component["component"]
            if not isinstance(label, str) or label not in EXPECTED_COMPONENTS or label in seen:
                raise ValueError
            seen.add(label)
            repository, commit, expected_files = EXPECTED_COMPONENTS[label]
            if (component["repository"] != "https://github.com/xinhi6716/" + repository
                    or component["commit"] != commit or component["license"] != "MIT"
                    or component["license_file"] != label + "/LICENSE"
                    or not isinstance(component["files"], list)
                    or len(component["files"]) != len(expected_files)):
                raise ValueError
            seen_files = set()
            for entry in component["files"]:
                if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "git_blob_sha1", "bytes"}:
                    raise ValueError
                name = entry["path"]
                if (not isinstance(name, str) or name not in expected_files or name in seen_files
                        or type(entry["bytes"]) is not int or entry["bytes"] < 0
                        or not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                        or not isinstance(entry["git_blob_sha1"], str) or not re.fullmatch(r"[0-9a-f]{40}", entry["git_blob_sha1"])):
                    raise ValueError
                seen_files.add(name)
                data = (root / label / name).read_bytes()
                blob_id = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                if (len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]
                        or blob_id != entry["git_blob_sha1"]):
                    raise ValueError
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        raise ComponentError("bundled component resources are missing or changed; use a complete reviewed distribution") from None
