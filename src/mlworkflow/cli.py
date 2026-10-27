"""Single-distribution entry point for the bounded measurement workflow.

No ML import on help, component inspection, or measurement extraction. Extraction
and prediction algorithms remain the pinned upstream components, not rewrites.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .components import ComponentError, RESOURCE_ROOT, verify_components


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ComponentError("invalid workflow arguments; use --help")


def run_features(arguments):
    """Delegate the documented CSV contract unchanged to the stdlib extractor."""
    verify_components()
    script = RESOURCE_ROOT / "semiconductor/examples/measurement-to-features/build_features.py"
    # -S makes the promised standard-library boundary executable, not aspirational.
    return subprocess.run([sys.executable, "-S", str(script), *arguments],
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=False).returncode


def _snapshot_calculator(destination):
    """Stage existing installed Python source for the unchanged legacy runner.

    No installation, network, executable model, or independently forked algorithm.
    The legacy runner needs a source-shaped tree for subprocesses and hashes.
    find_spec on this top-level package does not execute groupbench/__init__.py.
    """
    spec = importlib.util.find_spec("groupbench")
    if spec is None or spec.origin is None:
        raise ComponentError("the calculator Python source is unavailable in this distribution")
    source = Path(spec.origin).parent
    files = sorted(source.rglob("*.py"))
    if not files or not (source / "model.py").is_file():
        raise ComponentError("the calculator Python source is unavailable in this distribution")
    shutil.copytree(RESOURCE_ROOT / "calculator", destination)
    for path in files:
        target = destination / "src/groupbench" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())


def run_demo(arguments):
    parser = Parser(prog="mlcalc-workflow demo", description="Entirely invented software example, not scientific validation.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-invented-example", action="store_true", required=True)
    args = parser.parse_args(arguments)
    output = Path(args.output_dir).absolute()
    if os.path.lexists(output) or not output.parent.is_dir():
        raise ComponentError("output must be a fresh directory with an existing parent")
    verify_components()
    with tempfile.TemporaryDirectory(prefix="ml-workflow-") as directory:
        calculator = Path(directory) / "calculator"
        _snapshot_calculator(calculator)
        command = [sys.executable, str(calculator / "examples/semiconductor-heldout-walkthrough/run.py"),
                   "--semiconductor-source", str(RESOURCE_ROOT / "semiconductor"),
                   "--calculator-source", str(calculator), "--output-dir", str(output),
                   "--run-invented-example"]
        return subprocess.run(command, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=False).returncode


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        # Delegate feature parsing to its existing contract, rather than maintaining
        # a divergent option list. Unknown options still use its safe error parser.
        if arguments and arguments[0] == "features":
            return run_features(arguments[1:])
        if arguments and arguments[0] == "demo":
            return run_demo(arguments[1:])
        parser = Parser(description=__doc__)
        parser.add_argument("command", nargs="?", choices=("components",),
                            help="verify and print pinned component identities")
        args = parser.parse_args(arguments)
        if args.command == "components":
            print(json.dumps(verify_components(), indent=2, sort_keys=True))
        else:
            parser.print_help()
            print("\nCommands: features --help | demo --help | components")
        return 0
    except (ComponentError, OSError):
        print("error: workflow arguments, local resources or output are unavailable; use --help and a complete distribution", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
