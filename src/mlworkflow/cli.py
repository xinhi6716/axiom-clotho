"""Axiom Clotho: local measurement-to-regression workflows and calculator core.

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

from . import __version__
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
        if arguments and arguments[0] == "tabular":
            from groupbench.cli import main as calculator_main
            return calculator_main(arguments[1:])
        if arguments and arguments[0] == "init":
            return init_main(arguments[1:])
        if arguments and arguments[0] in ("prepare", "plan", "run", "score"):
            return regression_main(arguments)
        # Delegate feature parsing to its existing contract, rather than maintaining
        # a divergent option list. Unknown options still use its safe error parser.
        if arguments and arguments[0] == "features":
            return run_features(arguments[1:])
        if arguments and arguments[0] == "demo":
            return run_demo(arguments[1:])
        parser = Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="""Commands (each supports --help):
  init        Create blank RRAM/FET inputs; no fitting (0.3.0b4; not in frozen b3).
  prepare     Validate measurements, partitions and training-only targets.
  plan        Preview grouped splits and fit counts; do not fit.
  run         Fit the existing grouped models and save future predictions.
  score       Evaluate saved predictions against later outcomes; no new fits.
  tabular     Use the existing grouped tabular calculator.
  features    Use the existing standalone RRAM feature extractor.
  demo        Run a separate, wholly invented semiconductor example.
  components  Verify and print pinned component identities.

For your measurements: clotho init --workflow rram --out my-measurements
For FET:               clotho init --workflow fet --out my-transfer-data
Init writes header-only CSVs and an intentionally incomplete configuration.
Read the generated README.md before filling your own measurements and choices.""")
        parser.add_argument("--version", action="version", version=f"Axiom Clotho {__version__}; RRAM measurement-regression-v1; FET measurement-regression-v2")
        parser.add_argument("command", nargs="?", choices=("components",),
                            help="verify and print pinned component identities")
        args = parser.parse_args(arguments)
        if args.command == "components":
            print(json.dumps(verify_components(), indent=2, sort_keys=True))
        else:
            parser.print_help()
        return 0
    except (ComponentError, OSError):
        print("error: workflow arguments, local resources or output are unavailable; use --help and a complete distribution", file=sys.stderr)
        return 2


def init_main(arguments):
    from .contracts import WorkflowError
    from .scaffold import initialize
    parser = Parser(prog="clotho init", allow_abbrev=False,
                    description="Create blank RRAM or FET input files, not invented measurements. "
                                "Included in 0.3.0b4; not in the frozen 0.3.0b3 wheel. "
                                "No fitting, extraction, download or upload.")
    parser.add_argument("--workflow", required=True, choices=("rram", "fet"),
                        help="choose the existing measurement contract explicitly")
    parser.add_argument("--out", required=True,
                        help="fresh directory with an existing parent; rejects pre-existing outputs and symlink path components")
    args = parser.parse_args(arguments)
    try:
        initialize(args.workflow, args.out)
    except WorkflowError as error:
        print("error: " + str(error), file=sys.stderr)
        return 2
    except OSError:
        print("error: init could not finish local writes; a partial directory may remain. "
              "Do not use it; retry with a fresh output directory.", file=sys.stderr)
        return 2
    print("Blank " + args.workflow.upper() + " input scaffold created, not ready to prepare. "
          "Open README.md in the new directory; fill your measurements and explicit choices. "
          "No models were fitted.")
    return 0


def regression_main(arguments):
    from . import regression
    from .contracts import WorkflowError
    parser = Parser(prog="mlcalc-workflow " + arguments[0], description="Axiom Clotho: explicit device joins, grouped regression and separate saved scoring.")
    command = arguments[0]
    if command == "prepare":
        parser.add_argument("--input-dir", required=True)
        parser.add_argument("--partitions", required=True)
        parser.add_argument("--training-targets", required=True)
        parser.add_argument("--config", required=True)
    elif command in ("plan", "run"):
        parser.add_argument("--prepared", required=True)
    else:
        parser.add_argument("--run-dir", required=True)
        parser.add_argument("--outcomes", required=True)
    if command != "plan":
        parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(arguments[1:])
    try:
        if command == "prepare":
            regression.prepare(args.input_dir, args.partitions, args.training_targets, args.config, args.output_dir)
        elif command == "plan":
            print(json.dumps(regression.plan(args.prepared), indent=2, sort_keys=True, allow_nan=False))
            return 0
        elif command == "run":
            regression.run(args.prepared, args.output_dir)
        else:
            regression.score(args.run_dir, args.outcomes, args.output_dir)
        print("Completed " + command + ". Keep the complete bundle together; checksums are not scientific validation.")
        return 0
    except WorkflowError as error:
        print("error: " + str(error), file=sys.stderr)
        return 2
    except (OSError, ImportError):
        print("error: local I/O or existing ML dependencies are unavailable; incomplete bundles are not usable", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
