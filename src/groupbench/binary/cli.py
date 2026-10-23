"""Additive binary CLI namespace; regression commands retain their defaults."""
from pathlib import Path
import json
from .report import ensure_fresh_binary_output
from .workflow import benchmark_binary, fit_predict_binary, plan_binary_benchmark, plan_binary_prediction


def _options(parser, *, outer, output):
    from ..cli import _add_columns
    _add_columns(parser)
    if outer:
        parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baselines", default="dummy,logistic,extra_trees",
                        help="Distinct ordered families: dummy,logistic,extra_trees")
    if output:
        parser.add_argument("--out", type=Path, required=True, help="Fresh local directory; no overwrite")


def add_binary_parser(sub):
    binary = sub.add_parser("binary", allow_abbrev=False, help="Explicit numeric 0/1 classification; fixed group Brier and p1>=0.5")
    commands = binary.add_subparsers(dest="binary_command", required=True)
    for mode in ("run", "predict"):
        parser = commands.add_parser(mode, allow_abbrev=False,
            description="Local numeric 0/1 classification. Label 1 is user-defined; both classes are required in every training partition.")
        parser.add_argument("csv_path", type=Path)
        if mode == "predict":
            parser.add_argument("new_csv_path", type=Path)
        _options(parser, outer=mode == "run", output=True)
    plan = commands.add_parser("plan", allow_abbrev=False, help="No-fit/no-write JSON to stdout only")
    modes = plan.add_subparsers(dest="binary_plan_mode", required=True)
    for mode in ("run", "predict"):
        parser = modes.add_parser(mode, allow_abbrev=False)
        parser.add_argument("csv_path", type=Path)
        if mode == "predict":
            parser.add_argument("new_csv_path", type=Path)
        _options(parser, outer=mode == "run", output=False)


def run_binary(args):
    from ..cli import _read_training, _read_csv_with_hash, protect_input_files
    planning = args.binary_command == "plan"
    mode = args.binary_plan_mode if planning else args.binary_command
    if not planning:
        ensure_fresh_binary_output(args.out)
        inputs = [args.csv_path] + ([args.new_csv_path] if mode == "predict" else [])
        if args.features_file is not None:
            inputs.append(args.features_file)
        protect_input_files(inputs, args.out)
    frame, features, source_hash = _read_training(args)
    options = dict(target=args.target, group=args.group, features=features,
                   inner_splits=args.inner_splits, random_state=args.seed,
                   baselines=tuple(v.strip() for v in args.baselines.split(",")), source_sha256=source_hash)
    if mode == "predict":
        incoming, source_hash = _read_csv_with_hash(args.new_csv_path)
        action = plan_binary_prediction if planning else fit_predict_binary
        result = action(frame, incoming, **options, prediction_source_sha256=source_hash)
    else:
        action = plan_binary_benchmark if planning else benchmark_binary
        result = action(frame, **options, outer_splits=args.outer_splits)
    if planning:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        destination = result.write(args.out)
        if mode == "run":
            print(result.leaderboard[["model", "group_brier", "brier", "log_loss_clipped"]].to_string(index=False))
            print("Selected uses inner folds only. Raw probabilities, class supports and false positives/negatives are retained.")
        else:
            print("Full-training refit predictions, not held-out evaluation. Label 1 uses fixed p1 >= 0.5.")
        print(f"Local binary output: {destination}")
        print("No calibrated risk, OOD guarantee or autonomous physical decision is supported.")
    return 0
