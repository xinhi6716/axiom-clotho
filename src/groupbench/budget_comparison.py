"""Strictly descriptive, local comparison of two saved validation budgets."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import stat
import sys


META = {"metric_definition", "value_unit", "independent_unit_definition", "normalization", "direction"}
LIMITATIONS = [
    "Descriptive only: no confidence intervals, p-values, causal or population claims.",
    "Unit independence, metric definitions, normalization and provenance are user declarations, not verified facts.",
    "Repeated observations are averaged within each declared unit; units receive equal weight.",
    "No failed or missing cell is dropped: invalid or incomplete input rejects the entire comparison.",
]


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must have exactly these keys: {sorted(keys)}")


def _label(value):
    if not isinstance(value, str) or not value or value != value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError("Labels must be nonempty, trimmed text without control characters")
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("Values must be finite JSON numbers, not strings or booleans")
    try:
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("Values must be finite")
        if len(number.as_tuple().digits) > 100 or abs(number.as_tuple().exponent) > 400:
            raise ValueError("Value exceeds numeric precision/range limits")
        result = Fraction(number)
        _output(result)
        return result
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError("Value outside supported numeric range") from exc


def _output(value):
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("Result outside supported numeric range") from exc
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ValueError("Result outside supported numeric range")
    return result


def _mean(values):
    return sum(values, Fraction()) / len(values)


def _outcome(gain):
    return "win" if gain > 0 else "harm" if gain < 0 else "tie"


def compare_saved_budgets(document):
    """Compare ordered method B against A at ordered budgets low then high.

    All values must already be on the same declared metric scale. No fitting,
    selection, normalization or provenance verification occurs here.
    """
    _keys(document, {"schema_version", "methods", "budgets", "metadata", "rows"}, "document")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    methods, budgets = document["methods"], document["budgets"]
    for labels in (methods, budgets):
        if not isinstance(labels, list) or len(labels) != 2:
            raise ValueError("Exactly two ordered methods and two ordered budgets are required")
        if len({_label(x) for x in labels}) != 2:
            raise ValueError("Method and budget labels must be distinct")
    metadata = document["metadata"]
    _keys(metadata, META, "metadata")
    for value in metadata.values():
        _label(value)
    if metadata["direction"] not in ("minimize", "maximize"):
        raise ValueError("direction must be minimize or maximize")
    rows = document["rows"]
    if not isinstance(rows, list) or not rows or len(rows) > 20000:
        raise ValueError("rows must be a nonempty list of at most 20000 cells")
    cells = {}
    repeats = {}
    for row in rows:
        _keys(row, {"unit_id", "repeat_id", "method", "budget", "value", "status", "metric_definition", "value_unit"}, "row")
        unit, repeat = _label(row["unit_id"]), _label(row["repeat_id"])
        if row["method"] not in methods or row["budget"] not in budgets:
            raise ValueError("Unknown method or budget")
        if row["status"] != "ok":
            raise ValueError("Failed/non-ok cell: comparison rejected; no failures are omitted")
        if any(row[key] != metadata[key] for key in ("metric_definition", "value_unit")):
            raise ValueError("Mismatched metric definition or value unit")
        key = (unit, repeat, row["budget"], row["method"])
        if key in cells:
            raise ValueError("Duplicate unit/repeat/budget/method key")
        cells[key] = _number(row["value"])
        repeats.setdefault(unit, set()).add(repeat)
    for unit, ids in repeats.items():
        for repeat in ids:
            for budget in budgets:
                for method in methods:
                    if (unit, repeat, budget, method) not in cells:
                        raise ValueError("Missing matched combination: every unit/repeat needs both methods at both budgets")
    sign = 1 if metadata["direction"] == "maximize" else -1
    unit_rows, observation_rows, gains_by_budget = [], [], {budget: [] for budget in budgets}
    interactions = []
    for unit in sorted(repeats):
        means, gains = {}, {}
        for budget in budgets:
            means[budget] = {method: _mean([cells[unit, r, budget, method] for r in sorted(repeats[unit])]) for method in methods}
            gains[budget] = sign * (means[budget][methods[1]] - means[budget][methods[0]])
            gains_by_budget[budget].append(gains[budget])
        interaction = gains[budgets[1]] - gains[budgets[0]]
        interactions.append(interaction)
        unit_rows.append({"unit_id": unit, "repeat_count": len(repeats[unit]),
                          "method_means": {b: {m: _output(v) for m, v in mm.items()} for b, mm in means.items()},
                          "method_b_gain": {b: _output(g) for b, g in gains.items()},
                          "outcomes": {b: _outcome(g) for b, g in gains.items()},
                          "selector_by_budget_interaction": _output(interaction)})
        for repeat in sorted(repeats[unit]):
            values = {b: {m: cells[unit, repeat, b, m] for m in methods} for b in budgets}
            obs_gains = {b: sign * (v[methods[1]] - v[methods[0]]) for b, v in values.items()}
            observation_rows.append({"unit_id": unit, "repeat_id": repeat,
                                     "values": {b: {m: _output(v) for m, v in mm.items()} for b, mm in values.items()},
                                     "method_b_gain": {b: _output(g) for b, g in obs_gains.items()},
                                     "outcomes": {b: _outcome(g) for b, g in obs_gains.items()}})
    summaries = {b: {"mean_unit_gain": _output(_mean(gs)),
                     "strict_wins": sum(g > 0 for g in gs), "ties": sum(g == 0 for g in gs),
                     "harms": sum(g < 0 for g in gs)} for b, gs in gains_by_budget.items()}
    return {"schema_version": 1, "status": "complete_descriptive_comparison", "metadata": dict(metadata),
            "methods": list(methods), "budgets": list(budgets), "provenance_verified": False,
            "gain_convention": "Positive means B improves over A; interaction = high-budget gain minus low-budget gain.",
            "coverage": {"units": len(repeats), "matched_repeats": len(observation_rows), "input_cells": len(rows), "omitted_cells": 0},
            "by_budget": summaries, "mean_selector_by_budget_interaction": _output(_mean(interactions)),
            "units": unit_rows, "observations": observation_rows, "limitations": list(LIMITATIONS)}


def load_document(text):
    if len(text) > 8_000_000:
        raise ValueError("Input exceeds 8 MB character limit")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON constant")
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_float=Decimal, parse_constant=constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds supported depth") from exc


def _read_input(path):
    """Read only a regular local file, following explicitly supplied symlinks.

    Check before opening and again on the descriptor. Nonblocking open prevents
    waiting on a FIFO swapped in between checks where O_NONBLOCK is supported.
    These checks are not a filesystem-atomic snapshot or a wall-clock deadline.
    """
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Saved comparison input must be a regular local file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                 | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Saved comparison input must be a regular local file")
        with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as source:
            return source.read(8_000_001)
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local saved-comparison JSON; report is written to stdout")
    args = parser.parse_args(argv)
    try:
        report = compare_saved_budgets(load_document(_read_input(args.input)))
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, TypeError) as exc:
        print(f"budget comparison rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
