"""Handwritten no-fit fixtures for strict temporal metadata JSON and its CLI."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from groupbench import cli, group_time_json as transport
from groupbench.group_time_plan import MAX_TICK, MIN_TICK, GroupSpan, TimeWindow, plan_group_forward_windows
from mlworkflow import cli as workflow
from test_group_time_plan import mixed_fixture


def metadata():
    return {"schema_id": transport.METADATA_SCHEMA_ID, "schema_version": 1,
            "time_unit": "day", "clock_id": "invented-study-clock", "gap": 2,
            "row_groups": [1, "1", "01", 1],
            "group_spans": [{"group": 1, "start": 0, "end": 1, "label_available_at": 1},
                            {"group": "1", "start": 10, "end": 12, "label_available_at": 12},
                            {"group": "01", "start": 8, "end": 12, "label_available_at": 12}],
            "windows": [{"lower": 10, "upper": 15}]}


def rendered(value):
    return transport.plan_group_forward_windows_json(json.dumps(value))


def with_token(field, token):
    value = metadata()
    value[field] = "UNIQUE_TOKEN"
    return json.dumps(value).replace('"UNIQUE_TOKEN"', token)


def run_main(arguments, entry=transport.main):
    with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
        try:
            status = entry(arguments)
        except SystemExit as error:
            status = error.code
    return status, out.getvalue(), err.getvalue()


class MetadataIdentityTests(unittest.TestCase):
    def test_documented_example_and_exact_keys(self):
        result = json.loads(rendered(metadata()))
        self.assertEqual(set(result), {"schema_id", "schema_version", "protocol", "fits_performed", "coverage_mode",
                                      "time_unit", "clock_id", "gap", "row_groups", "group_spans", "folds", "total",
                                      "tested", "untested", "limitations"})
        self.assertEqual(result["schema_id"], transport.PLAN_SCHEMA_ID)
        self.assertNotEqual(result["schema_id"], metadata()["schema_id"])
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["fits_performed"], 0)
        self.assertEqual(result["coverage_mode"], "partial_predeclared")
        self.assertEqual(result["protocol"], "group-forward-window-plan-v1")
        fold = result["folds"][0]
        self.assertEqual(set(fold), {"index", "window", "train", "test", "unused", "exclusions"})
        self.assertEqual(fold["train"]["rows"], [0, 3])
        self.assertEqual(fold["test"]["rows"], [1])
        self.assertEqual(fold["unused"]["rows"], [2])
        self.assertEqual(result["tested"]["groups"], ["1"])
        self.assertEqual(result["untested"]["groups"], [1, "01"])
        for membership in [result[k] for k in ("total", "tested", "untested")] + [fold[k] for k in ("train", "test", "unused")]:
            self.assertEqual(set(membership), {"rows", "groups", "n_rows", "n_groups"})
            self.assertEqual(membership["n_rows"], len(membership["rows"]))
            self.assertEqual(membership["n_groups"], len(membership["groups"]))
        self.assertEqual(set(fold["exclusions"][0]), {"group", "train", "test"})
        self.assertEqual(len(result["limitations"]), 5)

    def test_identity_values_and_order_are_not_coerced(self):
        ids = [1, "1", "01", " 1", "1 ", "NA", "N/A", 2**53, 2**53 + 1, MIN_TICK, MAX_TICK,
               "e\u0301", "é", "\u0000", "🙂", "\ud800"]
        value = metadata()
        value["row_groups"] = ids + [1, "1"]
        value["group_spans"] = [{"group": g, "start": 0 if i == 0 else 10,
                                 "end": 1 if i == 0 else 12, "label_available_at": 1 if i == 0 else 12}
                                for i, g in enumerate(ids)]
        result = json.loads(rendered(value), parse_int=int)
        self.assertEqual(result["row_groups"], value["row_groups"])
        self.assertEqual([type(x) for x in result["row_groups"]], [type(x) for x in value["row_groups"]])
        self.assertEqual(result["total"]["groups"], ids)
        self.assertEqual([s["group"] for s in result["group_spans"]], ids)
        self.assertEqual(result["folds"][0]["test"]["groups"], ids[1:])
        self.assertTrue(rendered(value).isascii())

    def test_signed_endpoint_ticks_negative_ticks_and_negative_zero(self):
        value = metadata()
        value["gap"] = 0
        value["group_spans"] = [{"group": 1, "start": MIN_TICK, "end": MIN_TICK, "label_available_at": MIN_TICK},
                                {"group": "1", "start": -1, "end": -1, "label_available_at": MAX_TICK},
                                {"group": "01", "start": 0, "end": 0, "label_available_at": 0}]
        value["windows"] = [{"lower": -1, "upper": MAX_TICK}]
        result = json.loads(rendered(value))
        self.assertEqual(result["group_spans"], value["group_spans"])
        source = json.dumps(value).replace('"gap": 0', '"gap": -0')
        self.assertEqual(json.loads(transport.plan_group_forward_windows_json(source))["gap"], 0)

    def test_escape_equivalent_group_identity_duplicates_are_rejected(self):
        source = json.dumps(metadata()).replace('"01"', '"\\u0031"')
        with self.assertRaises(ValueError):
            transport.plan_group_forward_windows_json(source)

    def test_bad_identities_everywhere_are_rejected_before_planning(self):
        for invalid in (True, False, None, [], {}, "", " \t "):
            for field in ("row_groups", "group_spans"):
                value = metadata()
                if field == "row_groups": value[field][0] = invalid
                else: value[field][0]["group"] = invalid
                with self.subTest(field=field, value=invalid), patch.object(transport, "plan_group_forward_windows", side_effect=AssertionError("no core")), self.assertRaises(ValueError):
                    rendered(value)

    def test_bad_ticks_gap_and_schema_booleans_are_not_integers(self):
        for invalid in (True, False, None, "1", [], {}):
            for where in ("gap", "start", "end", "label_available_at", "lower", "upper", "schema_version"):
                value = metadata()
                if where in ("start", "end", "label_available_at"): value["group_spans"][0][where] = invalid
                elif where in ("lower", "upper"): value["windows"][0][where] = invalid
                else: value[where] = invalid
                with self.subTest(where=where, value=invalid), patch.object(transport, "plan_group_forward_windows", side_effect=AssertionError("no core")), self.assertRaises(ValueError):
                    rendered(value)

    def test_decimal_exponent_constants_and_huge_integers_rejected(self):
        for token in ("1.0", "0.0", "-0.0", "1e0", "1E+0", "0e-0", "1e999", "1e-999", "NaN", "Infinity", "-Infinity",
                      str(MAX_TICK + 1), str(MIN_TICK - 1), "1" * 20, "1" * 5000):
            with self.subTest(token=token[:30]), patch.object(transport, "plan_group_forward_windows", side_effect=AssertionError("no core")), self.assertRaises(ValueError):
                transport.plan_group_forward_windows_json(with_token("gap", token))

    def test_no_float_or_numpy_pandas_conversion(self):
        with ExitStack() as stack:
            for target in ("pandas.read_csv", "pandas.DataFrame", "numpy.asarray", "numpy.array", "builtins.float"):
                stack.enter_context(patch(target, side_effect=AssertionError("conversion forbidden")))
            result = json.loads(rendered(metadata()))
            self.assertEqual(result["row_groups"], [1, "1", "01", 1])


class StrictMetadataTests(unittest.TestCase):
    def test_duplicate_keys_all_object_levels_and_escape_equivalence(self):
        text = json.dumps(metadata())
        cases = [text.replace('"gap": 2', '"gap": 2, "gap": 2'),
                 text.replace('"start": 0', '"start": 0, "start": 0'),
                 text.replace('"lower": 10', '"lower": 10, "lower": 10'),
                 text.replace('"gap": 2', '"gap": 2, "g\\u0061p": 2')]
        for source in cases:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                transport.plan_group_forward_windows_json(source)

    def test_unknown_missing_and_malformed_objects(self):
        for where in ("root", "span", "window"):
            for action in ("missing", "unknown", "array", "null", "scalar"):
                value = metadata()
                obj = value if where == "root" else value["group_spans" if where == "span" else "windows"][0]
                if action == "missing": del obj[next(iter(obj))]
                elif action == "unknown": obj["features"] = "PRIVATE_SOURCE"
                else:
                    replacement = {"array": [], "null": None, "scalar": "x"}[action]
                    if where == "root": value = replacement
                    else: value["group_spans" if where == "span" else "windows"][0] = replacement
                with self.subTest(where=where, action=action), self.assertRaises(ValueError): rendered(value)
        for field in ("features", "targets", "data_path", "url", "model"):
            value = metadata(); value[field] = "PRIVATE"
            with self.assertRaises(ValueError): rendered(value)

    def test_schema_and_declarations_require_exact_values(self):
        for key, invalids in (("schema_id", [None, 1, "other", transport.PLAN_SCHEMA_ID]),
                              ("schema_version", [0, 2, "1"]),
                              ("time_unit", [None, 1, "", " ", " day", "day "]),
                              ("clock_id", [None, 1, "", " ", " clock", "clock "])):
            for invalid in invalids:
                value = metadata(); value[key] = invalid
                with self.assertRaises(ValueError): rendered(value)

    def test_invalid_json_bom_surrogates_and_trailing_data(self):
        for source in ("", "PRIVATE", "[", "{", "]", "true", "[]", "{} {}", "\ufeff" + json.dumps(metadata()),
                       json.dumps(metadata()) + "null", '\ud800', '{"x": "PRIVATE\n"}'):
            with self.assertRaises(ValueError) as caught: transport.plan_group_forward_windows_json(source)
            self.assertNotIn("PRIVATE", str(caught.exception))
        for source in (None, b"{}", {}, []):
            with self.assertRaises(ValueError): transport.plan_group_forward_windows_json(source)

    def test_depth_exact_limit_and_escaped_brackets(self):
        transport._check_depth("[" * transport.MAX_DEPTH + "0" + "]" * transport.MAX_DEPTH)
        for source in ("[" * 17 + "0" + "]" * 17, '{"x":' * 17, "[" * 10000):
            with self.assertRaisesRegex(ValueError, "nesting"): transport.plan_group_forward_windows_json(source)
        value = metadata(); value["clock_id"] = '[{\\"}]' * 30
        self.assertEqual(json.loads(rendered(value))["clock_id"], value["clock_id"])
        with patch.object(transport.json, "loads", side_effect=RecursionError("PRIVATE")), self.assertRaises(ValueError):
            transport.plan_group_forward_windows_json(json.dumps(metadata()))

    def test_utf8_byte_limit_exact_and_over(self):
        source = json.dumps(metadata())
        exact = source + " " * (transport.MAX_INPUT_BYTES - len(source))
        self.assertEqual(json.loads(transport.plan_group_forward_windows_json(exact))["fits_performed"], 0)
        for too_big in (exact + " ", "界" * (transport.MAX_INPUT_BYTES // 3 + 1)):
            with patch.object(transport, "plan_group_forward_windows", side_effect=AssertionError("no core")), self.assertRaises(ValueError):
                transport.plan_group_forward_windows_json(too_big)

    def test_structural_size_limits_exact_over_and_empty(self):
        for field, cap in (("row_groups", transport.MAX_ROWS), ("group_spans", transport.MAX_GROUPS), ("windows", transport.MAX_WINDOWS)):
            value = metadata(); value[field] = [value[field][0]] * cap
            self.assertEqual(len(transport._metadata(json.dumps(value))[field]), cap)
            for invalid in ([], value[field] + value[field][:1], {}, "x", None):
                value[field] = invalid
                with patch.object(transport, "plan_group_forward_windows", side_effect=AssertionError("no core")), self.assertRaises(ValueError): rendered(value)


class PlanEquivalenceTests(unittest.TestCase):
    def assert_equivalent(self, value):
        plan = plan_group_forward_windows(value["row_groups"], group_spans=[GroupSpan(**s) for s in value["group_spans"]],
                                          windows=[TimeWindow(**w) for w in value["windows"]], gap=value["gap"],
                                          time_unit=value["time_unit"], clock_id=value["clock_id"])
        result = json.loads(rendered(value))
        for name in ("row_groups", "group_spans", "folds", "total", "tested", "untested"):
            self.assertEqual(result[name], transport._document(plan)[name])
        for fold, core in zip(result["folds"], plan.folds):
            for name in ("train", "test", "unused"):
                part, saved = getattr(core, name), fold[name]
                self.assertEqual(saved["rows"], list(part.rows)); self.assertEqual(saved["groups"], list(part.groups))
                self.assertEqual((saved["n_rows"], saved["n_groups"]), (part.n_rows, part.n_groups))
            self.assertEqual(fold["exclusions"], [asdict(e) for e in core.exclusions])
        return result

    def test_mixed_fixture_exact_and_later_training_is_not_coverage(self):
        rows, spans, windows = mixed_fixture()
        value = metadata(); value.update(row_groups=rows, group_spans=[asdict(s) for s in spans], windows=[asdict(w) for w in windows])
        result = self.assert_equivalent(value)
        self.assertIn("cross", result["untested"]["groups"])
        self.assertIn("cross", result["folds"][1]["train"]["groups"])
        self.assertEqual(result["tested"]["rows"], [1, 4, 6, 10])
        value["group_spans"].reverse()
        self.assertEqual(result, json.loads(rendered(value)))

    def test_cutoff_upper_crossing_and_delayed_labels(self):
        value = metadata()
        ids = ["warm", "test", "equal", "upper", "lower", "both", "later", "delay"]
        value["row_groups"] = ids
        starts = [0, 10, 0, 11, 9, 9, 15, 0]
        ends = [1, 14, 1, 15, 11, 15, 15, 1]
        labels = [1, 99, 8, 15, 11, 15, 15, 99]
        value["group_spans"] = [dict(group=g, start=s, end=e, label_available_at=l) for g,s,e,l in zip(ids, starts, ends, labels)]
        result = self.assert_equivalent(value)
        fold = result["folds"][0]
        self.assertEqual(fold["train"]["groups"], ["warm"])
        self.assertEqual(fold["test"]["groups"], ["test"])
        self.assertEqual(fold["exclusions"][2]["train"], "gap_not_elapsed_before_window")
        self.assertEqual([e["test"] for e in fold["exclusions"]][3:7],
                         ["crosses_upper_boundary", "crosses_lower_boundary", "crosses_both_boundaries", "at_or_after_window"])

    def test_core_errors_all_or_nothing_and_private(self):
        cases = []
        value = metadata(); value["group_spans"].pop(); cases.append(value)
        value = metadata(); value["group_spans"].append(dict(group="extra", start=0, end=1, label_available_at=1)); cases.append(value)
        value = metadata(); value["group_spans"].append(value["group_spans"][0]); cases.append(value)
        value = metadata(); value["gap"] = -1; cases.append(value)
        value = metadata(); value["group_spans"][0]["label_available_at"] = MAX_TICK; cases.append(value)
        value = metadata(); value["group_spans"][0]["end"] = -1; cases.append(value)
        value = metadata(); value["windows"] = [{"lower": 15, "upper": 15}]; cases.append(value)
        value = metadata(); value["windows"] *= 2; cases.append(value)
        value = metadata(); value["windows"] += [{"lower": 100, "upper": 101}]; cases.append(value)
        value = metadata(); value["windows"] = [{"lower": -2, "upper": 100}]; cases.append(value)
        for value in cases:
            with self.assertRaises(ValueError) as caught: rendered(value)
            self.assertNotIn("100", str(caught.exception))

    def test_once_only_after_preflight_and_no_mutation(self):
        value = metadata(); before = copy.deepcopy(value)
        with patch.object(transport, "plan_group_forward_windows", wraps=plan_group_forward_windows) as planner:
            self.assert_equivalent(value)
        planner.assert_called_once()
        self.assertEqual(value, before)


class ResourcePreflightTests(unittest.TestCase):
    def test_work_exact_limit_then_over_before_planner(self):
        value = metadata()
        value["row_groups"] = list(range(250)) * 19  # 4750 rows, 250 groups, 50 windows.
        value["group_spans"] = [dict(group=i, start=0 if i < 200 else 10 + 2*(i-200),
                                     end=0 if i < 200 else 10 + 2*(i-200),
                                     label_available_at=0 if i < 200 else 10 + 2*(i-200)) for i in range(250)]
        value["windows"] = [dict(lower=10+2*i, upper=11+2*i) for i in range(50)]
        value["gap"] = 0
        self.assertEqual(len(value["windows"])*(len(value["row_groups"])+len(value["group_spans"])), transport.MAX_WORK)
        result = rendered(value)
        self.assertLessEqual(len(result), transport._output_bound(value))
        value["row_groups"].append(0)
        with patch.object(transport, "plan_group_forward_windows", side_effect=AssertionError("no core")), self.assertRaisesRegex(ValueError, "work limit"):
            rendered(value)

    def test_small_input_large_expansion_rejected_before_core(self):
        value = metadata(); value["row_groups"] = [1] * 1000; value["windows"] *= 1000
        self.assertLess(len(json.dumps(value)), 100_000)
        with patch.object(transport, "plan_group_forward_windows", side_effect=AssertionError("no core")), self.assertRaisesRegex(ValueError, "work limit"):
            rendered(value)

    def test_escaped_long_group_amplification_rejected_before_core(self):
        value = metadata(); value["row_groups"] = ["x" * 40000, "\u0000" * 40000]
        value["group_spans"] = [dict(group=g, start=i*10, end=i*10, label_available_at=i*10) for i,g in enumerate(value["row_groups"])]
        value["windows"] *= 40
        self.assertLess(len(json.dumps(value).encode()), transport.MAX_INPUT_BYTES)
        self.assertLess(40*4, transport.MAX_WORK)
        self.assertGreater(transport._output_bound(value), transport.MAX_OUTPUT_BYTES)
        with patch.object(transport, "plan_group_forward_windows", side_effect=AssertionError("no core")), self.assertRaisesRegex(ValueError, "serialized-output bound"):
            rendered(value)

    def test_output_preflight_exact_boundary_and_next_byte(self):
        value = metadata(); bound = transport._output_bound(value)
        with patch.object(transport, "MAX_OUTPUT_BYTES", bound):
            self.assertEqual(json.loads(rendered(value))["fits_performed"], 0)
        with patch.object(transport, "MAX_OUTPUT_BYTES", bound - 1), patch.object(transport, "plan_group_forward_windows", side_effect=AssertionError("no core")), self.assertRaises(ValueError):
            rendered(value)

    def test_final_serialized_size_includes_newline_independent_of_estimator(self):
        value = metadata(); size = len(rendered(value).encode())
        with patch.object(transport, "_output_bound", return_value=0), patch.object(transport, "MAX_OUTPUT_BYTES", size):
            self.assertEqual(len(rendered(value)), size)
        with patch.object(transport, "_output_bound", return_value=0), patch.object(transport, "MAX_OUTPUT_BYTES", size - 1), patch.object(transport, "plan_group_forward_windows", wraps=plan_group_forward_windows) as planner, self.assertRaises(ValueError):
            rendered(value)
        planner.assert_called_once()

    def test_bound_exceeds_actual_for_handwritten_escaped_endpoint_cases(self):
        for identity in ("a", "\\\"\t\r\n\u0000", "界🙂", "\ud800", "x" * 4096, MAX_TICK):
            value = metadata(); value["row_groups"][0] = identity; value["row_groups"][3] = identity
            value["group_spans"][0]["group"] = identity
            value["group_spans"][0].update(start=MIN_TICK, end=MIN_TICK, label_available_at=MIN_TICK)
            value["group_spans"][1]["label_available_at"] = MAX_TICK
            value["windows"][0]["upper"] = MAX_TICK
            value["gap"] = 0
            value["clock_id"] = "界🙂" * 100
            output = rendered(value)
            self.assertGreaterEqual(transport._output_bound(value), len(output.encode("utf-8")))
            self.assertEqual(output, transport._json(json.loads(output)) + "\n")
        self.assertLess(sum(len(transport._json(s)) for s in transport._LIMITATIONS), 8192)


class MetadataFileAndCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "PRIVATE_metadata.json"
        self.source.write_text(json.dumps(metadata()), encoding="utf-8")

    def test_regular_file_symlink_and_neighbors_unchanged(self):
        alias = self.root / "alias.json"; alias.symlink_to(self.source)
        for name in cli.OUTPUT_NAMES: (self.root / name).write_bytes(b"private neighboring artifact\r\n")
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        result = run_main([str(alias)])
        self.assertEqual(result[0], 0); self.assertEqual(result[2], "")
        self.assertEqual(result[1], rendered(metadata()))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        self.assertTrue(alias.is_symlink())

    def test_missing_directory_fifo_device_invalid_encoding_rejected(self):
        sources = [self.root / "missing_PRIVATE", self.root]
        if hasattr(os, "mkfifo"):
            fifo = self.root / "fifo"; os.mkfifo(fifo); sources.append(fifo)
        if Path("/dev/null").exists(): sources.append(Path("/dev/null"))
        for raw in (b"\xff", b"\xef\xbb\xbf{}", b"PRIVATE", b"{\"gap\":NaN}"):
            self.source.write_bytes(raw)
            sources.append(self.source)
            status, stdout, stderr = run_main([str(self.source)])
            self.assertEqual((status, stdout), (2, "")); self.assertNotIn("PRIVATE", stderr)
        for source in sources:
            status, stdout, stderr = run_main([str(source)])
            self.assertEqual((status, stdout), (2, "")); self.assertNotIn("PRIVATE", stderr)

    def test_oversized_file_before_content_read_and_descriptor_closed(self):
        with self.source.open("wb") as handle: handle.truncate(transport.MAX_INPUT_BYTES + 1)
        opened = []
        real = cli._open_csv_source
        @contextmanager
        def wrapped(path):
            with real(path) as source:
                opened.append(source)
                class Guard:
                    def fileno(self): return source.fileno()
                    def read(self, _size): raise AssertionError("oversized content read")
                yield Guard()
        with patch.object(cli, "_open_csv_source", wrapped):
            self.assertEqual(run_main([str(self.source)])[:2], (2, ""))
        self.assertTrue(opened[0].closed)

    def test_one_retained_read_no_reopen_for_parsing(self):
        from groupbench import saved_scoring
        with patch.object(saved_scoring, "_read_outcomes", wraps=saved_scoring._read_outcomes) as reader, \
             patch.object(cli, "_open_csv_source", wraps=cli._open_csv_source) as opener:
            self.assertEqual(run_main([str(self.source)])[0], 0)
        reader.assert_called_once_with(self.source); opener.assert_called_once_with(self.source)

    def test_growth_stream_cap_and_source_closed(self):
        from groupbench import saved_scoring
        real = cli._open_csv_source; opened = []
        @contextmanager
        def grows(path):
            with real(path) as source:
                opened.append(source)
                class Growing:
                    def fileno(self): return source.fileno()
                    def read(self, size): return b" " * size
                yield Growing()
        with patch.object(cli, "_open_csv_source", grows), patch.object(saved_scoring, "MAX_OUTCOMES_BYTES", 1024):
            self.assertEqual(run_main([str(self.source)])[:2], (2, ""))
        self.assertTrue(opened[0].closed)

    def test_path_replacement_and_postread_change_detected(self):
        real = cli._open_csv_source
        owner = self
        for mode in ("replace", "mutate"):
            self.source.write_text(json.dumps(metadata()))
            opened = []
            @contextmanager
            def changes(path):
                with real(path) as source:
                    opened.append(source)
                    changed = False
                    class Changing:
                        def fileno(self): return source.fileno()
                        def read(self, size):
                            nonlocal changed
                            raw = source.read(size)
                            if not changed:
                                changed = True
                                if mode == "replace":
                                    other = owner.root / "replacement.json"; other.write_bytes(owner.source.read_bytes()); other.replace(owner.source)
                                else:
                                    with owner.source.open("ab") as handle: handle.write(b" ")
                            return raw
                    yield Changing()
            with patch.object(cli, "_open_csv_source", changes):
                self.assertEqual(run_main([str(self.source)])[:2], (2, ""))
            self.assertTrue(opened[0].closed)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO test requires POSIX")
    def test_fifo_replacement_at_open_is_nonblocking_and_closed(self):
        real_open, real_close = os.open, os.close
        descriptors, flags_seen, closed = [], [], []
        def swapped(path, flags, *args, **kwargs):
            self.source.unlink(); os.mkfifo(self.source)
            descriptor = real_open(path, flags, *args, **kwargs)
            descriptors.append(descriptor); flags_seen.append(flags); return descriptor
        def closing(fd):
            closed.append(fd); return real_close(fd)
        with patch.object(cli.os, "open", side_effect=swapped), patch.object(cli.os, "close", side_effect=closing):
            self.assertEqual(run_main([str(self.source)])[:2], (2, ""))
        self.assertEqual(closed, descriptors)
        self.assertTrue(flags_seen[0] & os.O_NONBLOCK)

    def test_cli_rejects_output_model_options_abbreviations_before_read(self):
        for arguments in ([], ["--out", "PRIVATE"], [str(self.source), "--output", "PRIVATE"],
                          [str(self.source), "--overwrite"], [str(self.source), "--features", "PRIVATE"],
                          [str(self.source), "--target", "PRIVATE"], [str(self.source), "--baselines", "PRIVATE"],
                          [str(self.source), "--h"], [str(self.source), "extra_PRIVATE"]):
            with patch.object(transport, "_read_metadata", side_effect=AssertionError("must parse before read")):
                status, stdout, stderr = run_main(arguments)
            self.assertEqual((status, stdout), (2, "")); self.assertNotIn("PRIVATE", stderr)

    def test_help_discovery_forwarding_and_sys_argv_unchanged(self):
        status, stdout, stderr = run_main(["--help"], cli.main)
        self.assertEqual(status, 0); self.assertIn("plan-group-forward-windows", stdout)
        self.assertEqual(run_main(["--help"])[0], 0)
        args = ["plan-group-forward-windows", str(self.source)]
        saved = sys.argv[:]
        with patch.object(transport, "main", return_value=7) as delegated:
            self.assertEqual(cli.main(args), 7)
        delegated.assert_called_once_with([str(self.source)])
        self.assertEqual(args, ["plan-group-forward-windows", str(self.source)])
        self.assertEqual(sys.argv, saved)
        self.assertEqual(run_main(args, cli.main), run_main(["tabular", *args], workflow.main))

    def test_aliases_module_and_workflow_entrypoints(self):
        project = Path(__file__).resolve().parents[1]
        config = (project / "pyproject.toml").read_text()
        self.assertIn('mlcalc = "groupbench.cli:main"', config)
        self.assertIn('groupbench = "groupbench.cli:main"', config)
        env = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        for module, arguments in (("groupbench", ["plan-group-forward-windows"]),
                                  ("groupbench.group_time_json", []),
                                  ("mlworkflow", ["tabular", "plan-group-forward-windows"])):
            run = subprocess.run([sys.executable, "-m", module, *arguments, str(self.source)], env=env,
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr); self.assertEqual(run.stdout, rendered(metadata()))

    def test_existing_early_dispatch_and_measurement_plan_unchanged(self):
        for command, target in (("compare-predictions", "groupbench.saved_scoring.compare_main"),
                                ("score-predictions", "groupbench.saved_scoring.main"),
                                ("evaluate-binary", "groupbench.binary.saved_evaluation.main"),
                                ("evaluate-intervals", "groupbench.interval_evaluation.main")):
            with patch(target, return_value=7) as delegated:
                self.assertEqual(cli.main([command, "SENTINEL"]), 7)
            delegated.assert_called_once_with(["SENTINEL"])
        with patch.object(workflow, "regression_main", return_value=7) as delegated:
            self.assertEqual(workflow.main(["plan", "SENTINEL"]), 7)
        delegated.assert_called_once_with(["plan", "SENTINEL"])
        for mode in ("run", "predict"):
            status, stdout, _ = run_main(["plan", mode, "--help"], cli.main)
            self.assertEqual(status, 0); self.assertIn("--inner-splits", stdout)
            self.assertNotIn("plan-group-forward-windows", stdout)

    def test_failure_before_stdout_and_private_errors_not_echoed(self):
        for failure in (ValueError("PRIVATE_PATH"), TypeError("PRIVATE_VALUE"), OverflowError("PRIVATE"), RecursionError("PRIVATE")):
            with patch.object(transport, "_document", side_effect=failure):
                status, stdout, stderr = run_main([str(self.source)])
            self.assertEqual((status, stdout), (2, "")); self.assertNotIn("PRIVATE", stderr)
        with patch.object(transport, "_json", side_effect=ValueError("PRIVATE")):
            self.assertEqual(run_main([str(self.source)])[:2], (2, ""))

    def test_stdout_write_failure_is_not_claimed_atomic_or_successful(self):
        class Broken(io.StringIO):
            def write(self, text):
                super().write(text[:10])
                raise OSError("PRIVATE")
        output = Broken()
        with redirect_stdout(output), redirect_stderr(io.StringIO()) as error:
            self.assertEqual(transport.main([str(self.source)]), 2)
        self.assertTrue(output.getvalue())
        self.assertIn("could not be written completely", error.getvalue())
        self.assertNotIn("PRIVATE", error.getvalue())

    def test_no_fitting_rng_csv_execution_or_bundle_writes(self):
        targets = ("sklearn.pipeline.Pipeline.fit", "sklearn.preprocessing.StandardScaler.fit", "sklearn.dummy.DummyRegressor.fit",
                   "numpy.random.seed", "numpy.random.default_rng", "random.random", "csv.reader", "pandas.read_csv",
                   "groupbench.cli.benchmark", "groupbench.cli.fit_predict", "groupbench.cli.plan_benchmark", "groupbench.cli.plan_prediction",
                   "groupbench.report.write_payloads", "groupbench.saved_scoring._outcomes")
        before = self.source.read_bytes()
        with ExitStack() as stack:
            for target in targets: stack.enter_context(patch(target, side_effect=AssertionError("non-planner activity")))
            self.assertEqual(run_main([str(self.source)])[0], 0)
            with patch.object(transport, "_read_metadata", return_value='{"targets": "PRIVATE"}'):
                self.assertEqual(run_main([str(self.source)])[:2], (2, ""))
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(list(self.root.iterdir()), [self.source])
        with patch("builtins.open", side_effect=AssertionError("I/O forbidden")), patch("os.open", side_effect=AssertionError("I/O forbidden")):
            self.assertEqual(json.loads(rendered(metadata()))["fits_performed"], 0)


if __name__ == "__main__":
    unittest.main()
