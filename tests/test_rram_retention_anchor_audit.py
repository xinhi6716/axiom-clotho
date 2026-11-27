import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "audit_rram_retention_anchors.py"
IDS = ("lot", "wafer", "die", "cell", "episode")
TIMES = ("nominal_us", "clock", "epoch", "tick_us", "error_us")
BASE = ("privateLot", "W1", "D1", "C1", "E1")


def episode(key=BASE, early=1_000_000, final=100_000_000_000, error=0):
    origin = 47
    return [
        [*key, nominal, "counter", "epoch", origin + delta, error]
        for nominal, delta in (
            (0, 0), (1_000_000, early), (100_000_000_000, final)
        )
    ]


class RetentionAnchorAudit(unittest.TestCase):
    def run_capture(self, roster, rows, capture_header=IDS + TIMES, *, extra=(), block_map=None, stress=None, readout_paths=None, read_bias=None, apertures=None, schedule=None, terminations=None, planned_paths=None, native_reads=None, planned_bias=None, planned_roles=None, logged_roles=None, aperture_widths=None):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, header, payload in (
                    ("roster.csv", IDS, roster), ("capture.csv", capture_header, rows)):
                with (folder / name).open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((header, *payload))
            if block_map is not None:
                block_map_path = folder / "blocks.csv"
                with block_map_path.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((IDS + ("acquisition_block",), *block_map))
                extra = (*extra, "--acquisition-block-map", str(block_map_path))
            if stress is not None:
                history = folder / "history.csv"
                with history.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((IDS + ("bias_0_to_1s", "bias_1s_to_100ks"), *stress))
                extra = (*extra, "--stress-history", str(history))
            if readout_paths is not None:
                paths = folder / "paths.csv"
                with paths.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((IDS + ("nominal_us", "readout_path"), *readout_paths))
                extra = (*extra, "--readout-paths", str(paths))
            if planned_paths is not None:
                plan = folder / "route_plan.csv"
                with plan.open("w", newline="", encoding="utf-8") as stream:
                    cols = IDS + ("nominal_us", "planned_readout_path")
                    csv.writer(stream).writerows((cols, *planned_paths))
                extra = (*extra, "--planned-readout-paths", str(plan))
            if native_reads is not None:
                native = folder / "native_reads.csv"
                with native.open("w", newline="", encoding="utf-8") as stream:
                    cols = IDS + ("nominal_us", "native_read_namespace", "native_read_id")
                    csv.writer(stream).writerows((cols, *native_reads))
                extra = (*extra, "--native-read-identities", str(native))
            if read_bias is not None:
                bias_path = folder / "bias.csv"
                with bias_path.open("w", newline="", encoding="utf-8") as stream:
                    columns = IDS + ("nominal_us", "observed_read_bias_uV", "read_bias_error_uV")
                    csv.writer(stream).writerows((columns, *read_bias))
                extra = (*extra, "--read-bias", str(bias_path))
            if planned_bias is not None:
                path = folder / "bias_plan.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    cols = IDS + ("nominal_us", "planned_bias_min_uV", "planned_bias_max_uV")
                    csv.writer(stream).writerows((cols, *planned_bias))
                extra = (*extra, "--planned-read-bias", str(path))
            for mode, role_rows in (("planned", planned_roles), ("logged", logged_roles)):
                if role_rows is not None:
                    path = folder / f"{mode}_roles.csv"
                    with path.open("w", newline="", encoding="utf-8") as stream:
                        cols = IDS + ("nominal_us", f"{mode}_event_role")
                        csv.writer(stream).writerows((cols, *role_rows))
                    extra = (*extra, f"--{mode}-event-roles", str(path))
            if apertures is not None:
                aperture_file = folder / "apertures.csv"
                with aperture_file.open("w", newline="", encoding="utf-8") as stream:
                    fields = IDS + ("nominal_us", "aperture_open_offset_us", "aperture_close_offset_us")
                    csv.writer(stream).writerows((fields, *apertures))
                extra = (*extra, "--read-apertures", str(aperture_file))
            if aperture_widths is not None:
                width_file = folder / "width_plan.csv"
                with width_file.open("w", newline="", encoding="utf-8") as stream:
                    fields = IDS + ("nominal_us", "min_aperture_width_us", "max_aperture_width_us")
                    csv.writer(stream).writerows((fields, *aperture_widths))
                extra = (*extra, "--planned-aperture-widths", str(width_file))
            if schedule is not None:
                schedule_path = folder / "schedule.csv"
                with schedule_path.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((IDS + ("nominal_us",), *schedule))
                extra = (*extra, "--commanded-schedule", str(schedule_path))
            if terminations is not None:
                path = folder / "termination.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows((IDS + ("termination_disposition",), *terminations))
                extra = (*extra, "--termination-dispositions", str(path))
            return subprocess.run(
                [sys.executable, str(TOOL), "--frozen-roster", str(folder / "roster.csv"),
                 "--capture", str(folder / "capture.csv"), *extra],
                text=True, capture_output=True, check=False,
            )

    def test_archived_roster_digest_binds_exact_bytes_before_capture_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            roster = folder / "roster.csv"
            capture = folder / "capture.csv"
            original = (",".join(IDS) + "\r\n" + ",".join(BASE) + "\r\n").encode()
            digest = hashlib.sha256(original).hexdigest()
            roster.write_bytes(original)
            with capture.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows((IDS + TIMES, *episode()))

            def run(*extra):
                return subprocess.run(
                    [sys.executable, str(TOOL), "--frozen-roster", str(roster),
                     "--capture", str(capture), *extra],
                    text=True, capture_output=True, check=False,
                )

            before = json.loads(run().stdout)
            for byte_change in (
                original.replace(b"\r\n", b"\n"),
                b"\xef\xbb\xbf" + original,
                original.replace(b"E1", b"E2"),
                original + b" ",
            ):
                roster.write_bytes(byte_change)
                r = run("--frozen-roster-sha256", digest)
                self.assertEqual(r.returncode, 2)
                self.assertIn("SHA-256 mismatch", r.stderr)
                self.assertFalse(r.stdout)
                self.assertNotIn(digest, r.stderr)
                self.assertNotIn("privateLot", r.stderr)
            roster.write_bytes(original)
            r = run("--frozen-roster-sha256", digest)
            self.assertEqual(r.returncode, 0, r.stderr)
            after = json.loads(r.stdout)
            self.assertIs(after.pop("frozen_roster_byte_digest_verified"), True)
            self.assertEqual(before, after)
            self.assertNotIn(digest, r.stdout)

            capture.unlink()
            r = run("--frozen-roster-sha256", "0" * 64)
            self.assertEqual(r.returncode, 2)
            self.assertIn("SHA-256 mismatch", r.stderr)
            self.assertFalse(r.stdout)
            r = run("--frozen-roster-sha256", digest.upper())
            self.assertEqual(r.returncode, 2)
            self.assertIn("64 lowercase", r.stderr)
            self.assertFalse(r.stdout)

    def test_capture_receipt_binds_timestamp_admission_and_exact_source_bytes(self):
        rows = episode()
        capture_text = ",".join(IDS + TIMES) + "\r\n" + "".join(
            ",".join(map(str, row)) + "\r\n" for row in rows
        )
        digest = hashlib.sha256(capture_text.encode()).hexdigest()
        receipt = ("--capture-sha256", digest)
        baseline = self.run_capture([BASE], rows)
        verified = self.run_capture([BASE], rows, extra=receipt)
        self.assertEqual((baseline.returncode, verified.returncode), (0, 0))
        report = json.loads(verified.stdout)
        self.assertIs(report.pop("timestamp_capture_byte_digest_verified"), True)
        self.assertEqual(report, json.loads(baseline.stdout))
        self.assertNotIn(digest, verified.stdout)

        revised = episode(final=79_999_999_999)
        unpinned = self.run_capture([BASE], revised)
        self.assertEqual(unpinned.returncode, 1)
        self.assertEqual(json.loads(unpinned.stdout)["frozen_episodes_by_primary_reason"],
                         {"terminal_outside_window": 1})
        for changed, header in (
            (revised, IDS + TIMES),
            (list(reversed(rows)), IDS + TIMES),
            ([], ("wrong_header",)),  # Fail on digest before interpreting the CSV.
        ):
            result = self.run_capture([BASE], changed, capture_header=header, extra=receipt)
            self.assertEqual(result.returncode, 2)
            self.assertIn("timestamp capture SHA-256 mismatch", result.stderr)
            self.assertFalse(result.stdout)
            self.assertNotIn(digest, result.stderr)
            self.assertNotIn("privateLot", result.stderr)
        invalid = self.run_capture([BASE], rows, extra=(receipt[0], digest.upper()))
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("timestamp capture SHA-256 must be 64 lowercase", invalid.stderr)
        self.assertFalse(invalid.stdout)

    def test_planned_aperture_width_is_independent_of_timing_window_fit(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(4)]
        early, end = 1_000_000, 100_000_000_000
        plan = [(*key, tick, 10, 20) for key in roster for tick in (early, end)]
        rows = sum((episode(key, early=1_019_999) for key in roster[:3]), [])
        apertures = [
            (*roster[0], early, -21, 0),  # Fits early window but too long.
            (*roster[0], end, -5, 5),    # Exactly lower duration boundary.
            (*roster[1], early, 0, 20),  # Legal width; aperture exceeds early window.
            (*roster[1], end, 0, 9),     # Too short.
            (*roster[2], early, 0, ""),  # Unknown duration, regardless of clock metadata.
        ]
        plain = self.run_capture(roster, rows, apertures=apertures)
        result = self.run_capture(roster, rows, apertures=apertures, aperture_widths=plan)
        self.assertEqual((plain.returncode, result.returncode), (1, 1))
        report = json.loads(result.stdout)
        cross = report.pop("planned_read_aperture_width_support_by_us")
        for tick, violation in ((early, "reported_width_above_plan"),
                                (end, "reported_width_below_plan")):
            self.assertEqual(cross[str(tick)], {
                "frozen_attempted_by_status": {
                    violation: 1, "reported_width_within_plan": 1, "missing_or_unknown_aperture_width": 2,
                },
                "timing_eligible_by_status": {
                    violation: 1, "reported_width_within_plan": 1, "missing_or_unknown_aperture_width": 1,
                },
            })
        self.assertEqual(report, json.loads(plain.stdout))
        timing = report["reported_read_aperture_support_by_us"][str(early)]["timing_eligible_by_status"]
        self.assertEqual(timing["full_aperture_within_window"], 1)

    def test_aperture_width_recipe_requires_complete_unsigned_ordered_bounds(self):
        e, t = 1_000_000, 100_000_000_000
        plan = [(*BASE, e, 10, 20), (*BASE, t, 10, 20)]
        for planned, observed, message in (
            (plan, None, "require a read aperture table"),
            (plan[:1], [], "must cover all frozen read anchors"),
            (plan + plan[:1], [], "duplicate planned aperture width"),
            (plan[:-1] + [(*BASE, t, 20, 10)], [], "bounds must be ordered"),
            (plan[:-1] + [(*BASE, t, "-1", 10)], [], "unsigned integral microseconds"),
        ):
            r = self.run_capture([BASE], episode(), apertures=observed, aperture_widths=planned)
            self.assertEqual(r.returncode, 2)
            self.assertFalse(r.stdout)
            self.assertIn(message, r.stderr)

    def test_event_role_audit_distinguishes_timed_stress_marker_from_read(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(4)]
        t = (0, 1_000_000, 100_000_000_000)
        expected = ("state_write_end", "measurement_read", "measurement_read")
        plan = [(*key, tick, role) for key in roster for tick, role in zip(t, expected)]
        rows = sum((episode(key) for key in roster[:3]), [])
        evidence = plan[:3] + [(*roster[1], t[1], "stress_end"), (*roster[2], t[2], "unknown")]
        ordinary = self.run_capture(roster, rows)
        r = self.run_capture(roster, rows, planned_roles=plan, logged_roles=evidence)
        self.assertEqual((r.returncode, ordinary.returncode), (1, 1))
        report = json.loads(r.stdout)
        self.assertEqual(report.pop("acquisition_event_role_agreement"), {
            "frozen_attempted_by_status": {"all_logged_event_roles_match_plan": 1,
                "logged_event_role_differs_from_plan": 1, "unresolved_event_role": 2},
            "timing_eligible_by_status": {"all_logged_event_roles_match_plan": 1,
                "logged_event_role_differs_from_plan": 1, "unresolved_event_role": 1},
        })
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertNotIn("privateLot", r.stdout)

    def test_event_role_audit_validates_independent_plan_and_unique_capture(self):
        plan = [(*BASE, t, role) for t, role in zip((0, 1_000_000, 100_000_000_000),
                ("state_write_end", "measurement_read", "measurement_read"))]
        for rows, recipe, log, diagnostic in (
            (episode(), plan, None, "requires both"),
            (episode(), plan[:1], plan, "must cover"),
            (episode(), plan[:-1] + [(*BASE, plan[-1][5], "unknown")], plan, "invalid event role"),
            (episode() + episode()[:1], plan, plan, "does not identify one captured"),
            (episode(), plan, plan + plan[:1], "duplicate event role"),
        ):
            r = self.run_capture([BASE], rows, planned_roles=recipe, logged_roles=log)
            self.assertEqual(r.returncode, 2)
            self.assertFalse(r.stdout)
            self.assertIn(diagnostic, r.stderr)

    def test_planned_bias_distinguishes_wrong_polarity_and_boundary_uncertainty(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(5)]
        ticks = (0, 1_000_000, 100_000_000_000)
        plan = [(*key, tick, -105000, -95000) for key in roster for tick in ticks]
        obs = [(*key, tick, point, err) for key, (point, err) in zip(
            roster[:4], ((-100000, 5000), (100000, 0), (-90000, 5000), (100000, 0))
        ) for tick in (ticks[:1] if key == roster[3] else ticks)]
        rows = sum((episode(key) for key in roster[:3]), []) + episode(roster[3])[:1]
        plain = self.run_capture(roster, rows, read_bias=obs)
        checked = self.run_capture(roster, rows, read_bias=obs, planned_bias=plan)
        self.assertEqual((plain.returncode, checked.returncode), (1, 1))
        report = json.loads(checked.stdout)
        self.assertEqual(report.pop("planned_read_bias_agreement"), {
            "frozen_attempted_by_status": {"all_logged_bias_intervals_contained_in_plan": 1,
                "logged_bias_interval_disjoint_from_plan": 2, "unresolved_bias": 2},
            "timing_eligible_by_status": {"all_logged_bias_intervals_contained_in_plan": 1,
                "logged_bias_interval_disjoint_from_plan": 1, "unresolved_bias": 1},
        })
        self.assertEqual(report, json.loads(plain.stdout))

    def test_planned_bias_rejects_incomplete_recipe_duplicate_units_and_reversed_bounds(self):
        plan = [(*BASE, t, -10, -1) for t in (0, 1_000_000, 100_000_000_000)]
        log = [(*BASE, 0, -4, 0)]
        for proposed, observed, diagnostic in (
            (plan, None, "requires observed"),
            (plan[:-1], log, "must cover all anchors"),
            (plan + plan[:1], log, "duplicate planned bias"),
            (plan[:-1] + [(*BASE, plan[-1][5], "-1mV", -1)], log, "integral microvolts"),
            (plan[:-1] + [(*BASE, plan[-1][5], -1, -10)], log, "must be ordered"),
        ):
            r = self.run_capture([BASE], episode(), read_bias=observed, planned_bias=proposed)
            self.assertEqual(r.returncode, 2)
            self.assertFalse(r.stdout)
            self.assertIn(diagnostic, r.stderr)

    def test_native_read_reuse_crosses_episodes_but_not_namespaces(self):
        keys = [(*BASE[:4], f"E{i}") for i in range(5)]
        ticks = (0, 1_000_000, 100_000_000_000)
        rows = sum((episode(key) for key in keys[:4]), []) + episode(keys[4])[:1]
        native = [
            (*keys[0], ticks[0], "privateA", "0"),
            (*keys[0], ticks[1], "privateA", "0"),  # Within one episode.
            (*keys[1], ticks[2], "privateA", "0"),  # Also across episodes.
            *[(*keys[2], tick, "privateB", str(n)) for n, tick in enumerate(ticks)],
            (*keys[3], ticks[0], "privateC", "0"),  # Reused counter, distinct namespace.
            (*keys[4], ticks[0], "privateD", "1"),
        ]
        result = self.run_capture(keys, rows, native_reads=native)
        ordinary = self.run_capture(keys, rows)
        self.assertEqual((result.returncode, ordinary.returncode), (1, 1))
        report = json.loads(result.stdout)
        self.assertEqual(report.pop("native_read_identity_support"), {
            "frozen_attempted_by_status": {
                "native_read_reused_across_anchor_assignments": 2,
                "all_anchors_have_distinct_logged_native_reads": 1,
                "missing_or_ambiguous_native_read_assignment": 2,
            },
            "timing_eligible_by_status": {
                "native_read_reused_across_anchor_assignments": 2,
                "all_anchors_have_distinct_logged_native_reads": 1,
                "missing_or_ambiguous_native_read_assignment": 1,
            },
        })
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertNotIn("privateA", result.stdout)

    def test_native_read_metadata_does_not_resolve_duplicate_capture(self):
        native = (*BASE, 0, "run", "read")
        for rows, metadata, diagnostic in (
            (episode() + episode()[:1], [native], "does not identify one"),
            (episode(), [native, native], "duplicate native read identity"),
            (episode(), [(*BASE, 0, "", "read")], "invalid native read namespace"),
        ):
            result = self.run_capture([BASE], rows, native_reads=metadata)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(result.stdout)
            self.assertIn(diagnostic, result.stderr)

    def test_planned_paths_detect_same_wrong_route_but_accept_planned_change(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(5)]
        anchor = (0, 1_000_000, 100_000_000_000)
        route = ("privateRouteA", "privateRouteA", "privateRouteB")
        plan = [(*key, tick, path) for key in roster for tick, path in zip(anchor, route)]
        observed = [(*key, tick, path) for key, paths in zip(roster, (
            route, ("privateRouteA",) * 3, ("", "", ""), (),
            ("", "privateRouteB"),
        )) for tick, path in zip(anchor, paths)]
        rows = sum((episode(key) for key in roster[:3]), []) + episode(roster[4])[:2]
        plain = self.run_capture(roster, rows, readout_paths=observed)
        checked = self.run_capture(roster, rows, readout_paths=observed, planned_paths=plan)
        self.assertEqual((plain.returncode, checked.returncode), (1, 1))
        self.assertNotIn("privateRoute", checked.stdout)
        report = json.loads(checked.stdout)
        self.assertEqual(report.pop("planned_readout_path_agreement"), {
            "frozen_attempted_by_status": {"all_logged_routes_match_plan": 1,
                "logged_route_differs_from_plan": 2, "unresolved_route": 2},
            "timing_eligible_by_status": {"all_logged_routes_match_plan": 1,
                "logged_route_differs_from_plan": 1, "unresolved_route": 1},
        })
        self.assertEqual(report, json.loads(plain.stdout))
        self.assertEqual(report["declared_readout_path_support"]["timing_eligible_by_status"]
                         ["same_declared_path_at_all_anchors"], 1)

    def test_planned_paths_require_independent_complete_anchor_recipe(self):
        plan = [(*BASE, tick, "declared") for tick in (0, 1_000_000, 100_000_000_000)]
        observed = [(*BASE, 0, "declared")]
        for mapping, log, diagnostic in (
            (plan, None, "require observed readout paths"),
            (plan[:-1], observed, "must cover all anchors"),
            (plan + [plan[0]], observed, "duplicate planned readout path"),
            (plan + [(*BASE, 1, "declared")], observed, "does not identify a frozen anchor"),
            (plan[:-1] + [(*BASE, 100_000_000_000, "")], observed, "invalid planned readout path"),
        ):
            r = self.run_capture([BASE], episode(), readout_paths=log, planned_paths=mapping)
            self.assertEqual(r.returncode, 2)
            self.assertFalse(r.stdout)
            self.assertIn(diagnostic, r.stderr)

    def test_complete_interleaved_episodes_use_physical_die_keys(self):
        other = ("privateLot", "W2", "D1", "C1", "E1")
        rows = [row for pair in zip(episode(), episode(other)) for row in pair]
        r = self.run_capture([BASE, other], rows)
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(report["frozen_attempted_dies"], 2)
        self.assertEqual(report["eligible_episodes"], 2)
        self.assertNotIn("privateLot", r.stdout)

    def test_diagnostic_counts_die_support_on_frozen_not_captured_denominator(self):
        # Reused die names on different wafers are distinct; different cells
        # and episodes on that same physical die are not additional dies.
        same_die = (*BASE[:3], "C2", "E2")
        another = (BASE[0], "W2", *BASE[2:])
        partial_other = (*another[:4], "E2")
        entirely_absent = (BASE[0], "W3", *BASE[2:])
        extra_die = (BASE[0], "W4", *BASE[2:])
        roster = [BASE, same_die, another, partial_other, entirely_absent]
        capture = episode(BASE) + episode(same_die) + episode(another)
        capture += episode(partial_other, early=0) + episode(extra_die)
        ordinary = self.run_capture(roster, capture)
        diag = self.run_capture(roster, capture, extra=("--diagnose-dies",))
        self.assertEqual((ordinary.returncode, diag.returncode), (1, 1))
        report = json.loads(diag.stdout)
        self.assertEqual(report.pop("declared_die_timing_support"), {
            "dies_with_all_frozen_episodes_eligible": 1,
            "dies_with_some_but_not_all_frozen_episodes_eligible": 1,
            "dies_with_no_frozen_episodes_eligible": 1,
            "max_eligible_episodes_on_one_die": 2,
        })
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertEqual(report["frozen_attempted_dies"], 3)
        self.assertEqual(report["eligible_episodes"], 3)
        self.assertEqual(report["unexpected_capture_episodes"], 1)
        self.assertNotIn("privateLot", diag.stdout)

    def test_explicit_blocks_distinguish_repeated_die_from_shared_session(self):
        repeat = (*BASE[:4], "E2")
        another = (BASE[0], "W2", *BASE[2:])
        absent = (*BASE[:4], "E3")
        roster = [BASE, repeat, another, absent]
        capture = episode(BASE) + episode(repeat) + episode(another)
        block_map = [
            (*another, "privateBlockA"), (*BASE, "privateBlockA"),
            (*absent, "privateBlockC"), (*repeat, "privateBlockB"),
        ]
        ordinary = self.run_capture(roster, capture)
        diag = self.run_capture(roster, capture, block_map=block_map)
        self.assertEqual((ordinary.returncode, diag.returncode), (1, 1))
        report = json.loads(diag.stdout)
        self.assertEqual(report.pop("declared_acquisition_block_timing_support"), {
            "frozen_declared_blocks": 3, "blocks_with_eligible_episodes": 2,
            "blocks_with_eligible_episodes_from_multiple_dies": 1,
            "dies_with_eligible_episodes_in_multiple_blocks": 1,
        })
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertNotIn("privateBlock", diag.stdout)
        self.assertNotIn("privateLot", diag.stdout)

    def test_archived_block_map_digest_checks_exact_bytes_before_capture(self):
        other = (*BASE[:4], "E2")
        blocks = [(*BASE, "privateBlockA"), (*other, "privateBlockB")]
        payload = (",".join(IDS + ("acquisition_block",)) + "\r\n"
                   + "".join(",".join(row) + "\r\n" for row in blocks)).encode()
        digest = hashlib.sha256(payload).hexdigest()
        gate = ("--acquisition-block-map-sha256", digest)
        capture = episode(BASE) + episode(other)
        ordinary = self.run_capture([BASE, other], capture, block_map=blocks)
        verified = self.run_capture([BASE, other], capture, block_map=blocks, extra=gate)
        self.assertEqual((ordinary.returncode, verified.returncode), (0, 0))
        report = json.loads(verified.stdout)
        self.assertIs(report.pop("acquisition_block_map_byte_digest_verified"), True)
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertNotIn(digest, verified.stdout)

        for changed in (list(reversed(blocks)), [(*BASE, "differentBlock"), blocks[1]]):
            # The roster join succeeds if the mapping is used without a receipt;
            # with the receipt, the changed bytes fail before a malformed capture.
            self.assertEqual(self.run_capture([BASE, other], capture, block_map=changed)
                             .returncode, 0)
            invalid = self.run_capture([BASE, other], [], capture_header=("broken",),
                                       block_map=changed, extra=gate)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("acquisition block map SHA-256 mismatch", invalid.stderr)
            self.assertFalse(invalid.stdout)
            self.assertNotIn(digest, invalid.stderr)
            self.assertNotIn("privateBlock", invalid.stderr)

        missing = self.run_capture([BASE, other], capture, extra=gate)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("requires an acquisition block map", missing.stderr)
        self.assertFalse(missing.stdout)
        malformed = self.run_capture([BASE, other], capture, block_map=blocks,
                                     extra=(gate[0], digest.upper()))
        self.assertEqual(malformed.returncode, 2)
        self.assertIn("64 lowercase hexadecimal", malformed.stderr)
        self.assertFalse(malformed.stdout)

    def test_block_map_cannot_omit_duplicate_or_add_an_episode(self):
        for mapping, diagnostic in [
            ([], "does not cover"),
            ([(*BASE, "block1"), (*BASE, "block1")], "duplicate episode"),
            ([(*BASE, "block1"), (*BASE[:4], "E2", "block2")], "non-roster episode"),
            ([(*BASE, "")], "invalid acquisition block label"),
        ]:
            with self.subTest(diagnostic=diagnostic):
                r = self.run_capture([BASE], episode(), block_map=mapping)
                self.assertEqual(r.returncode, 2)
                self.assertFalse(r.stdout)
                self.assertIn(diagnostic, r.stderr)
                self.assertNotIn("privateLot", r.stderr)

    def test_stress_history_separates_unknown_absent_and_observed_by_interval(self):
        second = (*BASE[:4], "E2")
        third = (*BASE[:4], "E3")
        fourth = (*BASE[:4], "E4")
        roster = [BASE, second, third, fourth]
        rows = episode(BASE) + episode(second) + episode(third, early=0)
        documented = "no_off_schedule_bias_documented"
        biased = "off_schedule_bias_observed"
        history = [(*BASE, documented, biased), (*second, "unknown", documented),
                   (*third, biased, "unknown")]
        ordinary = self.run_capture(roster, rows)
        diagnostic = self.run_capture(roster, rows, stress=history)
        self.assertEqual((ordinary.returncode, diagnostic.returncode), (1, 1))
        report = json.loads(diagnostic.stdout)
        self.assertEqual(report.pop("declared_stress_history_support"), {
            "bias_0_to_1s": {
                "frozen_attempted_by_status": {
                    documented: 1, biased: 1, "unknown": 1, "missing_declaration": 1,
                },
                "timing_eligible_by_status": {documented: 1, "unknown": 1},
            },
            "bias_1s_to_100ks": {
                "frozen_attempted_by_status": {
                    documented: 1, biased: 1, "unknown": 1, "missing_declaration": 1,
                },
                "timing_eligible_by_status": {documented: 1, biased: 1},
            },
        })
        self.assertEqual(report.pop("declared_whole_episode_stress_history_support"), {
            "frozen_attempted_by_status": {
                "off_schedule_bias_observed_in_any_interval": 2,
                "unknown_without_observed_off_schedule_bias": 1, "missing_declaration": 1,
            },
            "timing_eligible_by_status": {
                "off_schedule_bias_observed_in_any_interval": 1,
                "unknown_without_observed_off_schedule_bias": 1,
            },
        })
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertNotIn("privateLot", diagnostic.stdout)
        absent = self.run_capture([BASE], episode(), stress=[])
        self.assertEqual(absent.returncode, 0)
        self.assertEqual(json.loads(absent.stdout)["declared_stress_history_support"]
                         ["bias_1s_to_100ks"]["timing_eligible_by_status"],
                         {"missing_declaration": 1})

    def test_interval_marginals_do_not_determine_whole_episode_support(self):
        second = (*BASE[:4], "E2")
        roster = [BASE, second]
        capture = episode(BASE) + episode(second)
        clear = "no_off_schedule_bias_documented"
        biased = "off_schedule_bias_observed"
        aligned = self.run_capture(roster, capture,
                                   stress=[(*BASE, clear, clear), (*second, biased, biased)])
        crossed = self.run_capture(roster, capture,
                                   stress=[(*BASE, clear, biased), (*second, biased, clear)])
        self.assertEqual((aligned.returncode, crossed.returncode), (0, 0))
        a, b = map(lambda result: json.loads(result.stdout), (aligned, crossed))
        joint_key = "declared_whole_episode_stress_history_support"
        joint_a, joint_b = a.pop(joint_key), b.pop(joint_key)
        self.assertEqual(a, b)
        for field in ("frozen_attempted_by_status", "timing_eligible_by_status"):
            self.assertEqual(joint_a[field], {
                "no_off_schedule_bias_documented_in_both_intervals": 1,
                "off_schedule_bias_observed_in_any_interval": 1,
            })
            self.assertEqual(joint_b[field], {"off_schedule_bias_observed_in_any_interval": 2})

    def test_preserved_episode_marginals_can_hide_die_stress_concentration(self):
        a2 = (*BASE[:4], "E2")
        b1 = (BASE[0], "W2", *BASE[2:])
        b2 = (*b1[:4], "E2")
        absent = (BASE[0], "W3", *BASE[2:])
        roster = [BASE, a2, b1, b2, absent]
        capture = [r for key in roster[:4] for r in episode(key)]
        clear = "no_off_schedule_bias_documented"
        biased = "off_schedule_bias_observed"
        no_bias_status = "no_off_schedule_bias_documented_in_both_intervals"
        extra_bias_status = "off_schedule_bias_observed_in_any_interval"
        flags = ("--diagnose-dies", "--diagnose-die-stress")

        def report(states, options):
            result = self.run_capture(
                roster, capture, extra=options,
                stress=[(*key, state, state) for key, state in zip(roster[:4], states)],
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertNotIn("privateLot", result.stdout)
            return json.loads(result.stdout)

        split = (clear, biased, clear, biased)
        concentrated = (clear, clear, biased, biased)
        diag_key = "declared_die_stress_history_support"
        left, right = report(concentrated, flags), report(split, flags)
        left_diag, right_diag = left.pop(diag_key), right.pop(diag_key)
        self.assertEqual(left, right)
        self.assertEqual(right, report(split, ("--diagnose-dies",)))
        self.assertEqual(left_diag, {
            "frozen_die_denominator": 3, "timing_eligible_die_denominator": 2,
            "frozen_dies_with_status": {
                "missing_declaration": 1, no_bias_status: 1, extra_bias_status: 1,
            },
            "timing_eligible_dies_with_status": {no_bias_status: 1, extra_bias_status: 1},
        })
        self.assertEqual(right_diag, {
            "frozen_die_denominator": 3, "timing_eligible_die_denominator": 2,
            "frozen_dies_with_status": {
                "missing_declaration": 1, no_bias_status: 2, extra_bias_status: 2,
            },
            "timing_eligible_dies_with_status": {no_bias_status: 2, extra_bias_status: 2},
        })

        missing = self.run_capture([BASE], episode(), extra=("--diagnose-die-stress",))
        self.assertEqual(missing.returncode, 2)
        self.assertIn("requires a stress history", missing.stderr)
        self.assertFalse(missing.stdout)

    def test_stress_history_checks_membership_duplicates_and_explicit_status(self):
        valid = (*BASE, "unknown", "unknown")
        for rows, diagnostic in (
            ([valid, valid], "duplicate episode"),
            ([(*BASE[:4], "not_enrolled", "unknown", "unknown")], "non-roster"),
            ([(*BASE, "", "unknown")], "invalid stress history status"),
        ):
            with self.subTest(diagnostic=diagnostic):
                result = self.run_capture([BASE], episode(), stress=rows)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(result.stdout)
                self.assertIn(diagnostic, result.stderr)
                self.assertNotIn("privateLot", result.stderr)

    def test_cannot_borrow_late_or_other_episode_measurement(self):
        rows = episode()[:2]
        rows.append([*BASE, 95_000_000_000, "counter", "epoch", 100_001_000_047, 0])
        other = (*BASE[:4], "E2")
        rows.append(episode(other)[-1])
        absent = (*BASE[:4], "E3")
        r = self.run_capture([BASE, absent], rows)
        self.assertEqual(r.returncode, 1, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(report["unexpected_capture_episodes"], 1)
        self.assertEqual(report["nonscheduled_rows_in_frozen_episodes"], 1)
        self.assertEqual(report["frozen_episodes_by_primary_reason"], {
            "missing_capture": 1, "missing_nominal_anchor": 1,
        })

    def test_clock_errors_duplicates_and_numeric_boundaries_remain_distinct(self):
        cases = [
            ("reset", episode(), "different_clock_or_epoch"),
            ("missing_clock", episode(), "unknown_clock_or_time"),
            ("duplicate", episode(), "duplicate_nominal_anchor"),
            ("early_zero", episode(early=0), "early_outside_window"),
            ("early_uncertain", episode(early=1_020_000, error=1),
             "early_boundary_uncertain"),
            ("late_uncertain", episode(final=80_000_000_000, error=1),
             "terminal_boundary_uncertain"),
            ("late_outside", episode(final=79_999_999_999), "terminal_outside_window"),
            ("closed_lower", episode(final=80_000_000_000), "eligible"),
            ("closed_upper", episode(early=1_020_000, final=120_000_000_000), "eligible"),
        ]
        cases[0][1][2][7] = "epoch-reset"
        cases[1][1][0][6] = ""
        cases[2][1].append(cases[2][1][-1])
        for name, rows, reason in cases:
            with self.subTest(name=name):
                r = self.run_capture([BASE], rows)
                self.assertEqual(r.returncode, int(reason != "eligible"), r.stderr)
                self.assertEqual(json.loads(r.stdout)["frozen_episodes_by_primary_reason"],
                                 {reason: 1})

    def test_anchor_diagnosis_exposes_secondary_faults_on_the_same_denominator(self):
        off_early = episode(early=0, final=80_000_000_000, error=1)
        off_early[2][7] = "other-epoch"
        other = (*BASE[:4], "E2")
        duplicate_origin = episode(other)
        duplicate_origin.append(duplicate_origin[0])
        duplicate_origin[2][8] = ""
        absent = (*BASE[:4], "E3")
        ok = (*BASE[:4], "E4")
        roster = [BASE, other, absent, ok]
        capture = off_early + duplicate_origin + episode(ok)
        before = self.run_capture(roster, capture)
        richer = self.run_capture(roster, capture, extra=("--diagnose-anchors",))
        self.assertEqual((before.returncode, richer.returncode), (1, 1))
        result = json.loads(richer.stdout)
        counts = result.pop("frozen_nominal_anchor_statuses_by_us")
        self.assertEqual(result, json.loads(before.stdout))
        self.assertEqual(counts, {
            "0": {
                "duplicate_nominal_anchor": 1, "missing_nominal_anchor": 1,
                "usable_timestamp": 2,
            },
            "1000000": {
                "boundary_uncertain": 1, "missing_nominal_anchor": 1,
                "unusable_origin": 1, "within_window": 1,
            },
            "100000000000": {
                "different_clock_or_epoch": 1, "missing_nominal_anchor": 1,
                "unknown_clock_or_time": 1, "within_window": 1,
            },
        })
        self.assertTrue(all(sum(statuses.values()) == len(roster) for statuses in counts.values()))
        self.assertNotIn("other-epoch", richer.stdout)
        self.assertNotIn("privateLot", richer.stdout)

    def test_single_fault_does_not_disable_unrelated_anchor_diagnosis(self):
        rows = episode(early=0)
        richer = self.run_capture([BASE], rows, extra=("--diagnose-anchors",))
        self.assertEqual(richer.returncode, 1, richer.stderr)
        data = json.loads(richer.stdout)
        self.assertEqual(data["frozen_episodes_by_primary_reason"], {"early_outside_window": 1})
        self.assertEqual(data["frozen_nominal_anchor_statuses_by_us"]["100000000000"],
                         {"within_window": 1})

    def test_duplicate_anchor_diagnosis_does_not_repair_or_guess_readings(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(4)]
        ticks = [episode(key) for key in roster]
        copy_origin = ticks[0][0].copy()
        copy_origin[8] = "00047"  # Same exact tick even though the CSV spelling differs.
        ticks[0].append(copy_origin)
        unknown_early = ticks[1][1].copy()
        unknown_early[6] = ""
        ticks[1].append(unknown_early)
        conflict_early = ticks[1][1].copy()
        conflict_early[9] = 1  # A different uncertainty bound is conflicting metadata.
        ticks[1].append(conflict_early)
        unknown_final = ticks[2][-1].copy()
        unknown_final[7] = ""
        ticks[2].append(unknown_final)
        ticks[3][-1][7] = ""
        second_unknown_final = ticks[3][-1].copy()
        second_unknown_final[7] = "epoch"
        second_unknown_final[8] = ""
        ticks[3].append(second_unknown_final)
        rows = [row for group in ticks for row in group]
        ordinary = self.run_capture(roster, rows)
        richer = self.run_capture(roster, rows, extra=("--diagnose-duplicate-anchors",))
        self.assertEqual((ordinary.returncode, richer.returncode), (1, 1))
        result = json.loads(richer.stdout)
        self.assertEqual(result.pop("duplicate_frozen_anchor_metadata_by_us"), {
            "0": {"identical_usable_stamps": 1},
            "1000000": {"conflicting_usable_stamps": 1},
            "100000000000": {"usable_and_unresolved_stamps": 1, "all_stamps_unresolved": 1},
        })
        self.assertEqual(result, json.loads(ordinary.stdout))
        self.assertEqual(result["eligible_episodes"], 0)
        self.assertEqual(result["frozen_episodes_by_primary_reason"], {"duplicate_nominal_anchor": 4})
        self.assertNotIn("privateLot", richer.stdout)

    def test_alternative_times_diagnose_absent_terminal_without_substitution(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(7)]
        rows = [episode(key) for key in roster]

        def off_slot(row, delta, *, epoch="epoch", error=0):
            r = row.copy()
            r[5], r[7], r[8], r[9] = 95_000_000_000, epoch, delta + 47, error
            return r

        inside = off_slot(rows[0].pop(), 100_001_000_000)
        rows[0].append(inside)
        rows[0].append(off_slot(inside, 80_000_000_000, error=1))
        rows[1].append(off_slot(rows[1].pop(), 80_000_000_000, error=1))
        wrong_epoch = off_slot(rows[2].pop(), 100_001_000_000, epoch="unknown_epoch")
        rows[2].append(wrong_epoch)
        rows[2].append(off_slot(wrong_epoch, 70_000_000_000))
        rows[3].append(off_slot(rows[3].pop(), 70_000_000_000))
        rows[4].append(off_slot(rows[4].pop(), 100_001_000_000))
        rows[4].pop(0)  # Missing origin, even though the raw timer seems plausible.
        rows[5].clear()  # Completely missing frozen capture.
        rows[6].append(off_slot(rows[6][-1], 100_001_000_001))
        capture = [item for group in rows for item in group]
        baseline = self.run_capture(roster, capture)
        checked = self.run_capture(roster, capture, extra=("--diagnose-terminal-alternatives",))
        self.assertEqual((baseline.returncode, checked.returncode), (1, 1))
        report = json.loads(checked.stdout)
        self.assertEqual(report.pop("nonscheduled_read_timing_when_nominal_terminal_absent"), {
            "frozen_episodes_without_nominal_terminal": 6,
            "episodes_by_status": {
                "all_outside_window": 1, "at_least_one_in_window": 1,
                "boundary_uncertain_without_in_window": 1, "no_nonscheduled_rows": 1,
                "unresolved_without_proven_in_window": 1, "unusable_origin": 1,
            },
        })
        self.assertEqual(report, json.loads(baseline.stdout))
        self.assertEqual(report["eligible_episodes"], 1)
        self.assertNotIn("unknown_epoch", checked.stdout)
        self.assertNotIn("privateLot", checked.stdout)

    def test_observed_readout_path_is_not_inferred_from_common_clock(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(4)]
        capture = [row for key in roster[:3] for row in episode(key)]
        capture.extend(episode(roster[3])[:2])
        ns = (0, 1_000_000, 100_000_000_000)
        paths = [(*roster[0], nominal, "privatePathA") for nominal in ns]
        paths += [(*roster[1], ns[0], "privatePathA"), (*roster[1], ns[2], "privatePathB")]
        paths += [(*roster[2], ns[1], ""), (*roster[2], ns[2], "privatePathA")]
        paths += [(*roster[3], ns[0], "privatePathA")]
        ordinary = self.run_capture(roster, capture)
        checked = self.run_capture(roster, capture, readout_paths=paths)
        self.assertEqual((ordinary.returncode, checked.returncode), (1, 1))
        report = json.loads(checked.stdout)
        timing = {"same_declared_path_at_all_anchors": 1, "observed_path_change": 1,
                  "missing_or_unknown_path_without_observed_change": 1}
        self.assertEqual(report.pop("declared_readout_path_support"), {
            "frozen_attempted_by_status": {**timing, "missing_or_ambiguous_capture": 1},
            "timing_eligible_by_status": timing,
        })
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertNotIn("privatePath", checked.stdout)

    def test_readout_path_map_cannot_bind_missing_repeated_or_off_anchor_records(self):
        rows = episode()
        path = (*BASE, 100_000_000_000, "privatePath")
        for capture, mapping, diagnostic in (
            (rows[:2], [path], "does not identify one captured frozen anchor"),
            (rows + [rows[-1]], [path], "does not identify one captured frozen anchor"),
            (rows, [path, path], "duplicate readout path"),
            (rows, [(*BASE, 99_999_000_000, "privatePath")], "off-anchor slot"),
        ):
            with self.subTest(diagnostic=diagnostic):
                denied = self.run_capture([BASE], capture, readout_paths=mapping)
                self.assertEqual(denied.returncode, 2)
                self.assertFalse(denied.stdout)
                self.assertIn(diagnostic, denied.stderr)
                self.assertNotIn("privatePath", denied.stderr)

    def test_read_bias_keeps_clock_and_route_separate_from_observed_bias(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(5)]
        capture = [row for key in roster[:4] for row in episode(key)]
        capture.extend(episode(roster[4])[:2])
        ns = (0, 1_000_000, 100_000_000_000)
        paths = [(*key, n, "same_private_path") for key in roster[:4] for n in ns]
        bias = [(*roster[0], n, v, 5) for n, v in zip(ns, (-10, 0, 10))]
        bias += [(*roster[1], ns[0], -10, 0), (*roster[1], ns[2], 0, 0)]
        bias += [(*roster[2], n, v, 5) for n, v in zip(ns, (-10, 0, -5))]
        bias += [(*roster[3], ns[0], -7, ""), (*roster[3], ns[2], -7, 1)]
        ordinary = self.run_capture(roster, capture, readout_paths=paths)
        checked = self.run_capture(roster, capture, readout_paths=paths, read_bias=bias)
        self.assertEqual((ordinary.returncode, checked.returncode), (1, 1))
        report = json.loads(checked.stdout)
        timing = {"common_reported_bias_ruled_out": 2, "all_known_common_bias_not_ruled_out": 1,
                  "missing_or_unknown_bias_without_known_conflict": 1}
        self.assertEqual(report.pop("reported_read_bias_support"), {
            "frozen_attempted_by_status": {**timing, "missing_or_ambiguous_capture": 1},
            "timing_eligible_by_status": timing,
        })
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertNotIn("same_private_path", checked.stdout)

    def test_read_bias_rejects_unsupported_bindings_and_malformed_signed_data(self):
        rows = episode()
        terminal = (*BASE, 100_000_000_000, "-731", "1")
        for capture, mapping, diagnostic in (
            (rows[:2], [terminal], "does not identify one captured frozen anchor"),
            (rows, [terminal, terminal], "duplicate read bias"),
            (rows, [(*terminal[:-1], "-1")], "error cannot be negative"),
            (rows, [(*terminal[:-2], "-3.5", "1")], "integral microvolts"),
        ):
            with self.subTest(diagnostic=diagnostic):
                denied = self.run_capture([BASE], capture, read_bias=mapping)
                self.assertEqual(denied.returncode, 2)
                self.assertFalse(denied.stdout)
                self.assertIn(diagnostic, denied.stderr)
                self.assertNotIn("-731", denied.stderr)

    def test_read_timestamp_eligibility_does_not_imply_full_aperture_support(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(5)]
        capture = episode(roster[0], early=1, final=80_000_000_000)
        capture += episode(roster[1], early=12, final=80_000_000_012, error=2)
        capture += episode(roster[2], early=5, error=5)
        capture += episode(roster[3]) + episode(roster[4])[:2]
        first, last = 1_000_000, 100_000_000_000
        apertures = [
            (*roster[0], first, -1, 0), (*roster[0], last, 0, 1),
            (*roster[1], first, -11, 0), (*roster[1], last, -12, 1),
            # The aperture is wider than the entire early window; translating
            # the nominal read within its error support cannot make it fit.
            (*roster[2], first, -5, 1_020_000), (*roster[2], last, 0, 1),
            (*roster[3], first, "", ""), (*roster[4], first, -1, 1),
        ]
        baseline = self.run_capture(roster, capture)
        checked = self.run_capture(roster, capture, apertures=apertures)
        self.assertEqual((baseline.returncode, checked.returncode), (1, 1))
        report = json.loads(checked.stdout)
        self.assertEqual(report.pop("reported_read_aperture_support_by_us"), {
            str(first): {
                "frozen_attempted_by_status": {
                    "aperture_fit_depends_on_timestamp_error": 1, "full_aperture_cannot_fit": 2,
                    "full_aperture_within_window": 1, "missing_or_unknown_aperture": 1,
                },
                "timing_eligible_by_status": {
                    "aperture_fit_depends_on_timestamp_error": 1, "full_aperture_cannot_fit": 1,
                    "missing_or_unknown_aperture": 1,
                },
            },
            str(last): {
                "frozen_attempted_by_status": {
                    "aperture_fit_depends_on_timestamp_error": 1, "full_aperture_within_window": 2,
                    "missing_or_ambiguous_capture": 1, "missing_or_unknown_aperture": 1,
                },
                "timing_eligible_by_status": {
                    "aperture_fit_depends_on_timestamp_error": 1, "full_aperture_within_window": 1,
                    "missing_or_unknown_aperture": 1,
                },
            },
        })
        self.assertEqual(report, json.loads(baseline.stdout))
        self.assertNotIn("privateLot", checked.stdout)

    def test_read_aperture_rejects_reversed_offsets_and_unlinked_rows(self):
        rows = episode()
        for aperture, diagnostic in (
            ((*BASE, 0, 0, 0), "only the early and terminal"),
            ((*BASE, 1_000_000, 1, -1), "opens after it closes"),
            ((*BASE, 1_000_000, "1.5", 2), "integral microseconds"),
        ):
            result = self.run_capture([BASE], rows, apertures=[aperture])
            self.assertEqual(result.returncode, 2)
            self.assertFalse(result.stdout)
            self.assertIn(diagnostic, result.stderr)

    def test_joint_aperture_distinguishes_conflicting_shared_origin_from_marginal_fit(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(4)]
        first, last = 1_000_000, 100_000_000_000
        rows = [episode(key, early=1_019_998, final=80_000_000_002) for key in roster]
        for triplet in rows:
            triplet[0][9] = 2  # Shared origin bound; both per-read residuals are fixed.
        capture = [row for triple in rows for row in triple]
        apertures = [
            (*roster[0], first, 1, 3), (*roster[0], last, -3, -1),
            (*roster[1], first, 1, 3), (*roster[1], last, 1, 3),
            (*roster[2], first, -2, 0), (*roster[2], last, 0, 2),
            (*roster[3], first, -2, 0), (*roster[3], last, "", ""),
        ]
        gate = ("--diagnose-joint-aperture",)
        baseline = self.run_capture(roster, capture, apertures=apertures)
        checked = self.run_capture(roster, capture, apertures=apertures, extra=gate)
        self.assertEqual((baseline.returncode, checked.returncode), (0, 0))
        report = json.loads(checked.stdout)
        aggregate = {
            "no_joint_fit_under_read_residual_box": 1,
            "joint_fit_possible_under_read_residual_box": 1,
            "joint_fit_for_all_errors_in_read_residual_box": 1,
            "missing_or_unknown_aperture": 1,
        }
        self.assertEqual(report.pop("conditional_shared_origin_aperture_support"), {
            "frozen_attempted_by_status": aggregate, "timing_eligible_by_status": aggregate,
        })
        self.assertEqual(report, json.loads(baseline.stdout))
        self.assertEqual(report["eligible_episodes"], 4)
        self.assertEqual(report["reported_read_aperture_support_by_us"][str(first)]
                         ["timing_eligible_by_status"]["aperture_fit_depends_on_timestamp_error"], 2)
        self.assertEqual(report["reported_read_aperture_support_by_us"][str(last)]
                         ["timing_eligible_by_status"]["aperture_fit_depends_on_timestamp_error"], 1)

        missing = self.run_capture(roster, capture, extra=gate)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("requires a read aperture table", missing.stderr)
        self.assertFalse(missing.stdout)

    def test_common_read_error_is_distinct_from_independently_bounded_read_errors(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(2)]
        early, terminal = 1_000_000, 100_000_000_000
        rows = [episode(key, early=1_019_998, final=80_000_000_002) for key in roster]
        for stamps in rows:
            stamps[1][9] = stamps[2][9] = 2
        rows[1][1][9] = 1
        apertures = [
            (*roster[0], early, 1, 3), (*roster[0], terminal, -3, -1),
            (*roster[1], early, -1, 0), (*roster[1], terminal, -1, 0),
        ]
        capture = [row for triple in rows for row in triple]
        flags = ("--diagnose-joint-aperture",)
        common = "--diagnose-common-read-error-aperture"
        baseline = self.run_capture(roster, capture, apertures=apertures, extra=flags)
        checked = self.run_capture(roster, capture, apertures=apertures, extra=(*flags, common))
        self.assertEqual((baseline.returncode, checked.returncode), (0, 0))
        report = json.loads(checked.stdout)
        statuses = {"no_joint_fit_under_common_read_error": 1,
                    "joint_fit_for_all_errors_under_common_read_error": 1}
        self.assertEqual(report.pop("conditional_common_read_error_aperture_support"), {
            "frozen_attempted_by_status": statuses, "timing_eligible_by_status": statuses,
        })
        self.assertEqual(report, json.loads(baseline.stdout))
        self.assertEqual(report["conditional_shared_origin_aperture_support"]
                         ["timing_eligible_by_status"],
                         {"joint_fit_possible_under_read_residual_box": 2})

        result = self.run_capture(roster, capture, extra=(common,))
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires a read aperture table", result.stderr)

    def test_trusted_commanded_plan_prevents_post_capture_schedule_rewrite(self):
        rows = episode()[:2]
        declared = [(*BASE, tick) for tick in (0, 1_000_000, 100_000_000_000)]
        original = (",".join(IDS + ("nominal_us",)) + "\r\n"
                    + "".join(",".join(map(str, row)) + "\r\n" for row in declared)).encode()
        digest = hashlib.sha256(original).hexdigest()
        flag = "--commanded-schedule-sha256"
        plain = self.run_capture([BASE], rows, schedule=declared)
        verified = self.run_capture([BASE], rows, schedule=declared, extra=(flag, digest))
        self.assertEqual((plain.returncode, verified.returncode), (1, 1))
        report = json.loads(verified.stdout)
        self.assertIs(report.pop("commanded_schedule_byte_digest_verified"), True)
        self.assertEqual(report, json.loads(plain.stdout))
        self.assertNotIn(digest, verified.stdout)
        rewritten = self.run_capture([BASE], rows, schedule=declared[:2])
        self.assertEqual(json.loads(rewritten.stdout)["commanded_nominal_slot_coverage"]
                         ["episodes_with_exact_slot_coverage"], 1)
        for changed in (declared[:2], list(reversed(declared))):
            with self.subTest(changed=changed):
                # Fail before attempting to parse even a malformed capture.
                rejected = self.run_capture([BASE], [], capture_header=("broken",),
                                            schedule=changed, extra=(flag, digest))
                self.assertEqual(rejected.returncode, 2)
                self.assertEqual(rejected.stdout, "")
                self.assertIn("commanded schedule SHA-256 mismatch", rejected.stderr)
                self.assertNotIn(digest, rejected.stderr)
        for schedule, supplied, message in (
            (None, digest, "requires a commanded schedule"),
            (declared, digest.upper(), "64 lowercase hexadecimal"),
        ):
            invalid = self.run_capture([BASE], rows, schedule=schedule, extra=(flag, supplied))
            self.assertEqual(invalid.returncode, 2)
            self.assertEqual(invalid.stdout, "")
            self.assertIn(message, invalid.stderr)

    def test_equal_counts_do_not_prove_commanded_slot_completeness(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(4)]
        ref = (0, 1_000_000, 100_000_000_000)
        off_anchor = 90_000_000_000
        schedule = [(*key, nominal) for key in roster for nominal in ref]
        schedule.append((*roster[2], off_anchor))
        complete, wrong_anchor, repeated_off_anchor = (episode(key) for key in roster[:3])
        wrong_anchor[-1][5] = ref[1]  # Duplicated early, missing terminal; 3 rows still.
        dense = repeated_off_anchor[-1].copy()
        dense[5] = off_anchor
        repeated_off_anchor += [dense, dense]
        extra = complete[0].copy()
        extra[5] = 7_000
        complete.append(extra)
        capture = complete + wrong_anchor + repeated_off_anchor
        ordinary = self.run_capture(roster, capture)
        checked = self.run_capture(roster, capture, schedule=schedule)
        self.assertEqual((ordinary.returncode, checked.returncode), (1, 1))
        report = json.loads(checked.stdout)
        self.assertEqual(report.pop("commanded_nominal_slot_coverage"), {
            "commanded_slots": 13, "observed_rows_in_frozen_episodes": 12,
            "missing_commanded_slots": 4, "commanded_slots_observed_once": 7,
            "commanded_slots_observed_more_than_once": 2, "extra_rows_at_commanded_slots": 2,
            "uncommanded_slots_observed": 1, "uncommanded_rows": 1,
            "episodes_with_exact_slot_coverage": 0,
            "episodes_with_equal_row_count_but_slot_mismatch": 1,
        })
        self.assertEqual(report, json.loads(ordinary.stdout))
        self.assertNotIn("privateLot", checked.stdout)

    def test_commanded_schedule_requires_exact_frozen_episodes_and_unique_nominal(self):
        rows = episode()
        for schedule, diagnostic in (
            ([], "does not cover"),
            ([(*BASE, 7), (*BASE, "0007")], "duplicate nominal slot"),
            ([(*BASE[:4], "not_frozen", 0)], "non-roster episode"),
        ):
            result = self.run_capture([BASE], rows, schedule=schedule)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(result.stdout)
            self.assertIn(diagnostic, result.stderr)
        exact = self.run_capture([BASE], rows,
                                 schedule=[(*BASE, nominal) for nominal in (0, 1_000_000, 100_000_000_000)])
        self.assertEqual(exact.returncode, 0, exact.stderr)
        self.assertEqual(json.loads(exact.stdout)["commanded_nominal_slot_coverage"]
                         ["episodes_with_exact_slot_coverage"], 1)

    def test_termination_disposition_does_not_substitute_for_observed_endpoint(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(6)]
        rows = episode(roster[0])
        rows += episode(roster[1])[:2] + episode(roster[2])[:2]
        rows += episode(roster[3], early=0) + episode(roster[4], final=79_999_999_999)
        dispositions = [(*key, state) for key, state in zip(
            roster, ("completed", "completed", "value_threshold_stop", "operator_abort", "unknown")
        )]
        plain = self.run_capture(roster, rows)
        checked = self.run_capture(roster, rows, terminations=dispositions)
        self.assertEqual((plain.returncode, checked.returncode), (1, 1))
        report = json.loads(checked.stdout)
        fields = ("frozen_attempted_episodes", "episodes_with_capture",
                  "episodes_with_nominal_terminal_row", "episodes_within_terminal_time_window",
                  "timing_eligible_episodes")
        expected = {
            "completed": (2, 2, 1, 1, 1), "value_threshold_stop": (1, 1, 0, 0, 0),
            "operator_abort": (1, 1, 1, 1, 0), "unknown": (1, 1, 1, 0, 0),
            "missing_declaration": (1, 0, 0, 0, 0),
        }
        self.assertEqual(report.pop("declared_termination_disposition_support"), {
            reason: dict(zip(fields, counts)) for reason, counts in expected.items()
        })
        self.assertEqual(report, json.loads(plain.stdout))
        self.assertNotIn("privateLot", checked.stdout)

    def test_joint_termination_and_schedule_audit_recovers_both_marginals(self):
        roster = [(*BASE[:4], f"E{i}") for i in range(5)]
        ref = (0, 1_000_000, 100_000_000_000)
        schedule = [(*key, nominal) for key in roster for nominal in ref]
        schedule.append((*roster[3], 90_000_000_000))
        bad, threshold, unknown, no_declaration = (episode(key) for key in
                                                 (roster[0], roster[1], roster[2], roster[3]))
        bad[-1][5] = ref[1]  # Same row total, distinct commanded terminal slot lost.
        rows = bad + threshold[:2] + unknown + no_declaration
        states = ("completed", "value_threshold_stop", "unknown")
        dispositions = [(*key, state) for key, state in zip(roster, states)]
        schedule_only = self.run_capture(roster, rows, schedule=schedule)
        result = self.run_capture(roster, rows, schedule=schedule, terminations=dispositions)
        self.assertEqual((schedule_only.returncode, result.returncode), (1, 1))
        report = json.loads(result.stdout)
        joint = report.pop("commanded_nominal_slot_coverage_by_declared_termination")
        disposition_support = report.pop("declared_termination_disposition_support")
        self.assertEqual(report, json.loads(schedule_only.stdout))
        self.assertEqual(set(joint), set(states) | {"missing_declaration"})
        for field, total in report["commanded_nominal_slot_coverage"].items():
            self.assertEqual(sum(group[field] for group in joint.values()), total)
        for disposition, group in joint.items():
            self.assertEqual(group["frozen_attempted_episodes"],
                             disposition_support[disposition]["frozen_attempted_episodes"])
        self.assertEqual(joint["completed"]["episodes_with_equal_row_count_but_slot_mismatch"], 1)
        self.assertEqual(joint["value_threshold_stop"]["missing_commanded_slots"], 1)
        self.assertEqual(joint["unknown"]["episodes_with_exact_slot_coverage"], 1)
        self.assertEqual(joint["missing_declaration"]["missing_commanded_slots"], 4)
        self.assertEqual(joint["missing_declaration"]["episodes_with_exact_slot_coverage"], 0)
        self.assertNotIn("privateLot", result.stdout)

    def test_termination_receipt_rejects_stratum_relabel_with_unchanged_schedule(self):
        other = (*BASE[:4], "E2")
        roster = [BASE, other]
        rows = episode(BASE) + episode(other)[:2]
        schedule = [(*key, tick) for key in roster for tick in (0, 1_000_000, 100_000_000_000)]
        logged = [(*BASE, "completed"), (*other, "value_threshold_stop")]
        swapped = [(*BASE, "value_threshold_stop"), (*other, "completed")]

        def receipt(fields, entries):
            text = ",".join(fields) + "\r\n" + "".join(
                ",".join(map(str, row)) + "\r\n" for row in entries
            )
            return hashlib.sha256(text.encode()).hexdigest()

        plan_flag = ("--commanded-schedule-sha256", receipt(IDS + ("nominal_us",), schedule))
        status_flag = "--termination-dispositions-sha256"
        digest = receipt(IDS + ("termination_disposition",), logged)
        original = self.run_capture(roster, rows, schedule=schedule, terminations=logged, extra=plan_flag)
        unpinned = self.run_capture(roster, rows, schedule=schedule, terminations=swapped, extra=plan_flag)
        accepted = self.run_capture(roster, rows, schedule=schedule, terminations=logged,
                                    extra=(*plan_flag, status_flag, digest))
        self.assertEqual((original.returncode, unpinned.returncode, accepted.returncode), (1, 1, 1))
        report, changed, verified = map(lambda r: json.loads(r.stdout), (original, unpinned, accepted))
        self.assertIs(verified.pop("termination_dispositions_byte_digest_verified"), True)
        self.assertEqual(verified, report)
        group = "commanded_nominal_slot_coverage_by_declared_termination"
        self.assertEqual(report[group]["completed"]["missing_commanded_slots"], 0)
        self.assertEqual(changed[group]["completed"]["missing_commanded_slots"], 1)
        self.assertIs(changed["commanded_schedule_byte_digest_verified"], True)
        self.assertEqual(report["eligible_episodes"], changed["eligible_episodes"])
        for payload, pin, message in (
            (swapped, digest, "termination dispositions SHA-256 mismatch"),
            (list(reversed(logged)), digest, "termination dispositions SHA-256 mismatch"),
            (None, digest, "requires a termination dispositions table"),
            (logged, digest.upper(), "64 lowercase hexadecimal"),
        ):
            bad = self.run_capture(roster, [], capture_header=("broken",), schedule=schedule,
                                   terminations=payload, extra=(*plan_flag, status_flag, pin))
            self.assertEqual(bad.returncode, 2)
            self.assertFalse(bad.stdout)
            self.assertIn(message, bad.stderr)
            self.assertNotIn(digest, bad.stderr)

    def test_termination_declarations_validate_keys_and_vocabulary(self):
        valid = (*BASE, "instrument_abort")
        for mapping, diagnostic in (
            ([valid, valid], "duplicate episode"),
            ([(*BASE[:4], "not_frozen", "unknown")], "non-roster episode"),
            ([(*BASE, "invented_reason")], "invalid termination disposition"),
        ):
            result = self.run_capture([BASE], episode(), terminations=mapping)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(result.stdout)
            self.assertIn(diagnostic, result.stderr)
            self.assertNotIn("invented_reason", result.stderr)

    def test_capture_cannot_supply_labels_or_change_frozen_identity_set(self):
        r = self.run_capture([BASE], [(*row, "sensitive-value") for row in episode()],
                             IDS + TIMES + ("conductance_S",))
        self.assertEqual(r.returncode, 2)
        self.assertFalse(r.stdout)
        self.assertNotIn("sensitive-value", r.stderr)
        r = self.run_capture([BASE, BASE], episode())
        self.assertEqual(r.returncode, 2)
        self.assertIn("duplicate", r.stderr)


if __name__ == "__main__":
    unittest.main()
