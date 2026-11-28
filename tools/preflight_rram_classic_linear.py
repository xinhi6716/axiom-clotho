"""Plan uniform starts for one confirmed Classic I/V-t linear standard-grid setup."""
import argparse
import json

TOTAL_SAMPLE_CAP = 100_001
INTERVAL_MIN_US, INTERVAL_MAX_US, INTERVAL_STEP_US = 2_000, 65_535_000, 1_000
EARLY_END_US = 1_020_000
TERMINAL_START_US, TERMINAL_END_US = 80_000_000_000, 120_000_000_000


def ceiling_divide(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def planned_start_count(lower_us, upper_us, interval_us, samples):
    low = max(1, ceiling_divide(lower_us, interval_us))
    high = min(samples - 1, upper_us // interval_us)
    return max(0, high - low + 1)


def joint_anchor_design(cap):
    """Find minimum-sample uniform plans; nominal equality is stricter than window coverage."""
    window_plans, exact_plans = [], []
    for interval in range(INTERVAL_MIN_US, min(EARLY_END_US, INTERVAL_MAX_US) + 1, INTERVAL_STEP_US):
        first_terminal = ceiling_divide(TERMINAL_START_US, interval) * interval
        required = first_terminal // interval + 1
        if first_terminal <= TERMINAL_END_US and required <= cap:
            window_plans.append((required, first_terminal, interval))
        target = 100_000_000_000
        required = target // interval + 1
        if 1_000_000 % interval == 0 and target % interval == 0 and required <= cap:
            exact_plans.append((required, target, interval))

    def witness(plans):
        if not plans:
            return None
        count, terminal, interval = min(plans)
        return {"sample_interval_us": interval, "samples_per_channel": count,
                "witness_terminal_start_us": terminal}

    return {"minimum_samples_window_witness": witness(window_plans),
            "minimum_samples_exact_nominal_witness": witness(exact_plans)}


def preflight(channels, interval_us, samples, *, diagnose_design=False):
    if type(channels) is not int or not 1 <= channels <= TOTAL_SAMPLE_CAP:
        raise ValueError("measurement-channel count does not admit any samples")
    if (type(interval_us) is not int or not INTERVAL_MIN_US <= interval_us <= INTERVAL_MAX_US
            or interval_us % INTERVAL_STEP_US):
        raise ValueError("interval must be on the documented 2–65535 ms standard grid")
    cap = TOTAL_SAMPLE_CAP // channels
    if type(samples) is not int or not 1 <= samples <= cap:
        raise ValueError("samples per channel must be between 1 and the configured channel cap")

    def exact_slot(elapsed_us):
        index, remainder = divmod(elapsed_us, interval_us)
        return remainder == 0 and index < samples

    min_interval = (ceiling_divide(TERMINAL_START_US, (cap - 1) * INTERVAL_STEP_US)
                    * INTERVAL_STEP_US if cap > 1 else None)
    if min_interval is not None and min_interval > INTERVAL_MAX_US:
        min_interval = None
    report = {
        "schema": "clotho.rram_classic_linear_plan.v1",
        "mode": "confirmed_classic_ivt_linear_standard_grid",
        "declared_measurement_channels": channels,
        "per_channel_sample_cap": cap,
        "planned_samples_per_channel": samples,
        "unused_samples_per_channel": cap - samples,
        "planned_uniform_interval_us": interval_us,
        "displayed_sampling_duration_us": samples * interval_us,
        "first_nonorigin_planned_start_us": interval_us if samples > 1 else None,
        "last_planned_start_elapsed_us": (samples - 1) * interval_us,
        "planned_nonorigin_starts_in_early_window": planned_start_count(1, EARLY_END_US, interval_us, samples),
        "planned_starts_in_terminal_window": planned_start_count(
            TERMINAL_START_US, TERMINAL_END_US, interval_us, samples
        ),
        "exact_1s_on_planned_grid": exact_slot(1_000_000),
        "exact_100ks_on_planned_grid": exact_slot(100_000_000_000),
        "latest_start_at_sample_cap_with_1p02s_interval_us": (cap - 1) * EARLY_END_US,
        "minimum_grid_interval_to_reach_80ks_at_sample_cap_us": min_interval,
    }
    if diagnose_design:
        report["joint_anchor_design"] = joint_anchor_design(cap)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-classic-linear-standard-grid", action="store_true",
                        help="confirm the saved setup uses this specific mode and interval grid")
    parser.add_argument("--measurement-channels", type=int, required=True)
    parser.add_argument("--sample-interval-us", type=int, required=True)
    parser.add_argument("--samples-per-channel", type=int, required=True)
    parser.add_argument("--diagnose-joint-anchor-design", action="store_true",
                        help="minimum-sample alternative recipes for both windows and both exact nominals")
    args = parser.parse_args()
    if not args.confirm_classic_linear_standard_grid:
        parser.exit(2, "error: explicit confirmation of Classic linear standard-grid setup is required\n")
    try:
        report = preflight(args.measurement_channels, args.sample_interval_us, args.samples_per_channel,
                           diagnose_design=args.diagnose_joint_anchor_design)
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
