use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-rram-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn write(&self, data: &str) {
        fs::write(self.0.join("data.csv"), data).unwrap();
    }
    fn run(&self, extra: &[&str]) -> Output {
        let mut args = vec![
            "extract-rram",
            "--csv",
            "data.csv",
            "--out",
            "out",
            "--max-bracket-v",
            "1",
            "--accept-assumptions",
            "true",
        ];
        args.extend_from_slice(extra);
        Command::new(env!("CARGO_BIN_EXE_clotho"))
            .current_dir(&self.0)
            .args(args)
            .output()
            .unwrap()
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
const DATA:&str="batch,wafer,device,cycle,replicate,branch,state,condition,protocol,read_voltage_V,temperature_K,read_delay_s,voltage_V,current_A\nB,W,D,1,R1,H,HRS,C,P,1,300,0,0.5,0.005\nB,W,D,1,R1,H,HRS,C,P,1,300,0,1.5,0.015\nB,W,D,1,R1,L,LRS,C,P,1,300,0,1,0.1\n";
#[test]
fn native_bundle_retains_input_lineage_and_no_overwrite() {
    let w = Workspace::new();
    w.write(DATA);
    let r = w.run(&[]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(
        fs::read(w.0.join("out/measurement_input.csv")).unwrap(),
        DATA.as_bytes()
    );
    let j: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("out/extraction.json")).unwrap()).unwrap();
    assert_eq!(j["schema_version"], "clotho.rram_si_binary64.v1");
    assert_eq!(j["ml_fits"], 0);
    assert_eq!(j["feature_rows"], 1);
    assert_eq!(
        j["lineage"][0]["pairs"][0]["HRS"]["selected_source_rows"],
        serde_json::json!([2, 3])
    );
    assert_eq!(j["lineage"][0]["pairs"][0]["resistance_ratio"], 10.0);
    assert!(w.0.join("out/COMPLETE").is_file());
    let before = fs::read(w.0.join("out/features.csv")).unwrap();
    assert!(!w.run(&[]).status.success());
    assert_eq!(before, fs::read(w.0.join("out/features.csv")).unwrap());
}
#[test]
fn bad_input_fails_before_writes_with_private_diagnostics() {
    for data in [
        DATA.replace("0.005", "NaN"),
        DATA.replace("B,W,D", "B,W,=secret"),
        DATA.replace(",300,0,1,", ",301,0,1,"),
        DATA.replace("current_A", "current"),
        DATA.replace(",L,LRS,", ",L,HRS,"),
        DATA.replace(",0.1\n", ",-0.1\n"),
        DATA.lines().take(3).collect::<Vec<_>>().join("\n"),
    ] {
        let w = Workspace::new();
        w.write(&data);
        let r = w.run(&[]);
        assert!(!r.status.success());
        assert!(!w.0.join("out").exists());
        assert!(!String::from_utf8_lossy(&r.stderr).contains("secret"));
    }
}
#[test]
fn numeric_underflow_rejected_in_conditions_and_options() {
    for tiny in ["1e-999", "-1e-999", "0.0001E-999"] {
        let w = Workspace::new();
        w.write(&DATA.replace(",300,0,", &format!(",300,{tiny},")));
        assert!(!w.run(&[]).status.success());
        assert!(!w.0.join("out").exists());
        let w = Workspace::new();
        w.write(DATA);
        let r = Command::new(env!("CARGO_BIN_EXE_clotho"))
            .current_dir(&w.0)
            .args([
                "extract-rram",
                "--csv",
                "data.csv",
                "--out",
                "out",
                "--max-bracket-v",
                tiny,
                "--accept-assumptions",
                "true",
            ])
            .output()
            .unwrap();
        assert!(!r.status.success());
        assert!(!w.0.join("out").exists());
    }
    let w = Workspace::new();
    w.write(&DATA.replace(",300,0,", ",300,-0.000e100,"));
    assert!(w.run(&[]).status.success());
}
#[test]
fn explicit_assumption_acceptance_is_required() {
    let w = Workspace::new();
    w.write(DATA);
    let r = Command::new(env!("CARGO_BIN_EXE_clotho"))
        .current_dir(&w.0)
        .args([
            "extract-rram",
            "--csv",
            "data.csv",
            "--out",
            "out",
            "--max-bracket-v",
            "1",
            "--accept-assumptions",
            "false",
        ])
        .output()
        .unwrap();
    assert!(!r.status.success());
    assert!(!w.0.join("out").exists());
}
#[test]
fn condition_rows_need_opt_in_and_stay_separate() {
    let second = DATA
        .lines()
        .skip(1)
        .collect::<Vec<_>>()
        .join("\n")
        .replace("D,1,R1", "D,2,R2")
        .replace(",C,P,", ",C2,P,")
        .replace(",300,", ",310,");
    let w = Workspace::new();
    w.write(&format!("{DATA}{second}\n"));
    assert!(!w.run(&[]).status.success());
    assert!(!w.0.join("out").exists());
    assert!(w
        .run(&["--feature-row-unit", "device-condition"])
        .status
        .success());
    let j: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("out/extraction.json")).unwrap()).unwrap();
    assert_eq!(j["feature_rows"], 2);
}
#[cfg(unix)]
#[test]
fn bundle_is_private() {
    use std::os::unix::fs::PermissionsExt;
    let w = Workspace::new();
    w.write(DATA);
    assert!(w.run(&[]).status.success());
    assert_eq!(
        fs::metadata(w.0.join("out")).unwrap().permissions().mode() & 0o777,
        0o700
    );
    for name in [
        "features.csv",
        "measurement_input.csv",
        "extraction.json",
        "COMPLETE",
    ] {
        assert_eq!(
            fs::metadata(w.0.join("out").join(name))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}

#[test]
fn extraction_report_binds_exact_input_and_feature_bytes() {
    use sha2::{Digest, Sha256};
    let capture = |data: &str| {
        let w = Workspace::new();
        w.write(data);
        let output = w.run(&[]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let digest = |name| format!("{:x}", Sha256::digest(fs::read(w.0.join(name)).unwrap()));
        let j: serde_json::Value =
            serde_json::from_slice(&fs::read(w.0.join("out/extraction.json")).unwrap()).unwrap();
        let input = digest("out/measurement_input.csv");
        let features = digest("out/features.csv");
        assert_eq!(j["input_sha256"], input);
        assert_eq!(j["features_sha256"], features);
        (input, features)
    };
    let original = capture(DATA);
    // Equivalent formatting changes the exact input bytes, not the feature CSV.
    let reformatted = capture(&DATA.replace(",300,", ",300.0,"));
    assert_ne!(original.0, reformatted.0);
    assert_eq!(original.1, reformatted.1);
    // A same-shape trace change must not reproduce the old feature digest.
    let changed = capture(&DATA.replacen("0.1\n", "0.2\n", 1));
    assert_ne!(original.0, changed.0);
    assert_ne!(original.1, changed.1);
}

#[test]
fn malformed_quote_framing_does_not_complete_a_capture() {
    let stem = DATA.trim_end().strip_suffix("0.1").unwrap();
    for malformed in [
        format!("{stem}\"0.1"),
        DATA.replace(",C,P,", ",C,P\"raw,"),
        DATA.replace(",C,P,", ",C,\"P\"raw,"),
    ] {
        let w = Workspace::new();
        w.write(&malformed);
        let result = w.run(&[]);
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("Invalid RRAM CSV quoting"));
        assert!(!w.0.join("out").exists());
    }
    for last in ["0.1", "\"0.1\""] {
        let w = Workspace::new();
        let valid = format!("{stem}{last}").replace(",C,P,", ",C,\"P,\"\"quoted\"\"\",");
        w.write(&valid);
        let result = w.run(&[]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(w.0.join("out/COMPLETE").is_file());
    }
}

#[test]
fn readpoint_support_counts_are_disjoint_per_feature_row() {
    let w = Workspace::new();
    let mut trace = format!("{}\n", DATA.lines().next().unwrap());
    for (device, cycle, hrs_interpolated, lrs_interpolated) in [
        ("D", "1", false, false),
        ("D", "2", true, false),
        ("D2", "1", false, false),
        ("D", "3", true, false),
        ("D", "4", false, true),
        ("D", "5", true, true),
    ] {
        for (suffix, state, current, interpolate) in [
            ("H", "HRS", "0.125", hrs_interpolated),
            ("L", "LRS", "0.25", lrs_interpolated),
        ] {
            let voltages: &[&str] = if interpolate { &["0.5", "1.5"] } else { &["1"] };
            for voltage in voltages {
                trace.push_str(&format!(
                    "B,W,{device},{cycle},R{cycle},{suffix},{state},C,P,1,300,0,{voltage},{current}\n"
                ));
            }
        }
    }
    w.write(&trace);
    let result = w.run(&[]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let data: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("out/extraction.json")).unwrap()).unwrap();
    for (i, device, counts) in [(0, "D", [1, 2, 1, 1]), (1, "D2", [1, 0, 0, 0])] {
        let row = &data["lineage"][i];
        assert_eq!(row["identity"][2], device);
        assert_eq!(
            row["cycle_readpoint_support_counts"],
            serde_json::json!({
                "both_sampled": counts[0],
                "hrs_only_interpolated": counts[1],
                "lrs_only_interpolated": counts[2],
                "both_interpolated": counts[3],
            })
        );
        let pairs = row["pairs"].as_array().unwrap();
        assert_eq!(pairs.len(), counts.iter().sum::<usize>());
        assert!(pairs.iter().all(|pair| pair["resistance_ratio"] == 2.0));
    }
}

#[test]
fn sampled_only_policy_rejects_each_interpolation_pattern_without_partial_output() {
    let gate = ["--require-sampled-readpoints", "true"];
    let read = "B,W,D,1,R1,H,HRS,C,P,1,300,0,1,0.01";
    let regular: Vec<_> = DATA.lines().collect();
    let exact = format!("{}\n{read}\n{}\n", regular[0], regular[3]);
    let sampled = Workspace::new();
    sampled.write(&exact);
    let result = sampled.run(&gate);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let j: serde_json::Value =
        serde_json::from_slice(&fs::read(sampled.0.join("out/extraction.json")).unwrap()).unwrap();
    assert_eq!(j["settings"]["require_sampled_readpoints"], true);
    assert!(sampled.0.join("out/COMPLETE").is_file());

    let lrs_bracket = "B,W,D,1,R1,L,LRS,C,P,1,300,0,0.5,0.1\nB,W,D,1,R1,L,LRS,C,P,1,300,0,1.5,0.1";
    for trace in [
        DATA.to_owned(), // HRS only
        format!("{}\n{read}\n{lrs_bracket}\n", regular[0]),
        format!("{}\n{}\n{}\n{lrs_bracket}\n", regular[0], regular[1], regular[2]),
    ] {
        let old = Workspace::new();
        old.write(&trace);
        assert!(old.run(&[]).status.success());
        let new = Workspace::new();
        new.write(&trace);
        let rejected = new.run(&gate);
        assert!(!rejected.status.success());
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("requires sampled RRAM readpoints"));
        assert!(!new.0.join("out").exists());
    }

    let mixed = format!(
        "{exact}{}\n{}\n{}\n",
        regular[1].replace(",D,1,R1,", ",D,2,R2,"),
        regular[2].replace(",D,1,R1,", ",D,2,R2,"),
        regular[3].replace(",D,1,R1,", ",D,2,R2,"),
    );
    let unconfigured = Workspace::new();
    unconfigured.write(&mixed);
    assert!(unconfigured.run(&[]).status.success());
    let off = Workspace::new();
    off.write(&mixed);
    assert!(off.run(&["--require-sampled-readpoints", "false"]).status.success());
    for file in ["features.csv", "measurement_input.csv", "extraction.json", "COMPLETE"] {
        assert_eq!(
            fs::read(unconfigured.0.join("out").join(file)).unwrap(),
            fs::read(off.0.join("out").join(file)).unwrap()
        );
    }
    let strict = Workspace::new();
    strict.write(&mixed);
    assert!(!strict.run(&gate).status.success());
    assert!(!strict.0.join("out").exists());

    let invalid = Workspace::new();
    invalid.write(&exact);
    assert!(!invalid.run(&["--require-sampled-readpoints", "1"]).status.success());
    assert!(!invalid.0.join("out").exists());
}

#[test]
fn even_cycle_median_preserves_subnormal_ratios() {
    let w = Workspace::new();
    let mut capture = format!("{}\n", DATA.lines().next().unwrap());
    for (cycle, endpoint_current) in [(1, "5e-24"), (2, "1e-23")] {
        capture.push_str(&format!(
            "B,W,D,{cycle},R{cycle},H,HRS,C,P,1e-150,300,0,1e-150,1\n\
             B,W,D,{cycle},R{cycle},L,LRS,C,P,1e-150,300,0,0,0\n\
             B,W,D,{cycle},R{cycle},L,LRS,C,P,1e-150,300,0,1e150,{endpoint_current}\n"
        ));
    }
    w.write(&capture);
    let output = Command::new(env!("CARGO_BIN_EXE_clotho"))
        .current_dir(&w.0)
        .args([
            "extract-rram",
            "--csv",
            "data.csv",
            "--out",
            "out",
            "--max-bracket-v",
            "1e150",
            "--accept-assumptions",
            "true",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let j: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("out/extraction.json")).unwrap()).unwrap();
    let pairs = j["lineage"][0]["pairs"].as_array().unwrap();
    let bits: Vec<_> = pairs
        .iter()
        .map(|pair| pair["resistance_ratio"].as_f64().unwrap().to_bits())
        .collect();
    assert_eq!(bits, [1, 2]);
    let csv = fs::read_to_string(w.0.join("out/features.csv")).unwrap();
    let median = csv.lines().nth(1).unwrap().rsplit(',').next().unwrap();
    assert_eq!(median.parse::<f64>().unwrap().to_bits(), 2);
}

#[test]
fn rounded_bracket_tie_checks_exact_binary64_width() {
    let header = DATA.lines().next().unwrap();
    for (lo, accepted) in [("0", true), ("-5.551115123125783e-17", false)] {
        let mut hrs = [
            format!("B,W,D,1,R1,H,HRS,C,P,0.5,300,0,{lo},0.01"),
            "B,W,D,1,R1,H,HRS,C,P,0.5,300,0,1,0.01".to_owned(),
        ];
        for _ in 0..2 {
            let w = Workspace::new();
            w.write(&format!(
                "{header}\n{}\n{}\nB,W,D,1,R1,L,LRS,C,P,0.5,300,0,0.5,0.1\n",
                hrs[0], hrs[1]
            ));
            let result = w.run(&[]);
            assert_eq!(
                result.status.success(),
                accepted,
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            if accepted {
                assert!(w.0.join("out/COMPLETE").is_file());
            } else {
                assert!(String::from_utf8_lossy(&result.stderr).contains("BracketTooWide"));
                assert!(!w.0.join("out").exists());
            }
            hrs.reverse();
        }
    }
}
