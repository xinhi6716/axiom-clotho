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
