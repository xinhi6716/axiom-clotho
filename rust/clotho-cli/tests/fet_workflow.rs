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
            "clotho-fet-{}-{}-{}",
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
    fn write(&self, n: &str, s: &str) {
        fs::write(self.0.join(n), s).unwrap();
    }
    fn run(&self, a: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_clotho"))
            .current_dir(&self.0)
            .args(a)
            .output()
            .unwrap()
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
const DATA:&str="lot,wafer,device,condition,replicate,branch,temperature_K,drain_source_voltage_V,body_source_voltage_V,gate_source_voltage_V,drain_current_A\nL,W,D,C,R,B,300,1,0,0,1e-9\nL,W,D,C,R,B,300,1,0,1,1e-8\nL,W,D,C,R,B,300,1,0,2,1e-7\nL,W,D,C,R,B,300,1,0,3,1e-6\n";
fn extract(w: &Workspace, consent: &str) -> Output {
    w.run(&[
        "extract-fet",
        "--csv",
        "data.csv",
        "--out",
        "out",
        "--criterion-a",
        "1e-8",
        "--polarity",
        "positive",
        "--max-bracket-v",
        "1",
        "--window-min-v",
        "0",
        "--window-max-v",
        "3",
        "--accept-assumptions",
        consent,
    ])
}
#[test]
fn native_extraction_retains_input_and_lineage() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    let r = extract(&w, "true");
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(
        fs::read(w.0.join("out/transfer_input.csv")).unwrap(),
        DATA.as_bytes()
    );
    let j: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("out/extraction.json")).unwrap()).unwrap();
    assert_eq!(j["ml_fits"], 0);
    assert_eq!(j["device_rows"], 1);
    assert_eq!(
        j["lineage"][0]["criterion_source_rows"],
        serde_json::json!([3])
    );
    assert!(w.0.join("out/COMPLETE").is_file());
    let before = fs::read(w.0.join("out/features.csv")).unwrap();
    assert!(!extract(&w, "true").status.success());
    assert_eq!(fs::read(w.0.join("out/features.csv")).unwrap(), before);
}
#[test]
fn rejects_invalid_before_writing() {
    for data in [
        DATA.replace(",300,1,0,1,", ",301,1,0,1,"),
        DATA.replace("1e-8", "NaN"),
        DATA.replace("L,W,D", "L,W,=secret"),
    ] {
        let w = Workspace::new();
        w.write("data.csv", &data);
        let r = extract(&w, "true");
        assert!(!r.status.success());
        assert!(!w.0.join("out").exists());
        assert!(!String::from_utf8_lossy(&r.stderr).contains("secret"));
    }
}
#[test]
fn consent_required() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    assert!(!extract(&w, "false").status.success());
    assert!(!w.0.join("out").exists());
}
#[test]
fn requires_explicit_si_headers() {
    let w = Workspace::new();
    w.write(
        "data.csv",
        &DATA.replace("drain_current_A", "drain_current"),
    );
    assert!(!extract(&w, "true").status.success());
    assert!(!w.0.join("out").exists());
}
#[cfg(unix)]
#[test]
fn private_outputs() {
    use std::os::unix::fs::PermissionsExt;
    let w = Workspace::new();
    w.write("data.csv", DATA);
    assert!(extract(&w, "true").status.success());
    assert_eq!(
        fs::metadata(w.0.join("out")).unwrap().permissions().mode() & 0o777,
        0o700
    );
    for name in [
        "features.csv",
        "transfer_input.csv",
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
fn nonzero_underflow_cannot_masquerade_as_zero_condition() {
    for tiny in ["1e-999", "-1e-999", "0.0001E-999"] {
        let w = Workspace::new();
        w.write(
            "data.csv",
            &DATA.replacen(",300,1,0,0,", &format!(",300,1,{tiny},0,"), 1),
        );
        assert!(!extract(&w, "true").status.success());
        assert!(!w.0.join("out").exists());
    }
}
#[test]
fn underflow_window_option_rejected_but_explicit_zero_allowed() {
    for (value, accepted) in [
        ("1e-999", false),
        ("-1e-999", false),
        ("0e-999", true),
        ("-0.000e100", true),
    ] {
        let w = Workspace::new();
        w.write("data.csv", DATA);
        let result = w.run(&[
            "extract-fet",
            "--csv",
            "data.csv",
            "--out",
            "out",
            "--criterion-a",
            "1e-8",
            "--polarity",
            "positive",
            "--max-bracket-v",
            "1",
            "--window-min-v",
            value,
            "--window-max-v",
            "3",
            "--accept-assumptions",
            "true",
        ]);
        assert_eq!(
            result.status.success(),
            accepted,
            "option {value}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(w.0.join("out").exists(), accepted);
    }
}
