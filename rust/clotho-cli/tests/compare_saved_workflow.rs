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
            "clotho-saved-{}-{}-{}",
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
    fn write(&self, name: &str, s: &str) {
        fs::write(self.0.join(name), s).unwrap();
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

fn series(values: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "schema_version": "clotho.declared_regression_series.v1",
        "cohort": "synthetic-1", "target": "resistance", "unit": "ohm",
        "horizon": "1 write", "row_ids": ["a", "b"], "values": values
    })
}

fn setup() -> Workspace {
    let w = Workspace::new();
    for (name, values) in [
        ("observed.json", serde_json::json!([1., 3.])),
        ("model.json", serde_json::json!([1., 2.])),
        ("baseline.json", serde_json::json!([0., 1.])),
    ] {
        w.write(name, &series(values).to_string());
    }
    w
}

const ARGS: [&str; 9] = [
    "compare-saved", "--observed", "observed.json", "--model", "model.json",
    "--baseline", "baseline.json", "--out", "report.json",
];

#[test]
fn scores_existing_series_without_model_artifacts() {
    let w = setup();
    let output = w.run(&ARGS);
    assert!(output.status.success(), "{:?}", output);
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
    assert_eq!(
        report["schema_version"],
        "clotho.declared_regression_comparison.v1"
    );
    assert_eq!(report["cohort"], "synthetic-1");
    assert_eq!(report["n_rows"], 2);
    assert!((report["mse_skill"].as_f64().unwrap() - 0.8).abs() < 1e-14);
    for flag in ["fit_performed", "provenance_verified", "query_bytes_verified"] {
        assert_eq!(report[flag], false);
    }
    assert!(report.get("row_ids").is_none());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(w.0.join("report.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}

#[test]
fn invalid_inputs_leave_no_output() {
    for (field, value) in [
        ("row_ids", serde_json::json!(["b", "a"])),
        ("row_ids", serde_json::json!(["a", "a"])),
        ("row_ids", serde_json::json!(["a"])),
        ("row_ids", serde_json::json!(["a", " "])),
        ("values", serde_json::json!([1.])),
        ("values", serde_json::json!([1., null])),
        ("cohort", serde_json::json!("other")),
        ("target", serde_json::json!("conductance")),
        ("unit", serde_json::json!("kohm")),
        ("horizon", serde_json::json!("2 writes")),
        ("schema_version", serde_json::json!("unknown")),
        ("extra", serde_json::json!("not allowed")),
    ] {
        for file in ["observed.json", "model.json", "baseline.json"] {
            let w = setup();
            let mut bad = series(serde_json::json!([1., 2.]));
            bad[field] = value.clone();
            w.write(file, &bad.to_string());
            let output = w.run(&ARGS);
            assert_eq!(output.status.code(), Some(2), "{file} {field}");
            assert!(!w.0.join("report.json").exists());
        }
    }
}

#[test]
fn strict_json_and_cli_reject_before_output() {
    for contents in [
        "not json", "[]", "{\"schema_version\":\"x\",\"schema_version\":\"x\"}",
        r#"["clotho.declared_regression_series.v1","synthetic-1","resistance","ohm","1 write",["a","b"],[1,2]]"#,
        "row_index,prediction\n0,1\n1,2\n",
    ] {
        let w = setup();
        w.write("model.json", contents);
        assert_eq!(w.run(&ARGS).status.code(), Some(2));
        assert!(!w.0.join("report.json").exists());
    }
    let w = setup();
    let mut args = ARGS.to_vec();
    args.extend(["--artifact", "nonexistent.json"]);
    assert_eq!(w.run(&args).status.code(), Some(2));
    assert_eq!(w.run(&ARGS[..7]).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());
}

#[test]
fn perfect_baseline_is_json_null_and_existing_output_is_preserved() {
    let w = setup();
    w.write("baseline.json", &series(serde_json::json!([1., 3.])).to_string());
    assert!(w.run(&ARGS).status.success());
    let bytes = fs::read(w.0.join("report.json")).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(report["mse_skill"].is_null());
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert_eq!(fs::read(w.0.join("report.json")).unwrap(), bytes);
}

#[test]
fn matching_false_keys_do_not_verify_provenance() {
    let w = setup();
    w.write("model.json", &series(serde_json::json!([3., 1.])).to_string());
    assert!(w.run(&ARGS).status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["provenance_verified"], false);
    assert!(report["mse_skill"].as_f64().unwrap() < 0.);
}

#[test]
fn capacity_and_unrepresentable_scores_leave_no_output() {
    let w = setup();
    let mut bad = series(serde_json::json!([1., 2.]));
    bad["row_ids"] = serde_json::json!(vec!["a"; 20_001]);
    bad["values"] = serde_json::json!(vec![1.; 20_001]);
    w.write("model.json", &bad.to_string());
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());

    let w = setup();
    w.write("observed.json", &series(serde_json::json!([0., 0.])).to_string());
    w.write("baseline.json", &series(serde_json::json!([1e-300, 1e-300])).to_string());
    w.write("model.json", &series(serde_json::json!([1e300, 1e300])).to_string());
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());
}
