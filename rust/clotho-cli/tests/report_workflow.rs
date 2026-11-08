use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};
struct Work(PathBuf);
impl Work {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-report-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_clotho"))
            .args(args)
            .current_dir(&self.0)
            .output()
            .unwrap()
    }
    fn eval(&self, binary: bool) {
        fs::write(
            self.0.join("data.csv"),
            if binary {
                "wafer,x,y\na,0,0\na,1,1\nb,2,0\nb,3,1\nc,4,0\nc,5,1\n"
            } else {
                "wafer,x,y\na,0,1\na,1,3\nb,2,5\nb,3,7\nc,4,9\nc,5,11\n"
            },
        )
        .unwrap();
        let o = self.run(&[
            "evaluate",
            "--csv",
            "data.csv",
            "--features",
            "x",
            "--target",
            "y",
            "--group",
            "wafer",
            "--model",
            if binary { "majority" } else { "mean" },
            "--folds",
            "3",
            "--out",
            "evaluation",
        ]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    fn report(&self) -> Output {
        self.run(&["report", "--run", "evaluation", "--out", "report.html"])
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn regression_end_to_end_no_clobber() {
    let w = Work::new();
    w.eval(false);
    assert!(w.report().status.success());
    let h = fs::read_to_string(w.0.join("report.html")).unwrap();
    assert!(h.contains("Regression workspace"));
    assert!(h.contains("6 of 6 rows"));
    assert!(h.contains("script-src 'none'"));
    assert!(!h.contains("<script"));
    assert!(!w.report().status.success());
    assert_eq!(fs::read_to_string(w.0.join("report.html")).unwrap(), h);
}
#[test]
fn binary_end_to_end() {
    let w = Work::new();
    w.eval(true);
    assert!(w.report().status.success());
    let h = fs::read_to_string(w.0.join("report.html")).unwrap();
    assert!(h.contains("Binary classification"));
    assert!(h.contains("positive probability"));
}
#[test]
fn incomplete_runs_rejected() {
    let w = Work::new();
    w.eval(false);
    fs::remove_file(w.0.join("evaluation/COMPLETE")).unwrap();
    assert!(!w.report().status.success());
    assert!(!w.0.join("report.html").exists());
}
#[test]
fn schema_marker_mismatch_rejected() {
    let w = Work::new();
    w.eval(false);
    fs::write(w.0.join("evaluation/COMPLETE"), "other\n").unwrap();
    assert!(!w.report().status.success());
}
#[test]
fn row_alignment_rejected() {
    let w = Work::new();
    w.eval(false);
    let p = w.0.join("evaluation/predictions.csv");
    let s = fs::read_to_string(&p).unwrap().replacen("0,", "99,", 1);
    fs::write(p, s).unwrap();
    assert!(!w.report().status.success());
    assert!(!w.0.join("report.html").exists());
}
#[test]
fn metric_injection_is_escaped() {
    let w = Work::new();
    w.eval(false);
    let p = w.0.join("evaluation/metrics.json");
    let mut m: serde_json::Value = serde_json::from_slice(&fs::read(&p).unwrap()).unwrap();
    m["target"] = serde_json::json!("</pre><script>alert('x')</script>");
    fs::write(p, serde_json::to_vec(&m).unwrap()).unwrap();
    assert!(w.report().status.success());
    let h = fs::read_to_string(w.0.join("report.html")).unwrap();
    assert!(!h.contains("<script>"));
    assert!(h.contains("&lt;script&gt;"));
}
#[test]
fn unknown_options_rejected() {
    let w = Work::new();
    assert!(!w.run(&["report", "--typo", "x"]).status.success());
}
#[cfg(unix)]
#[test]
fn output_symlink_is_not_followed() {
    let w = Work::new();
    w.eval(false);
    fs::write(w.0.join("keep"), "untouched").unwrap();
    std::os::unix::fs::symlink("keep", w.0.join("report.html")).unwrap();
    assert!(!w.report().status.success());
    assert_eq!(fs::read_to_string(w.0.join("keep")).unwrap(), "untouched");
}
#[cfg(unix)]
#[test]
fn input_symlink_rejected() {
    let w = Work::new();
    w.eval(false);
    fs::rename(
        w.0.join("evaluation/metrics.json"),
        w.0.join("metrics.json"),
    )
    .unwrap();
    std::os::unix::fs::symlink("../metrics.json", w.0.join("evaluation/metrics.json")).unwrap();
    assert!(!w.report().status.success());
}
#[test]
fn predicted_labels_must_match_saved_threshold() {
    for invalid in ["banana", "2", "0"] {
        let w = Work::new();
        w.eval(true);
        let path = w.0.join("evaluation/predictions.csv");
        let text = fs::read_to_string(&path).unwrap();
        let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
        let mut cells: Vec<String> = lines[1].split(',').map(str::to_owned).collect();
        assert_eq!(cells[4], "1");
        cells[4] = invalid.into();
        lines[1] = cells.join(",");
        fs::write(path, lines.join("\n") + "\n").unwrap();
        assert!(!w.report().status.success());
        assert!(!w.0.join("report.html").exists());
    }
}
#[test]
fn binary_threshold_required_and_bounded() {
    for invalid in [
        serde_json::Value::Null,
        serde_json::json!(-0.1),
        serde_json::json!(1.1),
    ] {
        let w = Work::new();
        w.eval(true);
        let path = w.0.join("evaluation/metrics.json");
        let mut m: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        m["threshold"] = invalid;
        fs::write(path, serde_json::to_vec(&m).unwrap()).unwrap();
        assert!(!w.report().status.success());
    }
}
#[test]
fn binary_ui_preserves_classification_semantics() {
    let w = Work::new();
    w.eval(true);
    assert!(w.report().status.success());
    let h = fs::read_to_string(w.0.join("report.html")).unwrap();
    assert!(!h.contains("Mean group RMSE"));
    assert!(h.contains("<strong>Positive label:</strong> 1"));
    assert!(h.contains("<strong>Decision threshold:</strong> 0.5"));
    assert!(h.contains("<th scope=col>Predicted label</th>"));
    assert!(h.contains("<th scope=col>Positive probability</th>"));
}
#[test]
fn unsupported_positive_label_is_rejected() {
    let w = Work::new();
    w.eval(true);
    let path = w.0.join("evaluation/metrics.json");
    let mut m: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    m["positive_label"] = serde_json::json!(0);
    fs::write(path, serde_json::to_vec(&m).unwrap()).unwrap();
    assert!(!w.report().status.success());
}
#[test]
fn incomplete_or_inconsistent_metric_shapes_are_rejected() {
    for binary in [false, true] {
        for case in 0..8 {
            let w = Work::new();
            w.eval(binary);
            let path = w.0.join("evaluation/metrics.json");
            let mut m: serde_json::Value =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            match case {
                0 => m["n_groups"] = serde_json::json!(999),
                1 => m["groups"] = serde_json::json!([]),
                2 => m["folds"] = serde_json::json!([]),
                3 => m["pooled_row_weighted"] = serde_json::json!({}),
                4 => m["equal_group_weighted"] = serde_json::json!({}),
                5 => m["groups"][0]["n_rows"] = serde_json::json!(999),
                6 => m["folds"][0]["n_test"] = serde_json::json!(999),
                _ => m["folds"][0]["fold"] = serde_json::json!(2),
            }
            fs::write(path, serde_json::to_vec(&m).unwrap()).unwrap();
            assert!(!w.report().status.success(), "binary={binary} case={case}");
            assert!(!w.0.join("report.html").exists());
        }
    }
}
