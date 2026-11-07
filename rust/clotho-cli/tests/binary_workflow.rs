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
            "clotho-binary-{}-{}-{}",
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
const DATA: &str = "wafer,x,label\na,-2,0\na,2,1\nb,-1,0\nb,1,1\nc,-3,0\nc,3,1\n";
#[test]
fn logistic_fit_reload_and_probability_predictions() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    w.write("query.csv", "x\n-2\n2\n");
    let r = w.run(&[
        "fit",
        "--csv",
        "data.csv",
        "--target",
        "label",
        "--features",
        "x",
        "--model",
        "logistic",
        "--alpha",
        "0.1",
        "--out",
        "model.json",
    ]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let model: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("model.json")).unwrap()).unwrap();
    assert_eq!(model["schema_version"], "clotho.numeric.v3");
    let r = w.run(&[
        "predict",
        "--csv",
        "query.csv",
        "--artifact",
        "model.json",
        "--out",
        "p.csv",
    ]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let s = fs::read_to_string(w.0.join("p.csv")).unwrap();
    let rows: Vec<_> = s.lines().collect();
    assert_eq!(rows[0], "row_index,positive_probability,predicted_label");
    let low: f64 = rows[1].split(',').nth(1).unwrap().parse().unwrap();
    let high: f64 = rows[2].split(',').nth(1).unwrap().parse().unwrap();
    assert!(low < 0.5 && high > 0.5 && low >= 0.0 && high <= 1.0);
}
#[test]
fn grouped_logistic_evaluation_reports_probabilistic_metrics() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    let r = w.run(&[
        "evaluate",
        "--csv",
        "data.csv",
        "--target",
        "label",
        "--features",
        "x",
        "--model",
        "logistic",
        "--alpha",
        "0.1",
        "--group",
        "wafer",
        "--folds",
        "3",
        "--out",
        "run",
    ]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let m: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("run/metrics.json")).unwrap()).unwrap();
    assert_eq!(m["schema_version"], "clotho.grouped_binary_evaluation.v1");
    assert_eq!(m["n_rows"], 6);
    assert_eq!(m["pooled_row_weighted"]["accuracy"], 1.0);
    assert_eq!(m["pooled_row_weighted"]["roc_auc"], 1.0);
    assert!(m["pooled_row_weighted"]["log_loss"].as_f64().unwrap() > 0.0);
    assert!(w.0.join("run/COMPLETE").exists());
}
#[test]
fn majority_is_fitted_inside_each_group_fold_and_singleclass_auc_is_null() {
    let w = Workspace::new();
    w.write(
        "data.csv",
        "g,x,y\na,0,0\na,1,0\nb,2,1\nb,3,1\nc,4,1\nc,5,1\n",
    );
    let r = w.run(&[
        "evaluate",
        "--csv",
        "data.csv",
        "--target",
        "y",
        "--features",
        "x",
        "--model",
        "majority",
        "--group",
        "g",
        "--folds",
        "3",
        "--out",
        "run",
    ]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let s = fs::read_to_string(w.0.join("run/predictions.csv")).unwrap();
    let p: Vec<f64> = s
        .lines()
        .skip(1)
        .map(|r| r.split(',').nth(3).unwrap().parse().unwrap())
        .collect();
    assert_eq!(p, vec![1.0, 1.0, 0.5, 0.5, 0.5, 0.5]);
    let m: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("run/metrics.json")).unwrap()).unwrap();
    assert!(m["groups"][0]["metrics"]["roc_auc"].is_null());
    assert_eq!(
        m["equal_group_weighted"]["balanced_accuracy_group_denominator"],
        0
    );
}
#[test]
fn invalid_labels_singleclass_training_and_threshold_fail_without_outputs() {
    let w = Workspace::new();
    for data in [
        "g,x,y\na,0,0\na,1,0\nb,2,1\nb,3,1\n",
        "g,x,y\na,0,0\na,1,2\nb,2,0\nb,3,1\n",
    ] {
        w.write("data.csv", data);
        let r = w.run(&[
            "evaluate",
            "--csv",
            "data.csv",
            "--target",
            "y",
            "--features",
            "x",
            "--model",
            "logistic",
            "--group",
            "g",
            "--folds",
            "2",
            "--out",
            "bad",
        ]);
        assert!(!r.status.success());
        assert!(!w.0.join("bad").exists());
    }
    w.write("data.csv", DATA);
    for threshold in ["NaN", "-0.1", "1.1"] {
        let r = w.run(&[
            "evaluate",
            "--csv",
            "data.csv",
            "--target",
            "label",
            "--features",
            "x",
            "--model",
            "logistic",
            "--group",
            "wafer",
            "--folds",
            "3",
            "--threshold",
            threshold,
            "--out",
            "bad",
        ]);
        assert!(!r.status.success());
        assert!(!w.0.join("bad").exists());
    }
}
