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
            "clotho-compare-{}-{}-{}",
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
const DATA: &str = "g,x,y\na,0,1\na,1,3\nb,2,5\nb,3,7\nc,4,9\nc,5,11\nd,6,13\nd,7,15\n";
fn compare(w: &Workspace, candidates: &str, budget: &str, out: &str) -> Output {
    w.run(&[
        "compare",
        "--csv",
        "data.csv",
        "--target",
        "y",
        "--features",
        "x",
        "--group",
        "g",
        "--candidates",
        candidates,
        "--outer-folds",
        "4",
        "--inner-folds",
        "2",
        "--max-fits",
        budget,
        "--out",
        out,
    ])
}
#[test]
fn nested_compare_selects_inner_best_and_preserves_predictions() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    let r = compare(&w, "mean,ridge:0,knn:1", "28", "run");
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let m: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("run/selection.json")).unwrap()).unwrap();
    assert_eq!(m["schema_version"], "clotho.nested_regression.v1");
    assert_eq!(m["planned_fits"], 28);
    assert_eq!(m["attempted_fits"], 28);
    assert!(m["pooled_outer_metrics"]["rmse"].as_f64().unwrap() < 1e-10);
    for f in m["folds"].as_array().unwrap() {
        assert_eq!(f["selected_candidate_index"], 1);
        assert_eq!(f["inner_candidate_scores"].as_array().unwrap().len(), 3);
    }
    let csv = fs::read_to_string(w.0.join("run/predictions.csv")).unwrap();
    assert_eq!(csv.lines().count(), 9);
    assert!(w.0.join("run/COMPLETE").exists());
    let r = compare(&w, "mean,ridge:0,knn:1", "28", "run2");
    assert!(r.status.success());
    assert_eq!(
        fs::read(w.0.join("run/predictions.csv")).unwrap(),
        fs::read(w.0.join("run2/predictions.csv")).unwrap()
    );
}
#[test]
fn preflight_rejects_budget_invalid_candidates_and_no_overwrite() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    for (c, b) in [
        ("mean,ridge:0,knn:1", "27"),
        ("ridge:NaN", "100"),
        ("ridge:-1", "100"),
        ("knn:0", "100"),
        ("knn:100", "100"),
        ("randomforest", "100"),
        ("", "100"),
    ] {
        let r = compare(&w, c, b, "bad");
        assert!(!r.status.success(), "candidate {c} unexpectedly passed");
        assert!(!w.0.join("bad").exists());
    }
    assert!(compare(&w, "mean", "12", "run").status.success());
    let before = fs::read(w.0.join("run/selection.json")).unwrap();
    assert!(!compare(&w, "mean", "12", "run").status.success());
    assert_eq!(before, fs::read(w.0.join("run/selection.json")).unwrap());
}
#[test]
fn candidate_failures_are_recorded_not_discarded() {
    let w = Workspace::new();
    w.write(
        "data.csv",
        "g,x,y\na,1,1\na,1,2\nb,1,3\nb,1,4\nc,1,5\nc,1,6\nd,1,7\nd,1,8\n",
    );
    let r = compare(&w, "ridge:0,mean", "20", "run");
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let m: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("run/selection.json")).unwrap()).unwrap();
    assert_eq!(m["planned_fits"], 20);
    assert_eq!(m["attempted_fits"], 16);
    for f in m["folds"].as_array().unwrap() {
        assert_eq!(f["selected_candidate_index"], 1);
        assert!(f["inner_candidate_scores"][0]["pooled_inner_rmse"].is_null());
        assert_eq!(f["inner_candidate_scores"][0]["failure"]["stage"], "Fit");
    }
}
#[test]
fn compare_does_not_accept_globally_preprocessed_mixed_flags_or_target_features() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    let r = w.run(&[
        "compare",
        "--csv",
        "data.csv",
        "--target",
        "y",
        "--features",
        "x",
        "--group",
        "g",
        "--candidates",
        "mean",
        "--preprocess",
        "mixed",
        "--out",
        "bad",
    ]);
    assert!(!r.status.success());
    assert!(!w.0.join("bad").exists());
    let r = w.run(&[
        "compare",
        "--csv",
        "data.csv",
        "--target",
        "y",
        "--features",
        "y",
        "--group",
        "g",
        "--candidates",
        "mean",
        "--out",
        "bad",
    ]);
    assert!(!r.status.success());
    assert!(!w.0.join("bad").exists());
}
