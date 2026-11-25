use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn pooled_binary_auc_includes_cross_fold_pairs() {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let location = std::env::temp_dir().join(format!(
        "clotho-auc-{}-{nonce}", std::process::id()
    ));
    fs::create_dir(&location).unwrap();
    let scratch = Scratch(location);
    fs::write(scratch.0.join("fixture.csv"),
        "x,label,wafer\n0,1,A\n0,1,A\n0,1,A\n0,0,A\n0,1,B\n0,0,B\n0,0,B\n0,0,B\n")
        .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_clotho"))
        .current_dir(&scratch.0)
        .args(["evaluate", "--csv", "fixture.csv", "--target", "label", "--features", "x",
               "--group", "wafer", "--model", "majority", "--folds", "2",
               "--out", "report"])
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(scratch.0.join("report/metrics.json")).unwrap()).unwrap();
    assert_eq!(report["pooled_row_weighted"]["roc_auc"].as_f64(), Some(0.25));
    assert_eq!(report["pooled_row_weighted"]["brier"].as_f64(), Some(0.4375));
    let folds = report["folds"].as_array().unwrap();
    assert_eq!(folds.len(), 2);
    for fold in folds {
        assert_eq!(fold["n_test"].as_u64(), Some(4));
        assert_eq!(fold["metrics"]["roc_auc"].as_f64(), Some(0.5));
    }
}
