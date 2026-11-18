use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-numeric-underflow-{}-{}-{}",
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
    fn write(&self, name: &str, text: &str) {
        fs::write(self.0.join(name), text).unwrap();
    }
    fn run(&self, args: &[&str]) -> Output {
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



fn fit(w: &Workspace, extras: &[&str]) -> Output {
    let mut args = vec![
        "fit", "--csv", "train.csv", "--features", "x", "--target", "y",
        "--model", "mean", "--out", "model.json",
    ];
    args.extend_from_slice(extras);
    w.run(&args)
}

fn refused(w: &Workspace, output: Output, name: &str) {
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("Nonzero numeric cell underflows binary64"));
    assert!(!w.0.join(name).exists());
}

#[test]
fn selected_training_features_targets_and_binary_labels_refuse_underflow() {
    for token in ["1e-999", "-1E-999", "2e-324", "0.00001e-999"] {
        for (x, y) in [(token, "1"), ("1", token)] {
            let w = Workspace::new();
            w.write("train.csv", &format!("x,y\n{x},{y}\n2,3\n"));
            refused(&w, fit(&w, &[]), "model.json");
            refused(&w, fit(&w, &["--mean-inference", "true"]), "model.json");
            refused(&w, fit(&w, &["--large-training", "true"]), "model.json");
        }
    }
    let w = Workspace::new();
    w.write("train.csv", "x,y\n1,1e-999\n2,1\n");
    refused(&w, w.run(&[
        "fit", "--csv", "train.csv", "--features", "x", "--target", "y",
        "--model", "majority", "--out", "model.json",
    ]), "model.json");
}

#[test]
fn query_underflow_refused_for_legacy_inference_and_mixed_models() {
    for extras in [vec![], vec!["--mean-inference", "true"], vec!["--preprocess", "mixed"]] {
        let w = Workspace::new();
        w.write("train.csv", "x,y\n1,1\n2,3\n");
        assert!(fit(&w, &extras).status.success());
        w.write("query.csv", "x\n1e-999\n");
        let mut args = vec![
            "predict", "--artifact", "model.json", "--csv", "query.csv", "--out", "out.csv",
        ];
        if extras.contains(&"--mean-inference") {
            args.extend(["--mean-inference", "true"]);
        }
        refused(&w, w.run(&args), "out.csv");
    }
}

#[test]
fn mixed_numeric_cells_reject_but_categories_and_missing_keep_their_meaning() {
    let w = Workspace::new();
    w.write("train.csv", "x,y\n1e-999,1\n2,3\n");
    refused(&w, fit(&w, &["--preprocess", "mixed"]), "model.json");
    assert!(fit(&w, &["--preprocess", "mixed", "--categorical", "x"]).status.success());
    let missing = Workspace::new();
    missing.write("train.csv", "x,y\n,1\n2,3\n");
    assert!(fit(&missing, &["--preprocess", "mixed"]).status.success());
}

#[test]
fn true_zero_representable_subnormal_and_unused_cells_remain_valid() {
    for token in ["0", "-0", "+0.0", "0e-999", "-0.000E+999", "5e-324", "-5e-324", "3e-324"] {
        let w = Workspace::new();
        w.write("train.csv", &format!("x,y,unused\n{token},1,1e-999\n{token},1,1e-999\n"));
        let result = fit(&w, &[]);
        assert!(result.status.success(), "{token}: {result:?}");
        let value: serde_json::Value = serde_json::from_slice(
            &fs::read(w.0.join("model.json")).unwrap(),
        ).unwrap();
        assert_eq!(value["training_x"][0][0].as_f64().unwrap().to_bits(),
            token.parse::<f64>().unwrap().to_bits());
    }
}

#[test]
fn evaluation_and_selection_refuse_before_creating_output_directory() {
    let w = Workspace::new();
    w.write("train.csv", "x,y,g\n1,1,a\n2,2,b\n3,1e-999,c\n4,4,d\n");
    refused(&w, w.run(&[
        "evaluate", "--csv", "train.csv", "--features", "x", "--target", "y",
        "--group", "g", "--model", "mean", "--folds", "2", "--out", "evaluation",
    ]), "evaluation");
    refused(&w, w.run(&[
        "compare", "--csv", "train.csv", "--features", "x", "--target", "y",
        "--group", "g", "--candidates", "mean", "--outer-folds", "2", "--inner-folds", "2",
        "--out", "comparison",
    ]), "comparison");
}
