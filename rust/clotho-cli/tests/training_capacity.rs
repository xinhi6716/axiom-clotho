use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-training-capacity-{}-{}-{}",
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

fn data(n: usize) -> String {
    let mut csv = String::from("x,y,group\n");
    for i in 0..n {
        let x = i % 2;
        csv.push_str(&format!("{x},{},g{}\n", 3 + 2 * x, i % 4));
    }
    csv
}
#[test]
fn narrow_numeric_training_reloads_but_large_queries_still_fail() {
    for kind in ["mean", "ridge"] {
        let w = Workspace::new();
        w.write("train.csv", &data(32_768));
        let fit = ["fit", "--csv", "train.csv", "--target", "y", "--features", "x", "--model", kind, "--alpha", "0", "--out", "model.json", "--large-training", "true"];
        let result = w.run(&fit);
        assert!(result.status.success(), "{:?}", result);
        w.write("query.csv", "x\n0\n1\n");
        assert!(w.run(&["predict", "--csv", "query.csv", "--artifact", "model.json", "--out", "pred.csv"]).status.success());
        let text = fs::read_to_string(w.0.join("pred.csv")).unwrap();
        let values: Vec<f64> = text.lines().skip(1).map(|line| line.split(',').nth(1).unwrap().parse().unwrap()).collect();
        let expected = if kind == "mean" { [4., 4.] } else { [3., 5.] };
        for (a, b) in values.iter().zip(expected) { assert!((a - b).abs() < 1e-10); }
        w.write("large-query.csv", &data(20_001));
        assert_eq!(w.run(&["predict", "--csv", "large-query.csv", "--artifact", "model.json", "--out", "too-large.csv"]).status.code(), Some(2));
        assert!(!w.0.join("too-large.csv").exists());
        w.write("train-too-large.csv", &data(32_769));
        let mut excessive = fit;
        excessive[2] = "train-too-large.csv";
        excessive[12] = "too-large.json";
        assert_eq!(w.run(&excessive).status.code(), Some(2));
        assert!(!w.0.join("too-large.json").exists());
    }
}
#[test]
fn mixed_knn_classification_and_evaluation_keep_twenty_thousand_limit() {
    let w = Workspace::new();
    w.write("train.csv", &data(20_001));
    for (command, kind, extra) in [
        ("fit", "mean", vec!["--preprocess", "mixed"]),
        ("fit", "ridge", vec!["--preprocess", "mixed"]),
        ("fit", "knn", vec![]),
        ("fit", "logistic", vec![]),
        ("fit", "majority", vec![]),
        ("evaluate", "mean", vec!["--group", "group"]),
    ] {
        let mut args = vec![command, "--csv", "train.csv", "--target", "y", "--features", "x", "--model", kind, "--out", "rejected"];
        args.extend(extra);
        let output = w.run(&args);
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("20,000 row limit"));
        assert!(!w.0.join("rejected").exists());
    }
    assert_eq!(w.run(&["compare", "--csv", "train.csv", "--out", "rejected"]).status.code(), Some(2));
    assert!(!w.0.join("rejected").exists());
}

#[test]
fn extended_cell_and_ridge_work_rejections_do_not_create_artifacts() {
    for (kind, width) in [("mean", 50), ("ridge", 23)] {
        let w = Workspace::new();
        let features = (0..width).map(|i| format!("x{i}")).collect::<Vec<_>>().join(",");
        let mut csv = format!("{features},y\n");
        let row = format!("{}0\n", "0,".repeat(width));
        for _ in 0..20_001 { csv.push_str(&row); }
        w.write("train.csv", &csv);
        let result = w.run(&["fit", "--csv", "train.csv", "--target", "y", "--features", &features, "--model", kind, "--out", "model.json", "--large-training", "true"]);
        assert_eq!(result.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&result.stderr).contains("budgets"));
        assert!(!w.0.join("model.json").exists());
    }
}

#[test]
fn opt_in_bounds_raw_unused_fields_and_keeps_legacy_defaults() {
    let w = Workspace::new();
    w.write("train.csv", &data(20_001));
    let fit = ["fit", "--csv", "train.csv", "--target", "y", "--features", "x", "--model", "mean", "--out", "model.json"];
    assert_eq!(w.run(&fit).status.code(), Some(2));
    let mut opted = fit.to_vec();
    opted.extend(["--large-training", "true"]);
    let header = format!("x,y,{}\n", (0..128).map(|i| format!("extra{i}")).collect::<Vec<_>>().join(","));
    let row = format!("0,3,{}\n", vec!["0";128].join(","));
    w.write("train.csv", &(header + &row));
    let reject = w.run(&opted);
    assert_eq!(reject.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&reject.stderr).contains("128 raw fields"));
    assert!(!w.0.join("model.json").exists());
    assert!(w.run(&fit).status.success());
    fs::remove_file(w.0.join("model.json")).unwrap();
    let header = format!("x,y,{}\n", (0..49).map(|i| format!("extra{i}")).collect::<Vec<_>>().join(","));
    let row = format!("0,3,{}\n", vec!["0";49].join(","));
    w.write("train.csv", &(header + &row.repeat(20_001)));
    let reject = w.run(&opted);
    assert_eq!(reject.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&reject.stderr).contains("raw row/cell budgets"));
    assert!(!w.0.join("model.json").exists());
    fs::File::create(w.0.join("train.csv")).unwrap().set_len(16*1024*1024+1).unwrap();
    let reject = w.run(&opted);
    assert_eq!(reject.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&reject.stderr).contains("16 MiB"));
    assert!(!w.0.join("model.json").exists());
}

#[test]
fn opt_in_lexical_validation_and_model_scope_are_explicit() {
    let w = Workspace::new();
    let mut fit = vec!["fit", "--csv", "train.csv", "--target", "y", "--features", "x", "--model", "mean", "--out", "model.json", "--large-training", "true"];
    for csv in ["x,y\n\"0\"x,3\n", "x,y\n0,3\n\n", "x,y\n0,\"3", "x,y\r0,3\r"] {
        w.write("train.csv", csv);
        assert_eq!(w.run(&fit).status.code(), Some(2));
        assert!(!w.0.join("model.json").exists());
    }
    w.write("train.csv", "\u{feff}x,y,note\r\n0,3,\"a,b\"\r\n1,5,\"a\"\"b\"");
    for kind in ["knn", "majority", "logistic"] {
        fit[8] = kind;
        assert_eq!(w.run(&fit).status.code(), Some(2));
    }
    fit[8] = "mean";
    let mut mixed = fit.clone(); mixed.extend(["--preprocess", "mixed"]);
    assert_eq!(w.run(&mixed).status.code(), Some(2));
    fit[12] = "false";
    assert_eq!(w.run(&fit).status.code(), Some(2));
    fit[12] = "true";
    assert!(w.run(&fit).status.success());
}
