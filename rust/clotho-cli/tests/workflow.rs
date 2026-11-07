use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-cli-{}-{}-{}",
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
const DATA: &str = "wafer,x,response\na,0,1\na,1,3\nb,2,5\nb,3,7\nc,4,9\nc,5,11\n";
#[test]
fn knn_fit_predict_and_no_overwrite() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    w.write("q.csv", "x\n1\n4\n");
    let args = [
        "fit",
        "--csv",
        "data.csv",
        "--target",
        "response",
        "--features",
        "x",
        "--model",
        "knn",
        "--k",
        "1",
        "--out",
        "model.json",
    ];
    let first = w.run(&args);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let before = fs::read(w.0.join("model.json")).unwrap();
    assert!(!w.run(&args).status.success());
    assert_eq!(before, fs::read(w.0.join("model.json")).unwrap());
    let p = w.run(&[
        "predict",
        "--artifact",
        "model.json",
        "--csv",
        "q.csv",
        "--out",
        "pred.csv",
    ]);
    assert!(p.status.success(), "{}", String::from_utf8_lossy(&p.stderr));
    assert_eq!(
        fs::read_to_string(w.0.join("pred.csv")).unwrap(),
        "row_index,prediction\n0,3\n1,9\n"
    );
}
#[test]
fn grouped_ridge_evaluation_is_complete_and_reproducible() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    for out in ["run1", "run2"] {
        let r = w.run(&[
            "evaluate",
            "--csv",
            "data.csv",
            "--target",
            "response",
            "--features",
            "x",
            "--model",
            "ridge",
            "--alpha",
            "0",
            "--group",
            "wafer",
            "--folds",
            "3",
            "--out",
            out,
        ]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        assert!(w.0.join(out).join("COMPLETE").exists());
        let m: serde_json::Value =
            serde_json::from_slice(&fs::read(w.0.join(out).join("metrics.json")).unwrap()).unwrap();
        assert_eq!(m["n_rows"], 6);
        assert_eq!(m["n_groups"], 3);
        assert!(m["pooled_row_weighted"]["rmse"].as_f64().unwrap() < 1e-10);
    }
    assert_eq!(
        fs::read(w.0.join("run1/predictions.csv")).unwrap(),
        fs::read(w.0.join("run2/predictions.csv")).unwrap()
    );
}
#[test]
fn rejects_leakage_missing_nonfinite_duplicate_headers_and_small_training_folds() {
    let w = Workspace::new();
    for (data, features) in [
        (DATA, "response"),
        (DATA, "wafer"),
        ("wafer,x,response\na,,1\nb,1,3\nc,2,5\n", "x"),
        ("wafer,x,response\na,NaN,1\nb,1,3\nc,2,5\n", "x"),
        ("wafer,x,x,response\na,0,0,1\nb,1,1,3\nc,2,2,5\n", "x"),
    ] {
        w.write("bad.csv", data);
        let r = w.run(&[
            "evaluate",
            "--csv",
            "bad.csv",
            "--target",
            "response",
            "--features",
            features,
            "--model",
            "knn",
            "--k",
            "1",
            "--group",
            "wafer",
            "--folds",
            "3",
            "--out",
            "bad_run",
        ]);
        assert!(!r.status.success());
        assert!(!w.0.join("bad_run").exists());
    }
    w.write("data.csv", DATA);
    let r = w.run(&[
        "evaluate",
        "--csv",
        "data.csv",
        "--target",
        "response",
        "--features",
        "x",
        "--model",
        "knn",
        "--k",
        "5",
        "--group",
        "wafer",
        "--folds",
        "3",
        "--out",
        "too_many_neighbors",
    ]);
    assert!(!r.status.success());
    assert!(!w.0.join("too_many_neighbors").exists());
}
#[test]
fn validates_artifact_schema_and_feature_shape() {
    let w = Workspace::new();
    w.write("q.csv", "x\n1\n");
    for artifact in [
        r#"{"schema_version":"wrong","model":"knn","features":["x"],"target":"y","k":1,"alpha":1,"training_x":[[1]],"training_y":[1]}"#,
        r#"{"schema_version":"clotho.numeric.v1","model":"knn","features":["x"],"target":"y","k":1,"alpha":1,"training_x":[[1,2]],"training_y":[1]}"#,
        r#"{"schema_version":"clotho.numeric.v1","model":"knn","features":["x"],"target":"x","k":1,"alpha":1,"training_x":[[1]],"training_y":[1]}"#,
    ] {
        w.write("bad.json", artifact);
        assert!(!w
            .run(&[
                "predict",
                "--artifact",
                "bad.json",
                "--csv",
                "q.csv",
                "--out",
                "bad_pred.csv"
            ])
            .status
            .success());
        assert!(!w.0.join("bad_pred.csv").exists());
    }
}
#[test]
fn mean_baseline_and_strict_options() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    w.write("q.csv", "x\n100\n");
    assert!(w
        .run(&[
            "fit",
            "--csv",
            "data.csv",
            "--target",
            "response",
            "--features",
            "x",
            "--model",
            "mean",
            "--out",
            "mean.json"
        ])
        .status
        .success());
    assert!(w
        .run(&[
            "predict",
            "--artifact",
            "mean.json",
            "--csv",
            "q.csv",
            "--out",
            "mean.csv"
        ])
        .status
        .success());
    let csv = fs::read_to_string(w.0.join("mean.csv")).unwrap();
    let prediction: f64 = csv
        .lines()
        .nth(1)
        .unwrap()
        .split(',')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    assert!((prediction - 6.0).abs() < 1e-12);
    assert!(!w
        .run(&["predict", "--csv", "q.csv", "--bogus", "value"])
        .status
        .success());
    assert!(!w
        .run(&["fit", "--csv", "data.csv", "--csv", "data.csv"])
        .status
        .success());
    assert!(w.run(&["--help"]).status.success());
    assert!(w.run(&["--version"]).status.success());
}

#[test]
fn evaluation_mean_uses_only_training_targets() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    let r = w.run(&[
        "evaluate",
        "--csv",
        "data.csv",
        "--target",
        "response",
        "--features",
        "x",
        "--model",
        "mean",
        "--group",
        "wafer",
        "--folds",
        "3",
        "--out",
        "run",
    ]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let csv = fs::read_to_string(w.0.join("run/predictions.csv")).unwrap();
    let predicted: Vec<f64> = csv
        .lines()
        .skip(1)
        .map(|r| r.split(',').nth(3).unwrap().parse().unwrap())
        .collect();
    for (actual, expected) in predicted.iter().zip([8.0, 8.0, 6.0, 6.0, 4.0, 4.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
}
#[test]
fn evaluation_knn_fits_feature_scales_inside_each_fold() {
    let w = Workspace::new();
    w.write(
        "data.csv",
        "wafer,x,z,response\na,1.95,600,1\na,10000,0,2\nb,1,1000,10\nc,2,0,20\n",
    );
    let r = w.run(&[
        "evaluate",
        "--csv",
        "data.csv",
        "--target",
        "response",
        "--features",
        "x,z",
        "--model",
        "knn",
        "--k",
        "1",
        "--group",
        "wafer",
        "--folds",
        "2",
        "--out",
        "run",
    ]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let csv = fs::read_to_string(w.0.join("run/predictions.csv")).unwrap();
    let first: f64 = csv
        .lines()
        .nth(1)
        .unwrap()
        .split(',')
        .nth(3)
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(first, 20.0); // Global-data scaling would select target 10 instead.
}
#[test]
fn rejects_excessive_folds_and_aligns_features_by_header() {
    let w = Workspace::new();
    w.write("data.csv", DATA);
    w.write("query.csv", "unused,x\nignore,4\n");
    assert!(!w
        .run(&[
            "evaluate",
            "--csv",
            "data.csv",
            "--target",
            "response",
            "--features",
            "x",
            "--model",
            "mean",
            "--group",
            "wafer",
            "--folds",
            "20000",
            "--out",
            "bad"
        ])
        .status
        .success());
    assert!(!w.0.join("bad").exists());
    assert!(w
        .run(&[
            "fit",
            "--csv",
            "data.csv",
            "--target",
            "response",
            "--features",
            "x",
            "--model",
            "knn",
            "--k",
            "1",
            "--out",
            "model.json"
        ])
        .status
        .success());
    assert!(w
        .run(&[
            "predict",
            "--csv",
            "query.csv",
            "--artifact",
            "model.json",
            "--out",
            "p.csv"
        ])
        .status
        .success());
    assert_eq!(
        fs::read_to_string(w.0.join("p.csv")).unwrap(),
        "row_index,prediction\n0,9\n"
    );
}
#[cfg(unix)]
#[test]
fn private_permissions_and_symlink_refusal() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let w = Workspace::new();
    w.write("data.csv", DATA);
    let fit = [
        "fit",
        "--csv",
        "data.csv",
        "--target",
        "response",
        "--features",
        "x",
        "--model",
        "mean",
        "--out",
        "model.json",
    ];
    symlink("absent.json", w.0.join("model.json")).unwrap();
    assert!(!w.run(&fit).status.success());
    assert!(!w.0.join("absent.json").exists());
    fs::remove_file(w.0.join("model.json")).unwrap();
    assert!(w.run(&fit).status.success());
    assert_eq!(
        fs::metadata(w.0.join("model.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert!(w
        .run(&[
            "evaluate",
            "--csv",
            "data.csv",
            "--target",
            "response",
            "--features",
            "x",
            "--model",
            "mean",
            "--group",
            "wafer",
            "--folds",
            "3",
            "--out",
            "run"
        ])
        .status
        .success());
    assert_eq!(
        fs::metadata(w.0.join("run")).unwrap().permissions().mode() & 0o777,
        0o700
    );
}

#[test]
fn macro_metrics_do_not_underflow_each_group_before_summing() {
    let w = Workspace::new();
    w.write("data.csv", "g,x,y\na,0,0\nb,1,4.9406564584124654e-324\n");
    let r = w.run(&[
        "evaluate",
        "--csv",
        "data.csv",
        "--target",
        "y",
        "--features",
        "x",
        "--model",
        "mean",
        "--group",
        "g",
        "--folds",
        "2",
        "--out",
        "run",
    ]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let m: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("run/metrics.json")).unwrap()).unwrap();
    assert_eq!(
        m["equal_group_weighted"]["mean_group_mae"]
            .as_f64()
            .unwrap(),
        f64::from_bits(1)
    );
    assert_eq!(
        m["equal_group_weighted"]["mean_group_rmse"]
            .as_f64()
            .unwrap(),
        f64::from_bits(1)
    );
}

#[test]
fn saved_model_preserves_binary64_targets_on_json_reload() {
    let w = Workspace::new();
    for (i, target) in [1.9432074656054558e-171_f64, 1.7976931348623157e308]
        .iter()
        .enumerate()
    {
        w.write("data.csv", &format!("x,y\n1,{target}\n"));
        w.write("q.csv", "x\n1\n");
        let model = format!("model{i}.json");
        let prediction = format!("p{i}.csv");
        let fit = w.run(&[
            "fit",
            "--csv",
            "data.csv",
            "--target",
            "y",
            "--features",
            "x",
            "--model",
            "knn",
            "--k",
            "1",
            "--out",
            &model,
        ]);
        assert!(
            fit.status.success(),
            "{}",
            String::from_utf8_lossy(&fit.stderr)
        );
        let p = w.run(&[
            "predict",
            "--artifact",
            &model,
            "--csv",
            "q.csv",
            "--out",
            &prediction,
        ]);
        assert!(p.status.success(), "{}", String::from_utf8_lossy(&p.stderr));
        let csv = fs::read_to_string(w.0.join(prediction)).unwrap();
        let result: f64 = csv
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(result.to_bits(), target.to_bits());
    }
}
