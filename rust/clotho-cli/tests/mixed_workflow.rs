use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "clotho-mixed-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn write(&self, name: &str, contents: &str) {
        fs::write(self.0.join(name), contents).unwrap();
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_clotho"))
            .current_dir(&self.0)
            .args(args)
            .output()
            .unwrap()
    }
    fn fit(&self, model: &str, features: &str, extras: &[&str]) -> Output {
        let mut args = vec![
            "fit",
            "--csv",
            "data.csv",
            "--target",
            "y",
            "--features",
            features,
            "--model",
            model,
            "--out",
            "model.json",
        ];
        args.extend_from_slice(extras);
        self.run(&args)
    }
    fn evaluate(&self, model: &str, features: &str, extras: &[&str]) -> Output {
        let mut args = vec![
            "evaluate",
            "--csv",
            "data.csv",
            "--target",
            "y",
            "--features",
            features,
            "--model",
            model,
            "--group",
            "g",
            "--folds",
            "2",
            "--out",
            "run",
        ];
        args.extend_from_slice(extras);
        self.run(&args)
    }
    fn predict(&self, output: &str) -> Output {
        self.run(&[
            "predict",
            "--artifact",
            "model.json",
            "--csv",
            "query.csv",
            "--out",
            output,
        ])
    }
    fn json(&self, name: &str) -> serde_json::Value {
        serde_json::from_slice(&fs::read(self.0.join(name)).unwrap()).unwrap()
    }
    fn predictions(&self, name: &str, column: usize) -> Vec<f64> {
        let mut reader = csv::Reader::from_path(self.0.join(name)).unwrap();
        reader
            .records()
            .map(|row| row.unwrap()[column].parse().unwrap())
            .collect()
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn success(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn failure(output: Output, message: &str) {
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(message),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
const MIXED: &[&str] = &["--preprocess", "mixed", "--categorical", "material"];

#[test]
fn mixed_knn_reload_preserves_literal_categories_and_raw_feature_order() {
    let w = Workspace::new();
    w.write(
        "data.csv",
        "unused,material,y,x\nprivate,NA,10,0\nprivate,null,20,10\nprivate,,30,\nprivate, ,40,20\n",
    );
    success(w.fit(
        "knn",
        "x,material",
        &[
            "--preprocess",
            "mixed",
            "--categorical",
            "material",
            "--k",
            "1",
        ],
    ));
    let artifact = w.json("model.json");
    assert_eq!(artifact["schema_version"], "clotho.numeric.v3");
    assert_eq!(artifact["features"], serde_json::json!(["x", "material"]));
    assert_eq!(
        artifact["categorical_features"],
        serde_json::json!(["material"])
    );
    assert_eq!(artifact["training_x"], serde_json::json!([]));
    assert_eq!(
        artifact["training_rows"],
        serde_json::json!([["0", "NA"], ["10", "null"], ["", ""], ["20", " "]])
    );
    let before = fs::read(w.0.join("model.json")).unwrap();
    assert!(!String::from_utf8_lossy(&before).contains("private"));
    w.write(
        "query.csv",
        "material,extra,x\nNA,ignore,0\nnull,ignore,10\n,ignore,\n ,ignore,20\n",
    );
    success(w.predict("p.csv"));
    assert_eq!(w.predictions("p.csv", 1), vec![10.0, 20.0, 30.0, 40.0]);
    success(w.predict("p2.csv"));
    assert_eq!(
        fs::read(w.0.join("p.csv")).unwrap(),
        fs::read(w.0.join("p2.csv")).unwrap()
    );
    assert_eq!(before, fs::read(w.0.join("model.json")).unwrap());
    failure(w.fit("knn", "x,material", MIXED), "Output already exists");
    assert_eq!(before, fs::read(w.0.join("model.json")).unwrap());
}

#[test]
fn numeric_missing_queries_use_only_training_median() {
    let w = Workspace::new();
    w.write("data.csv", "x,y\n0,0\n2,2\n4,4\n");
    success(w.fit("ridge", "x", &["--preprocess", "mixed", "--alpha", "0"]));
    w.write(
        "query.csv",
        "x,extra\n,ignored\n100,ignored\n1000,ignored\n",
    );
    success(w.predict("p.csv"));
    let p = w.predictions("p.csv", 1);
    for (actual, expected) in p.iter().zip([2.0, 100.0, 1000.0]) {
        assert!((actual - expected).abs() < 1e-9);
    }
    w.write("query.csv", "x,extra\n,ignored\n");
    success(w.predict("single.csv"));
    assert_eq!(w.predictions("single.csv", 1)[0], p[0]);
}

#[test]
fn missing_and_unknown_categories_are_distinct_and_queries_do_not_grow_vocabulary() {
    let w = Workspace::new();
    w.write("data.csv", "material,y\n,10\nA,20\n");
    success(w.fit(
        "ridge",
        "material",
        &[
            "--preprocess",
            "mixed",
            "--categorical",
            "material",
            "--alpha",
            "1",
        ],
    ));
    let mut query = "material,extra\n,ignored\nA,ignored\n".to_owned();
    for i in 0..80 {
        query.push_str(&format!("unknown{i},ignored\n"));
    }
    w.write("query.csv", &query);
    success(w.predict("p.csv"));
    let p = w.predictions("p.csv", 1);
    assert_eq!(p.len(), 82);
    assert!(p[0] < p[2] && p[2] < p[1]);
    for unknown in &p[2..] {
        assert!((unknown - 15.0).abs() < 1e-12);
    }
    w.write("query.csv", "material,extra\nunknown0,ignored\n");
    success(w.predict("single.csv"));
    assert_eq!(w.predictions("single.csv", 1)[0], p[2]);
}

#[test]
fn all_missing_training_columns_keep_zero_numeric_and_missing_category_fallbacks() {
    let w = Workspace::new();
    w.write("data.csv", "x,material,y\n,,3\n,,5\n");
    success(w.fit("ridge", "x,material", MIXED));
    w.write("query.csv", "material,x\n,\nnew,100\n");
    success(w.predict("p.csv"));
    assert!(w
        .predictions("p.csv", 1)
        .iter()
        .all(|v| (v - 4.0).abs() < 1e-12));
}

#[test]
fn grouped_evaluation_learns_medians_inside_each_fold() {
    let w = Workspace::new();
    w.write("data.csv", "g,x,y\na,,99\na,1000,99\nb,0,0\nc,10,10\n");
    success(w.evaluate("ridge", "x", &["--preprocess", "mixed"]));
    let p = w.predictions("run/predictions.csv", 3);
    assert!((p[0] - 5.0).abs() < 1e-12); // Whole-data median would be 10.
    let metrics = w.json("run/metrics.json");
    assert_eq!(metrics["preprocess"], "mixed");
    assert_eq!(metrics["categorical_features"], serde_json::json!([]));
    assert_eq!(metrics["folds"][0]["encoded_feature_count"], 1);
    assert!(w.0.join("run/COMPLETE").exists());
}

#[test]
fn grouped_vocabularies_fit_after_splitting_even_when_global_vocabulary_exceeds_cap() {
    let w = Workspace::new();
    let mut data = "g,material,y\n".to_owned();
    for i in 0..80 {
        data.push_str(&format!(
            "{},{},{}\n",
            if i < 40 { "a" } else { "b" },
            format_args!("secret{i}"),
            i
        ));
    }
    w.write("data.csv", &data);
    success(w.evaluate(
        "knn",
        "material",
        &[
            "--preprocess",
            "mixed",
            "--categorical",
            "material",
            "--k",
            "1",
        ],
    ));
    let metrics = w.json("run/metrics.json");
    assert_eq!(
        metrics["categorical_features"],
        serde_json::json!(["material"])
    );
    assert_eq!(metrics["folds"][0]["encoded_feature_count"], 41);
    assert_eq!(metrics["folds"][1]["encoded_feature_count"], 41);
    assert!(!fs::read_to_string(w.0.join("run/metrics.json"))
        .unwrap()
        .contains("secret"));
    let p = w.predictions("run/predictions.csv", 3);
    assert!(p[..40].iter().all(|&v| v == 40.0));
    assert!(p[40..].iter().all(|&v| v == 0.0));
}

#[test]
fn mixed_logistic_fit_reload_and_grouped_evaluation() {
    let w = Workspace::new();
    w.write("data.csv", "g,x,material,y\na,-2,low,0\na,2,high,1\nb,,low,0\nb,1,high,1\nc,-3,,0\nc,3,high,1\nd,-4,low,0\nd,,high,1\n");
    success(w.fit("logistic", "x,material", MIXED));
    w.write("query.csv", "material,x\nlow,-2\nhigh,2\nnew,\n");
    success(w.predict("p.csv"));
    let probabilities = w.predictions("p.csv", 1);
    assert!(probabilities[0] < 0.5 && probabilities[1] > 0.5);
    assert!(probabilities.iter().all(|p| (0.0..=1.0).contains(p)));
    success(w.evaluate("logistic", "x,material", MIXED));
    let metrics = w.json("run/metrics.json");
    assert_eq!(
        metrics["schema_version"],
        "clotho.grouped_binary_evaluation.v1"
    );
    assert_eq!(metrics["preprocess"], "mixed");
    assert_eq!(metrics["group"], "g");
    assert_eq!(
        metrics["categorical_features"],
        serde_json::json!(["material"])
    );
    assert!(metrics["folds"]
        .as_array()
        .unwrap()
        .iter()
        .all(|f| f["encoded_feature_count"] == 4));
    assert!(w.0.join("run/COMPLETE").exists());
}

#[test]
fn numeric_v1_regression_and_v2_binary_artifacts_still_load() {
    let w = Workspace::new();
    w.write("query.csv", "x\n0\n1\n");
    let base = serde_json::json!({
        "schema_version":"clotho.numeric.v1", "model":"knn", "features":["x"],
        "target":"y", "k":1, "alpha":1.0, "training_x":[[0.0],[1.0]], "training_y":[0.0,1.0]
    });
    for (schema, model) in [
        ("clotho.numeric.v1", "knn"),
        ("clotho.numeric.v2", "ridge"),
        ("clotho.numeric.v2", "logistic"),
        ("clotho.numeric.v2", "majority"),
    ] {
        let mut artifact = base.clone();
        artifact["schema_version"] = schema.into();
        artifact["model"] = model.into();
        w.write("model.json", &artifact.to_string());
        success(w.predict(&format!("{schema}-{model}.csv")));
    }
    let mut invalid = base;
    invalid["model"] = "logistic".into();
    w.write("model.json", &invalid.to_string());
    failure(w.predict("invalid.csv"), "Unsupported model schema");
    assert!(!w.0.join("invalid.csv").exists());
}

#[test]
fn rejects_invalid_mixed_options_and_feature_leakage_without_outputs() {
    let w = Workspace::new();
    w.write(
        "data.csv",
        "g,x,material,y\na,0,A,0\na,1,B,1\nb,2,A,0\nb,3,B,1\n",
    );
    let cases: &[&[&str]] = &[
        &["--categorical", "material"],
        &["--categorical", ""],
        &["--preprocess", "auto"],
        &["--preprocess", "numeric"],
        &["--preprocess", "mixed", "--categorical", ""],
        &[
            "--preprocess",
            "mixed",
            "--categorical",
            "material,material",
        ],
        &["--preprocess", "mixed", "--categorical", "material,missing"],
        &["--preprocess", "mixed", "--categorical", "material,"],
        &["--preprocess", "mixed", "--categorical", "y"],
        &["--preprocess", "mixed", "--categorical", "g"],
    ];
    for extras in cases {
        assert!(!w.fit("mean", "x,material", extras).status.success());
        assert!(!w.0.join("model.json").exists());
        assert!(!w.evaluate("mean", "x,material", extras).status.success());
        assert!(!w.0.join("run").exists());
    }
    for features in ["x,y", "x,g", "material,material"] {
        assert!(!w.evaluate("mean", features, MIXED).status.success());
        assert!(!w.0.join("run").exists());
    }
}

#[test]
fn only_empty_cells_are_missing_and_nonfinite_numeric_is_rejected() {
    let w = Workspace::new();
    for cell in ["NA", "null", " ", "NaN", "inf", "-inf", "1e999"] {
        w.write(
            "data.csv",
            &format!("g,x,material,y\na,{cell},A,0\na,1,B,1\nb,2,A,0\nb,3,B,1\n"),
        );
        assert!(!w.fit("mean", "x,material", MIXED).status.success());
        assert!(!w.0.join("model.json").exists());
        assert!(!w.evaluate("mean", "x,material", MIXED).status.success());
        assert!(!w.0.join("run").exists());
    }
    w.write("data.csv", "x,y\n,1\n1,2\n");
    failure(w.fit("mean", "x", &[]), "without missing values");
    assert!(!w.0.join("model.json").exists());
    w.write("data.csv", "material,y\nA\u{0}SECRET,1\nB,2\n");
    let output = w.fit("mean", "material", MIXED);
    assert!(!output.status.success());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("SECRET"));
    assert!(!w.0.join("model.json").exists());
}

#[test]
fn invalid_query_schema_numeric_values_and_options_leave_no_output() {
    let w = Workspace::new();
    w.write("data.csv", "x,material,y\n0,A,1\n1,B,2\n");
    success(w.fit("mean", "x,material", MIXED));
    for query in [
        "x\n1\n",
        "x,material\nNaN,A\n",
        "x,material\nnull,A\n",
        "x,material,material\n1,A,A\n",
        "x,material\n0,bad\u{0}category\n",
    ] {
        w.write("query.csv", query);
        assert!(!w.predict("invalid.csv").status.success());
        assert!(!w.0.join("invalid.csv").exists());
    }
    w.write("query.csv", "x,material\n0,A\n");
    failure(
        w.run(&[
            "predict",
            "--csv",
            "query.csv",
            "--artifact",
            "model.json",
            "--out",
            "invalid.csv",
            "--preprocess",
            "mixed",
        ]),
        "Unknown option",
    );
    assert!(!w.0.join("invalid.csv").exists());
}

#[test]
fn category_and_encoded_feature_caps_fail_before_publishing() {
    let w = Workspace::new();
    let mut data = "material,y\n".to_owned();
    for i in 0..64 {
        data.push_str(&format!("category{i},{i}\n"));
    }
    w.write("data.csv", &data);
    failure(w.fit("mean", "material", MIXED), "63 observed strings");
    assert!(!w.0.join("model.json").exists());
    let mut data = "g,a,b,y\n".to_owned();
    for i in 0..100 {
        data.push_str(&format!(
            "{},a{},b{},{i}\n",
            if i < 50 { "first" } else { "second" },
            i % 50,
            i % 50
        ));
    }
    w.write("data.csv", &data);
    let extras = &["--preprocess", "mixed", "--categorical", "a,b"];
    failure(w.fit("mean", "a,b", extras), "100 encoded features");
    assert!(!w.0.join("model.json").exists());
    failure(w.evaluate("mean", "a,b", extras), "100 encoded features");
    assert!(!w.0.join("run").exists());
}

#[test]
fn mixed_artifacts_reject_ambiguous_legacy_roles_and_malformed_rows() {
    let w = Workspace::new();
    w.write("data.csv", "x,material,y\n0,A,1\n1,B,2\n");
    success(w.fit("mean", "x,material", MIXED));
    let good = w.json("model.json");
    w.write("query.csv", "x,material\n0,A\n");
    for change in 0..8 {
        let mut artifact = good.clone();
        match change {
            0 => artifact["schema_version"] = "clotho.numeric.v2".into(),
            1 => artifact["training_x"] = serde_json::json!([[0.0, 1.0]]),
            2 => artifact["categorical_features"] = serde_json::json!(["material", "material"]),
            3 => artifact["categorical_features"] = serde_json::json!(["unknown"]),
            4 => artifact["training_rows"] = serde_json::json!([["0", "A"], ["1"]]),
            5 => artifact["training_rows"] = serde_json::json!([]),
            6 => {
                artifact.as_object_mut().unwrap().remove("training_rows");
            }
            7 => artifact["training_y"] = serde_json::json!([1.0]),
            _ => unreachable!(),
        }
        w.write("model.json", &artifact.to_string());
        assert!(!w.predict("invalid.csv").status.success());
        assert!(!w.0.join("invalid.csv").exists());
    }
}

#[cfg(unix)]
#[test]
fn mixed_outputs_remain_private_and_existing_paths_are_preflighted() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let w = Workspace::new();
    w.write(
        "data.csv",
        "g,x,material,y\na,0,A,1\na,,A,2\nb,1,B,3\nb,,B,4\n",
    );
    symlink("absent.json", w.0.join("model.json")).unwrap();
    failure(w.fit("mean", "x,material", MIXED), "Output already exists");
    assert!(!w.0.join("absent.json").exists());
    fs::remove_file(w.0.join("model.json")).unwrap();
    success(w.fit("mean", "x,material", MIXED));
    success(w.evaluate("mean", "x,material", MIXED));
    for name in [
        "model.json",
        "run/predictions.csv",
        "run/metrics.json",
        "run/COMPLETE",
    ] {
        assert_eq!(
            fs::metadata(w.0.join(name)).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    assert_eq!(
        fs::metadata(w.0.join("run")).unwrap().permissions().mode() & 0o777,
        0o700
    );
    w.write("data.csv", "malformed input");
    failure(
        w.evaluate("mean", "x,material", MIXED),
        "Output already exists",
    );
    assert!(w.0.join("run/COMPLETE").exists());
}

#[test]
fn artifact_json_parse_errors_do_not_echo_private_values_or_field_names() {
    let w = Workspace::new();
    w.write("query.csv", "x\n0\n");
    let good = serde_json::json!({
        "schema_version":"clotho.numeric.v3", "model":"mean", "features":["x"],
        "target":"y", "k":1, "alpha":1.0, "training_x":[[0.0]], "training_y":[1.0]
    });
    for change in 0..5 {
        let mut artifact = good.clone();
        match change {
            0 => artifact["training_x"] = serde_json::json!([["PRIVATE_SENTINEL_VALUE"]]),
            1 => artifact["training_y"] = serde_json::json!(["PRIVATE_SENTINEL_VALUE"]),
            2 => artifact["model"] = "PRIVATE_SENTINEL_VALUE".into(),
            3 => artifact["PRIVATE_SENTINEL_VALUE"] = true.into(),
            4 => artifact["training_rows"] = serde_json::json!([[{"PRIVATE_SENTINEL_VALUE":1}]]),
            _ => unreachable!(),
        }
        w.write("model.json", &artifact.to_string());
        let output = w.predict("invalid.csv");
        assert!(!String::from_utf8_lossy(&output.stderr).contains("PRIVATE_SENTINEL_VALUE"));
        failure(output, "Invalid model JSON at line");
        assert!(!w.0.join("invalid.csv").exists());
    }
}
