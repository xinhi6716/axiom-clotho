use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-predict-saved-{}-{}-{}",
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

const CONTRACT: &str = r#"{"schema_version":"clotho.saved_prediction_contract.v1","cohort":"synthetic","target":"y","unit":"ohm","horizon":"1 write"}"#;
const ARGS: [&str; 11] = [
    "predict", "--artifact", "model.json", "--csv", "query.csv", "--out", "saved.json",
    "--row-id", "id", "--saved-contract", "contract.json",
];

fn setup() -> Workspace {
    let w = Workspace::new();
    w.write("train.csv", "x,y\n0,1\n1,3\n");
    w.write("query.csv", "id,x\na,5\nb,6\n");
    w.write("contract.json", CONTRACT);
    let output = w.run(&[
        "fit", "--csv", "train.csv", "--target", "y", "--features", "x",
        "--model", "mean", "--out", "model.json",
    ]);
    assert!(output.status.success(), "{:?}", output);
    w
}

#[test]
fn fit_predict_export_and_compare_saved_roundtrip() {
    let w = setup();
    assert!(w.run(&ARGS).status.success());
    let saved: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("saved.json")).unwrap()).unwrap();
    assert_eq!(saved["schema_version"], "clotho.declared_regression_series.v2");
    assert_eq!(saved["row_ids"], serde_json::json!(["a", "b"]));
    assert_eq!(saved["values"], serde_json::json!([2., 2.]));
    assert_eq!(saved.as_object().unwrap().len(), 8);
    let mut observed = saved.clone();
    observed["values"] = serde_json::json!([1., 3.]);
    w.write("observed.json", &observed.to_string());
    w.write("baseline-train.csv", "x,y\n0,0\n1,0\n");
    assert!(w.run(&[
        "fit", "--csv", "baseline-train.csv", "--target", "y", "--features", "x",
        "--model", "mean", "--out", "baseline-model.json",
    ]).status.success());
    let mut args = ARGS;
    args[2] = "baseline-model.json";
    args[6] = "baseline.json";
    assert!(w.run(&args).status.success());
    let compare = [
        "compare-saved", "--observed", "observed.json", "--model", "saved.json",
        "--baseline", "baseline.json", "--query", "query.csv", "--out", "report.json",
    ];
    assert!(w.run(&compare).status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
    assert!((report["mse_skill"].as_f64().unwrap() - 0.8).abs() < 1e-14);
    assert_eq!(report["query_bytes_verified"], true);
    assert_eq!(report["provenance_verified"], false);
    fs::remove_file(w.0.join("report.json")).unwrap();
    w.write("query.csv", "id,x\r\na,5\r\nb,6\r\n");
    assert_eq!(w.run(&compare).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());
}

#[test]
fn invalid_ids_contracts_and_partial_options_leave_no_output() {
    for query in [
        "id,x\na,5\na,6\n", "id,x\n,5\nb,6\n", "id,x\n\"a\nb\",5\nb,6\n",
        "other,x\na,5\nb,6\n",
    ] {
        let w = setup();
        w.write("query.csv", query);
        assert_eq!(w.run(&ARGS).status.code(), Some(2));
        assert!(!w.0.join("saved.json").exists());
    }
    for contract in [
        CONTRACT.replace("ohm", ""), CONTRACT.replace("\"y\"", "\"other\""),
        CONTRACT.replace("v1", "v9"), format!("{{\"target\":\"y\",{}", &CONTRACT[1..]),
        "[\"clotho.saved_prediction_contract.v1\",\"synthetic\",\"y\",\"ohm\",\"1 write\"]".into(),
    ] {
        let w = setup();
        w.write("contract.json", &contract);
        assert_eq!(w.run(&ARGS).status.code(), Some(2));
        assert!(!w.0.join("saved.json").exists());
    }
    let w = setup();
    assert_eq!(w.run(&ARGS[..9]).status.code(), Some(2));
    let mut missing_id = ARGS[..7].to_vec();
    missing_id.extend(["--saved-contract", "contract.json"]);
    assert_eq!(w.run(&missing_id).status.code(), Some(2));
    assert!(!w.0.join("saved.json").exists());
}

#[test]
fn export_preserves_order_and_literals_but_never_overwrites() {
    let w = setup();
    w.write("query.csv", "id,x,y\n b ,6,999\n001,5,-999\n");
    assert!(w.run(&ARGS).status.success());
    let bytes = fs::read(w.0.join("saved.json")).unwrap();
    let saved: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(saved["row_ids"], serde_json::json!([" b ", "001"]));
    assert_eq!(saved["values"], serde_json::json!([2., 2.]));
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert_eq!(fs::read(w.0.join("saved.json")).unwrap(), bytes);
    assert!(w.run(&[
        "predict", "--artifact", "model.json", "--csv", "query.csv", "--out", "legacy.csv",
    ]).status.success());
    assert_eq!(fs::read_to_string(w.0.join("legacy.csv")).unwrap(), "row_index,prediction\n0,2\n1,2\n");
}

#[test]
fn classification_threshold_and_feature_identity_are_not_silently_exported() {
    let w = setup();
    let mut args = ARGS.to_vec();
    args.extend(["--threshold", "0.5"]);
    assert_eq!(w.run(&args).status.code(), Some(2));
    let mut args = ARGS;
    args[8] = "x";
    assert_eq!(w.run(&args).status.code(), Some(2));
    args[8] = "y";
    assert_eq!(w.run(&args).status.code(), Some(2));
    let mut artifact: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("model.json")).unwrap()).unwrap();
    artifact["model"] = "majority".into();
    artifact["training_y"] = serde_json::json!([0., 1.]);
    w.write("model.json", &artifact.to_string());
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert!(!w.0.join("saved.json").exists());
}

#[test]
fn mixed_regression_exports_the_same_predictions_as_legacy_csv() {
    let w = Workspace::new();
    w.write("train.csv", "x,material,y\n1,A,1\n,B,3\n");
    w.write("query.csv", "id,x,material\na,,A\nb,4,B\n");
    w.write("contract.json", CONTRACT);
    assert!(w.run(&[
        "fit", "--csv", "train.csv", "--target", "y", "--features", "x,material",
        "--model", "mean", "--preprocess", "mixed", "--categorical", "material",
        "--out", "model.json",
    ]).status.success());
    assert!(w.run(&ARGS).status.success());
    let saved: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("saved.json")).unwrap()).unwrap();
    assert_eq!(saved["values"], serde_json::json!([2., 2.]));
    assert!(w.run(&[
        "predict", "--artifact", "model.json", "--csv", "query.csv", "--out", "legacy.csv",
    ]).status.success());
    assert_eq!(fs::read_to_string(w.0.join("legacy.csv")).unwrap(), "row_index,prediction\n0,2\n1,2\n");
}
