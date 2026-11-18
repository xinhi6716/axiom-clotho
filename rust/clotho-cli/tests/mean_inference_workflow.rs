use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-mean-inference-{}-{}-{}",
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


const MODEL: &str = r#"{"schema_version":"clotho.mean_inference.v1","model":"mean","features":["x"],"target":"y","mean":2.0}"#;
fn predict(w: &Workspace) -> Output {
    w.run(&["predict", "--mean-inference", "true", "--artifact", "m.json",
        "--csv", "q.csv", "--out", "out.csv"])
}

#[test]
fn fit_summary_roundtrips_without_training_file_and_legacy_is_unchanged() {
    for targets in ["1\n3", "5e-324\n5e-324", "1.7976931348623157e308\n1.7976931348623157e308", "-1e300\n1e300"] {
        let w = Workspace::new();
        let ys: Vec<_> = targets.split('\n').collect();
        w.write("t.csv", &format!("x,y\n0,{}\n1,{}\n", ys[0], ys[1]));
        w.write("q.csv", "x\n2\n3\n");
        let args = ["fit", "--csv", "t.csv", "--features", "x", "--target", "y", "--model", "mean", "--out"];
        let mut legacy = args.to_vec();
        legacy.push("legacy.json");
        assert!(w.run(&legacy).status.success());
        let mut compact = args.to_vec();
        compact.extend(["m.json", "--mean-inference", "true"]);
        assert!(w.run(&compact).status.success());
        fs::remove_file(w.0.join("t.csv")).unwrap();
        assert!(predict(&w).status.success());
        assert!(w.run(&["predict", "--artifact", "legacy.json", "--csv", "q.csv", "--out", "legacy.csv"]).status.success());
        assert_eq!(fs::read(w.0.join("out.csv")).unwrap(), fs::read(w.0.join("legacy.csv")).unwrap());
        let model: serde_json::Value = serde_json::from_slice(&fs::read(w.0.join("m.json")).unwrap()).unwrap();
        assert_eq!(model.as_object().unwrap().len(), 5);
        assert!(model.get("training_x").is_none());
        assert!(model.get("training_y").is_none());
        assert!(!w.run(&["predict", "--artifact", "m.json", "--csv", "q.csv", "--out", "wrong.csv"]).status.success());
        assert!(!w.0.join("wrong.csv").exists());
        assert!(!predict(&w).status.success()); // no replacement
    }
}

#[test]
fn hostile_summaries_and_query_errors_never_create_output() {
    let mut bad = vec![
        MODEL.replace("2.0", "1e-999"), MODEL.replace("2.0", "1e999"),
        MODEL.replace("2.0", "null"), MODEL.replace("2.0", "\"2\""),
        MODEL.replace("2.0", "2.0,\"mean\":3"),
        MODEL.replace("2.0}", "2.0,\"training_x\":[]}"),
        MODEL.replace("[\"x\"]", "[\"x\",\"x\"]"),
        MODEL.replace("[\"x\"]", "[\"y\"]"),
        MODEL.replace("[\"x\"]", "[]"),
        MODEL.replace("\"mean\",", "\"ridge\","),
        MODEL.replace("clotho.mean_inference.v1", "clotho.mean_inference.v2"),
        r#"["clotho.mean_inference.v1","mean",["x"],"y",2]"#.into(),
    ];
    bad.push(" ".repeat(65537));
    for m in bad {
        let w = Workspace::new();
        w.write("m.json", &m);
        w.write("q.csv", "x\n1\n");
        assert!(!predict(&w).status.success(), "accepted {m}");
        assert!(!w.0.join("out.csv").exists());
    }
    for q in ["z\n1\n".into(), "x\nNaN\n".into(), "x\n".into(), format!("x\n{}", "1\n".repeat(20001))] {
        let w = Workspace::new();
        w.write("m.json", MODEL);
        w.write("q.csv", &q);
        assert!(!predict(&w).status.success());
        assert!(!w.0.join("out.csv").exists());
    }
}

#[test]
fn strict_mean_only_and_bound_saved_export() {
    let w = Workspace::new();
    w.write("t.csv", "x,y\n0,1\n1,3\n");
    for extra in [vec!["--model", "ridge"], vec!["--model", "mean", "--preprocess", "mixed"], vec!["--model", "majority"]] {
        let mut args = vec!["fit", "--csv", "t.csv", "--features", "x", "--target", "y", "--out", "bad.json", "--mean-inference", "true"];
        args.extend(extra);
        assert!(!w.run(&args).status.success());
        assert!(!w.0.join("bad.json").exists());
    }
    w.write("m.json", MODEL);
    w.write("q.csv", "id,x\na,1\nb,2\n");
    w.write("contract.json", r#"{"schema_version":"clotho.saved_prediction_contract.v1","cohort":"synthetic","target":"y","unit":"u","horizon":"h"}"#);
    assert!(w.run(&["predict", "--mean-inference", "true", "--artifact", "m.json", "--csv", "q.csv", "--row-id", "id", "--saved-contract", "contract.json", "--out", "saved.json"]).status.success());
    let saved: serde_json::Value = serde_json::from_slice(&fs::read(w.0.join("saved.json")).unwrap()).unwrap();
    assert_eq!(saved["row_ids"], serde_json::json!(["a", "b"]));
    assert_eq!(saved["values"], serde_json::json!([2.0, 2.0]));
    assert_eq!(saved["schema_version"], "clotho.declared_regression_series.v2");
}
