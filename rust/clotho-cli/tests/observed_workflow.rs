use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-observed-{}-{}-{}",
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
    "prepare-observed", "--query", "query.csv", "--observed", "labels.csv",
    "--row-id", "id", "--saved-contract", "contract.json", "--out", "observed.json",
];
fn setup() -> Workspace {
    let w = Workspace::new();
    w.write("query.csv", "id,x\na,5\nb,6\n");
    w.write("labels.csv", "y,id\n3,b\n1,a\n");
    w.write("contract.json", CONTRACT);
    w
}

#[test]
fn shuffled_labels_join_exactly_and_roundtrip_into_saved_comparison() {
    let w = setup();
    let original = fs::read(w.0.join("query.csv")).unwrap();
    assert!(w.run(&ARGS).status.success());
    let observed: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("observed.json")).unwrap()).unwrap();
    assert_eq!(observed["row_ids"], serde_json::json!(["a", "b"]));
    assert_eq!(observed["values"], serde_json::json!([1., 3.]));
    assert_eq!(observed["schema_version"], "clotho.declared_regression_series.v2");
    assert_eq!(fs::read(w.0.join("query.csv")).unwrap(), original);
    w.write("train.csv", "x,y\n0,1\n1,3\n");
    assert!(w.run(&[
        "fit", "--csv", "train.csv", "--target", "y", "--features", "x",
        "--model", "mean", "--out", "model.json",
    ]).status.success());
    assert!(w.run(&[
        "predict", "--artifact", "model.json", "--csv", "query.csv",
        "--row-id", "id", "--saved-contract", "contract.json", "--out", "prediction.json",
    ]).status.success());
    let prediction: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("prediction.json")).unwrap()).unwrap();
    assert_eq!(observed["query_sha256"], prediction["query_sha256"]);
    let compare = [
        "compare-saved", "--query", "query.csv", "--observed", "observed.json",
        "--model", "prediction.json", "--baseline", "prediction.json", "--out", "report.json",
    ];
    assert!(w.run(&compare).status.success());
    fs::remove_file(w.0.join("report.json")).unwrap();
    w.write("query.csv", "id,x\r\na,5\r\nb,6\r\n");
    assert_eq!(w.run(&compare).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());
}

#[test]
fn missing_extra_duplicate_and_nonliteral_ids_fail_without_output() {
    for labels in [
        "id,y\na,1\n", "id,y\na,1\nb,3\nc,5\n", "id,y\na,1\na,3\n",
        "id,y\na,1\nc,3\n", "id,y\n a,1\nb,3\n", "id,y\n,1\nb,3\n",
        "id,y\n\"a\nb\",1\nb,3\n",
    ] {
        let w = setup();
        w.write("labels.csv", labels);
        assert_eq!(w.run(&ARGS).status.code(), Some(2), "{labels}");
        assert!(!w.0.join("observed.json").exists());
    }
    for query in ["id,x\na,1\na,2\n", "id,x\n,1\nb,2\n", "id,x,y\na,1,1\nb,2,3\n"] {
        let w = setup();
        w.write("query.csv", query);
        assert_eq!(w.run(&ARGS).status.code(), Some(2));
        assert!(!w.0.join("observed.json").exists());
    }
}

#[test]
fn label_schema_and_lossy_numeric_inputs_are_rejected() {
    for labels in [
        "id,wrong\na,1\nb,3\n", "wrong,y\na,1\nb,3\n",
        "id,y,extra\na,1,z\nb,3,z\n", "id,y,y\na,1,1\nb,3,3\n",
        "id,y\na,NaN\nb,3\n", "id,y\na,inf\nb,3\n",
        "id,y\na,1e309\nb,3\n", "id,y\na,1e-999\nb,3\n",
        "id,y\na,\nb,3\n", "id,y\na, 1\nb,3\n",
    ] {
        let w = setup();
        w.write("labels.csv", labels);
        assert_eq!(w.run(&ARGS).status.code(), Some(2), "{labels}");
        assert!(!w.0.join("observed.json").exists());
    }
}

#[test]
fn literal_ids_and_representable_values_survive_without_overwrite() {
    let w = setup();
    w.write("query.csv", "id,x\n001,5\n a ,6\n");
    w.write("labels.csv", "id,y\n a ,5e-324\n001,-0\n");
    assert!(w.run(&ARGS).status.success());
    let bytes = fs::read(w.0.join("observed.json")).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["row_ids"], serde_json::json!(["001", " a "]));
    assert!(value["values"][0].as_f64().unwrap().is_sign_negative());
    assert_eq!(value["values"][1].as_f64().unwrap(), f64::from_bits(1));
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert_eq!(fs::read(w.0.join("observed.json")).unwrap(), bytes);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(fs::metadata(w.0.join("observed.json")).unwrap().permissions().mode() & 0o777, 0o600);
    }
}

#[test]
fn malformed_contract_and_missing_options_never_publish() {
    for contract in [
        CONTRACT.replace("ohm", ""), CONTRACT.replace("\"y\"", "\"id\""),
        CONTRACT.replace("v1", "v9"), format!("{{\"target\":\"y\",{}", &CONTRACT[1..]),
        "[\"clotho.saved_prediction_contract.v1\",\"synthetic\",\"y\",\"ohm\",\"1 write\"]".into(),
    ] {
        let w = setup();
        w.write("contract.json", &contract);
        assert_eq!(w.run(&ARGS).status.code(), Some(2));
        assert!(!w.0.join("observed.json").exists());
    }
    let w = setup();
    assert_eq!(w.run(&ARGS[..9]).status.code(), Some(2));
    let mut unknown = ARGS.to_vec();
    unknown.extend(["--drop-missing", "true"]);
    assert_eq!(w.run(&unknown).status.code(), Some(2));
    assert!(!w.0.join("observed.json").exists());
}

#[test]
fn malformed_quotes_and_skipped_blank_records_fail_for_both_inputs() {
    for (query, labels) in [
        ("id,x\nax,7\n", "id,y\n\"a\"x,1\n"),
        ("id,x\n\"a\"x,7\n", "id,y\nax,1\n"),
        ("id,x\na,7\n", "id,y\na\",1\n"),
        ("id,x\na\",7\n", "id,y\na,1\n"),
        ("id,x\na,7\n", "id,y\na,\"1"),
        ("id,x\na,\"7", "id,y\na,1\n"),
        ("id,x\na,7\n\n", "id,y\na,1\n"),
        ("id,x\na,7\n", "id,y\n\na,1\n"),
        ("id,x\ra,7\r", "id,y\na,1\n"),
        ("id,x\na,7\n", "id,y\ra,1\r"),
        ("id,x\na,7\n", "id,y\n\"a\" ,1\n"),
    ] {
        let w = setup();
        w.write("query.csv", query);
        w.write("labels.csv", labels);
        assert_eq!(w.run(&ARGS).status.code(), Some(2), "{query:?} {labels:?}");
        assert!(!w.0.join("observed.json").exists());
    }
}

#[test]
fn valid_quotes_utf8_bom_and_line_endings_preserve_literal_ids() {
    let w = setup();
    let query = "\u{feff}id,x\r\n\"a,b\",7\r\n\"say\"\"hi\",8\r\n晶片,9";
    w.write("query.csv", query);
    w.write("labels.csv", "\u{feff}id,y\n晶片,3\n\"say\"\"hi\",2\n\"a,b\",1\n");
    assert!(w.run(&ARGS).status.success());
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("observed.json")).unwrap()).unwrap();
    assert_eq!(value["row_ids"], serde_json::json!(["a,b", "say\"hi", "晶片"]));
    assert_eq!(value["values"], serde_json::json!([1., 2., 3.]));
    assert_eq!(fs::read(w.0.join("query.csv")).unwrap(), query.as_bytes());
    for file in ["query.csv", "labels.csv"] {
        let w = setup();
        fs::write(w.0.join(file), b"id,x\na,\xff\n").unwrap();
        assert_eq!(w.run(&ARGS).status.code(), Some(2));
        assert!(!w.0.join("observed.json").exists());
    }
}
