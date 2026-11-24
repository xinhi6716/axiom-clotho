use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-pool-{}-{}-{}",
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


fn report(cohort: &str, digit: &str, n: usize, model: f64, baseline: f64) -> serde_json::Value {
    let skill = if baseline == 0.0 { None } else { Some(1. - (model / baseline).powi(2)) };
    serde_json::json!({
        "schema_version": "clotho.declared_regression_comparison.v2",
        "cohort": cohort, "target": "resistance", "unit": "ohm", "horizon": "1 write",
        "n_rows": n, "model_rmse": model, "baseline_rmse": baseline, "mse_skill": skill,
        "validation": "matching declared contracts, ordered unique row IDs and supplied query byte digest only",
        "provenance_verified": false, "query_bytes_verified": true, "fit_performed": false,
        "query_sha256": digit.repeat(64)
    })
}
fn setup(first: serde_json::Value, second: serde_json::Value) -> Workspace {
    let w = Workspace::new();
    w.write("a.json", &first.to_string());
    w.write("b.json", &second.to_string());
    w.write("manifest.json", r#"{"schema_version":"clotho.saved_report_manifest.v1","model_id":"ridge-fixed","baseline_id":"training-mean","reports":["a.json","b.json"]}"#);
    w
}
const ARGS: [&str; 5] = ["pool-saved", "--manifest", "manifest.json", "--out", "pooled.json"];
fn read(w: &Workspace) -> serde_json::Value {
    serde_json::from_slice(&fs::read(w.0.join("pooled.json")).unwrap()).unwrap()
}

#[test]
fn row_weighted_pool_is_not_mean_skill_and_keeps_denominator_zero_groups() {
    let w = setup(report("large", "a", 9, 0., 2.), report("small", "b", 1, 3., 0.));
    assert!(w.run(&ARGS).status.success());
    let value = read(&w);
    assert!((value["mse_skill"].as_f64().unwrap() - 0.75).abs() < 1e-14);
    assert!((value["model_rmse"].as_f64().unwrap() - 0.9_f64.sqrt()).abs() < 1e-14);
    assert_eq!(value["n_rows"], 10);
    assert_eq!(value["n_reports"], 2);
    assert_eq!(value["model_id"], "ridge-fixed");
    assert_eq!(value["baseline_id"], "training-mean");
    assert_eq!(value["identities_verified"], false);
    assert_eq!(value["membership_verified"], false);
    assert_eq!(value["sources"][1]["baseline_rmse"], 0.);
    for field in ["disjointness_verified", "provenance_verified", "query_bytes_verified", "roster_completeness_verified", "fit_performed"] {
        assert_eq!(value[field], false);
    }
    let original = fs::read(w.0.join("pooled.json")).unwrap();
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert_eq!(fs::read(w.0.join("pooled.json")).unwrap(), original);
}

#[test]
fn perfect_tied_negative_and_scaled_extremes_have_explicit_results() {
    for (model, baseline, expected) in [(0., 0., None), (2., 0., None), (2., 2., Some(0.)), (3., 2., Some(-1.25)), (1e200, 2e200, Some(0.75)), (1e-200, 2e-200, Some(0.75))] {
        let w = setup(report("a", "a", 2, model, baseline), report("b", "b", 3, model, baseline));
        assert!(w.run(&ARGS).status.success(), "{:?}", w.run(&ARGS));
        assert_eq!(read(&w)["mse_skill"].as_f64(), expected);
    }
}

#[test]
fn mismatches_duplicate_declarations_and_bad_scalars_never_publish() {
    for (field, value) in [
        ("cohort", serde_json::json!("a")), ("query_sha256", serde_json::json!("a".repeat(64))),
        ("target", serde_json::json!("other")), ("unit", serde_json::json!("kohm")),
        ("horizon", serde_json::json!("2 writes")), ("n_rows", serde_json::json!(0)),
        ("n_rows", serde_json::json!(20_001)), ("model_rmse", serde_json::json!(-1)),
        ("mse_skill", serde_json::json!(0.99)), ("query_bytes_verified", serde_json::json!(false)),
        ("provenance_verified", serde_json::json!(true)), ("query_sha256", serde_json::json!("A".repeat(64))),
        ("validation", serde_json::json!("trust me")), ("grouped_summary", serde_json::json!({})),
    ] {
        let mut b = report("b", "b", 1, 1., 2.);
        b[field] = value;
        let w = setup(report("a", "a", 1, 1., 2.), b);
        assert_eq!(w.run(&ARGS).status.code(), Some(2), "{field}");
        assert!(!w.0.join("pooled.json").exists());
    }
}

#[test]
fn malformed_or_incomplete_manifest_and_report_are_rejected() {
    for manifest in [
        r#"{"schema_version":"clotho.saved_report_manifest.v1","model_id":"ridge-fixed","baseline_id":"training-mean","reports":[]}"#.to_owned(),
        r#"{"schema_version":"clotho.saved_report_manifest.v1","model_id":"ridge-fixed","baseline_id":"training-mean","reports":["a.json","a.json"]}"#.to_owned(),
        r#"{"schema_version":"clotho.saved_report_manifest.v1","model_id":"ridge-fixed","baseline_id":"training-mean","reports":["missing.json"]}"#.to_owned(),
        r#"["clotho.saved_report_manifest.v1",["a.json"]]"#.to_owned(),
        serde_json::json!({"schema_version":"clotho.saved_report_manifest.v1","model_id":"ridge-fixed","baseline_id":"training-mean","reports":vec!["a.json";65]}).to_string(),
    ] {
        let w = setup(report("a", "a", 1, 1., 2.), report("b", "b", 1, 1., 2.));
        w.write("manifest.json", &manifest);
        assert_eq!(w.run(&ARGS).status.code(), Some(2));
        assert!(!w.0.join("pooled.json").exists());
    }
    let valid = report("b", "b", 1, 1., 2.).to_string();
    for invalid in [
        format!("{{\"n_rows\":1,{}", &valid[1..]),
        valid.replace("\"baseline_rmse\":2.0", "\"baseline_rmse\":1e-999"),
        valid.replace("\"model_rmse\":1.0", "\"model_rmse\":1e999"),
        "[]".to_owned(), " ".repeat(65 * 1024),
    ] {
        let w = setup(report("a", "a", 1, 1., 2.), report("b", "b", 1, 1., 2.));
        w.write("b.json", &invalid);
        assert_eq!(w.run(&ARGS).status.code(), Some(2));
        assert!(!w.0.join("pooled.json").exists());
    }
}

#[test]
fn overly_wide_scaled_range_fails_instead_of_erasing_small_report_error() {
    let w = setup(report("a", "a", 1, 1e200, 1e200), report("b", "b", 1, 1e-200, 1e-200));
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert!(!w.0.join("pooled.json").exists());
}

#[test]
fn native_compare_reports_pool_with_manifest_relative_paths() {
    use sha2::{Digest, Sha256};
    let w = Workspace::new();
    fs::create_dir(w.0.join("reports")).unwrap();
    for (name, query) in [("a", "query a"), ("b", "query b")] {
        w.write("query.bin", query);
        for (file, values) in [
            ("observed.json", serde_json::json!([1., 3.])),
            ("model.json", serde_json::json!([1., 2.])),
            ("baseline.json", serde_json::json!([0., 1.])),
        ] {
            w.write(file, &serde_json::json!({
                "schema_version":"clotho.declared_regression_series.v2",
                "cohort":name,"target":"resistance","unit":"ohm","horizon":"1 write",
                "row_ids":["first","second"],"values":values,
                "query_sha256":format!("{:x}", Sha256::digest(query.as_bytes()))
            }).to_string());
        }
        assert!(w.run(&[
            "compare-saved", "--observed", "observed.json", "--model", "model.json",
            "--baseline", "baseline.json", "--query", "query.bin",
            "--out", &format!("reports/{name}.json"),
        ]).status.success());
    }
    w.write("reports/manifest.json", r#"{"schema_version":"clotho.saved_report_manifest.v1","model_id":"ridge-fixed","baseline_id":"training-mean","reports":["a.json","b.json"]}"#);
    let mut args = ARGS;
    args[2] = "reports/manifest.json";
    assert!(w.run(&args).status.success());
    assert!((read(&w)["mse_skill"].as_f64().unwrap() - 0.8).abs() < 1e-14);
    // The source rosters above deliberately overlap: distinct digest/cohort
    // declarations cannot establish physical or declared-row disjointness.
    assert_eq!(read(&w)["disjointness_verified"], false);
}

#[test]
fn manifest_requires_distinct_explicit_model_and_baseline_declarations() {
    for value in [None, Some(serde_json::json!("")), Some(serde_json::json!("training-mean")), Some(serde_json::json!("x".repeat(257)))] {
        let w = setup(report("a", "a", 1, 1., 2.), report("b", "b", 1, 1., 2.));
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(w.0.join("manifest.json")).unwrap()).unwrap();
        if let Some(value) = value {
            manifest["model_id"] = value;
        } else {
            manifest.as_object_mut().unwrap().remove("model_id");
        }
        w.write("manifest.json", &manifest.to_string());
        assert_eq!(w.run(&ARGS).status.code(), Some(2));
        assert!(!w.0.join("pooled.json").exists());
    }
}

#[test]
fn subnormal_ratios_cannot_bypass_the_core_range_through_scalar_reports() {
    let tiny = f64::from_bits(1);
    for (model, baseline) in [(tiny, tiny), (tiny, 1.0), (1.0, tiny)] {
        let w = setup(
            report("a", "a", 1, model, baseline),
            report("b", "b", 1, model, baseline),
        );
        assert!(!w.run(&ARGS).status.success());
        assert!(!w.0.join("pooled.json").exists());
    }
    for (model, baseline, expected) in [
        (0.0, tiny, Some(1.0)),
        (tiny, 0.0, None),
        (f64::MIN_POSITIVE, f64::MIN_POSITIVE, Some(0.0)),
    ] {
        let w = setup(
            report("a", "a", 1, model, baseline),
            report("b", "b", 1, model, baseline),
        );
        assert!(w.run(&ARGS).status.success());
        assert_eq!(read(&w)["mse_skill"].as_f64(), expected);
    }
    // Each source has an exact zero exception, but the pooled pair does not.
    let w = setup(
        report("a", "a", 1, tiny, 0.0),
        report("b", "b", 1, 0.0, tiny),
    );
    assert!(!w.run(&ARGS).status.success());
    assert!(!w.0.join("pooled.json").exists());
}

#[test]
fn pooled_skill_retains_adjacent_rmse_differences() {
    let lower = f64::from_bits(1.0_f64.to_bits() - 1);
    let higher = f64::from_bits(2.7_f64.to_bits() + 1);
    // Reference values from exact rational arithmetic on these binary64 RMSEs.
    for (model, baseline, expected) in [
        (1.0, lower, -2.2204460492503136e-16),
        (2.7, higher, 3.2895497025930552e-16),
    ] {
        // Two equally sized reports preserve the identical RMSE in the pool.
        // Inputs intentionally use the historical native compare-saved formula.
        let w = setup(
            report("a", "a", 1, model, baseline),
            report("b", "b", 1, model, baseline),
        );
        let result = w.run(&ARGS);
        assert!(result.status.success(), "{:?}", result.stderr);
        let pooled = read(&w);
        assert_eq!(pooled["model_rmse"].as_f64(), Some(model));
        assert_eq!(pooled["baseline_rmse"].as_f64(), Some(baseline));
        let actual = pooled["mse_skill"].as_f64().unwrap();
        assert!((actual - expected).abs() < 1e-30, "{actual} vs {expected}");
    }
}
