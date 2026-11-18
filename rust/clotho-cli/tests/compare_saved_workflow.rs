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
            "clotho-saved-{}-{}-{}",
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
    fn write(&self, name: &str, s: &str) {
        fs::write(self.0.join(name), s).unwrap();
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

fn series(values: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "schema_version": "clotho.declared_regression_series.v1",
        "cohort": "synthetic-1", "target": "resistance", "unit": "ohm",
        "horizon": "1 write", "row_ids": ["a", "b"], "values": values
    })
}

fn setup() -> Workspace {
    let w = Workspace::new();
    for (name, values) in [
        ("observed.json", serde_json::json!([1., 3.])),
        ("model.json", serde_json::json!([1., 2.])),
        ("baseline.json", serde_json::json!([0., 1.])),
    ] {
        w.write(name, &series(values).to_string());
    }
    w
}

const ARGS: [&str; 9] = [
    "compare-saved", "--observed", "observed.json", "--model", "model.json",
    "--baseline", "baseline.json", "--out", "report.json",
];

#[test]
fn scores_existing_series_without_model_artifacts() {
    let w = setup();
    let output = w.run(&ARGS);
    assert!(output.status.success(), "{:?}", output);
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
    assert_eq!(
        report["schema_version"],
        "clotho.declared_regression_comparison.v1"
    );
    assert_eq!(report["cohort"], "synthetic-1");
    assert_eq!(report["n_rows"], 2);
    assert!((report["mse_skill"].as_f64().unwrap() - 0.8).abs() < 1e-14);
    for flag in ["fit_performed", "provenance_verified", "query_bytes_verified"] {
        assert_eq!(report[flag], false);
    }
    assert!(report.get("row_ids").is_none());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(w.0.join("report.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}

#[test]
fn invalid_inputs_leave_no_output() {
    for (field, value) in [
        ("row_ids", serde_json::json!(["b", "a"])),
        ("row_ids", serde_json::json!(["a", "a"])),
        ("row_ids", serde_json::json!(["a"])),
        ("row_ids", serde_json::json!(["a", " "])),
        ("values", serde_json::json!([1.])),
        ("values", serde_json::json!([1., null])),
        ("cohort", serde_json::json!("other")),
        ("target", serde_json::json!("conductance")),
        ("unit", serde_json::json!("kohm")),
        ("horizon", serde_json::json!("2 writes")),
        ("schema_version", serde_json::json!("unknown")),
        ("extra", serde_json::json!("not allowed")),
    ] {
        for file in ["observed.json", "model.json", "baseline.json"] {
            let w = setup();
            let mut bad = series(serde_json::json!([1., 2.]));
            bad[field] = value.clone();
            w.write(file, &bad.to_string());
            let output = w.run(&ARGS);
            assert_eq!(output.status.code(), Some(2), "{file} {field}");
            assert!(!w.0.join("report.json").exists());
        }
    }
}

#[test]
fn strict_json_and_cli_reject_before_output() {
    for contents in [
        "not json", "[]", "{\"schema_version\":\"x\",\"schema_version\":\"x\"}",
        r#"["clotho.declared_regression_series.v1","synthetic-1","resistance","ohm","1 write",["a","b"],[1,2]]"#,
        "row_index,prediction\n0,1\n1,2\n",
    ] {
        let w = setup();
        w.write("model.json", contents);
        assert_eq!(w.run(&ARGS).status.code(), Some(2));
        assert!(!w.0.join("report.json").exists());
    }
    let w = setup();
    let mut args = ARGS.to_vec();
    args.extend(["--artifact", "nonexistent.json"]);
    assert_eq!(w.run(&args).status.code(), Some(2));
    assert_eq!(w.run(&ARGS[..7]).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());
}

#[test]
fn perfect_baseline_is_json_null_and_existing_output_is_preserved() {
    let w = setup();
    w.write("baseline.json", &series(serde_json::json!([1., 3.])).to_string());
    assert!(w.run(&ARGS).status.success());
    let bytes = fs::read(w.0.join("report.json")).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(report["mse_skill"].is_null());
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert_eq!(fs::read(w.0.join("report.json")).unwrap(), bytes);
}

#[test]
fn matching_false_keys_do_not_verify_provenance() {
    let w = setup();
    w.write("model.json", &series(serde_json::json!([3., 1.])).to_string());
    assert!(w.run(&ARGS).status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["provenance_verified"], false);
    assert!(report["mse_skill"].as_f64().unwrap() < 0.);
}

#[test]
fn capacity_and_unrepresentable_scores_leave_no_output() {
    let w = setup();
    let mut bad = series(serde_json::json!([1., 2.]));
    bad["row_ids"] = serde_json::json!(vec!["a"; 20_001]);
    bad["values"] = serde_json::json!(vec![1.; 20_001]);
    w.write("model.json", &bad.to_string());
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());

    let w = setup();
    w.write("observed.json", &series(serde_json::json!([0., 0.])).to_string());
    w.write("baseline.json", &series(serde_json::json!([1e-300, 1e-300])).to_string());
    w.write("model.json", &series(serde_json::json!([1e300, 1e300])).to_string());
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());
}

const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

fn setup_bound(digest: &str) -> Workspace {
    let w = setup();
    for file in ["observed.json", "model.json", "baseline.json"] {
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(w.0.join(file)).unwrap()).unwrap();
        value["schema_version"] = "clotho.declared_regression_series.v2".into();
        value["query_sha256"] = digest.into();
        w.write(file, &value.to_string());
    }
    w.write("query.bin", "abc");
    w
}

fn bound_args() -> Vec<&'static str> {
    let mut args = ARGS.to_vec();
    args.extend(["--query", "query.bin"]);
    args
}

#[test]
fn v2_matches_known_sha256_vector_but_does_not_verify_prediction_origin() {
    // All declarations here are manually copied; no predictions were generated
    // from "abc". Passing is byte matching, never proof of producer provenance.
    let w = setup_bound(ABC_SHA256);
    let output = w.run(&bound_args());
    assert!(output.status.success(), "{:?}", output);
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "clotho.declared_regression_comparison.v2");
    assert_eq!(report["query_sha256"], ABC_SHA256);
    assert_eq!(report["query_bytes_verified"], true);
    assert_eq!(report["provenance_verified"], false);
    assert_eq!(report["fit_performed"], false);
    assert!((report["mse_skill"].as_f64().unwrap() - 0.8).abs() < 1e-14);
    assert!(report.get("query_path").is_none());
}

#[test]
fn changed_query_bytes_and_each_wrong_digest_fail_without_output() {
    for bytes in [b"abc\n".as_slice(), b"abd", b"ABC", b""] {
        let w = setup_bound(ABC_SHA256);
        fs::write(w.0.join("query.bin"), bytes).unwrap();
        assert_eq!(w.run(&bound_args()).status.code(), Some(2));
        assert!(!w.0.join("report.json").exists());
    }
    for file in ["observed.json", "model.json", "baseline.json"] {
        for digest in ["0".repeat(64), ABC_SHA256.to_uppercase(), "a".repeat(63)] {
            let w = setup_bound(ABC_SHA256);
            let mut value: serde_json::Value =
                serde_json::from_slice(&fs::read(w.0.join(file)).unwrap()).unwrap();
            value["query_sha256"] = digest.into();
            w.write(file, &value.to_string());
            assert_eq!(w.run(&bound_args()).status.code(), Some(2));
            assert!(!w.0.join("report.json").exists());
        }
    }
}

#[test]
fn v2_is_explicit_and_never_silently_downgrades_to_v1() {
    let w = setup_bound(ABC_SHA256);
    assert_eq!(w.run(&ARGS).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());
    let w = setup();
    w.write("query.bin", "abc");
    assert_eq!(w.run(&bound_args()).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());
    for file in ["observed.json", "model.json", "baseline.json"] {
        for digest in [None, Some(serde_json::Value::Null), Some(ABC_SHA256.into())] {
            let w = setup_bound(ABC_SHA256);
            let mut value = series(serde_json::json!([1., 2.]));
            if let Some(digest) = digest {
                value["query_sha256"] = digest;
            }
            w.write(file, &value.to_string());
            assert_eq!(w.run(&bound_args()).status.code(), Some(2));
            assert!(!w.0.join("report.json").exists());
        }
    }
}

#[test]
fn matching_digest_cannot_bypass_key_checks_or_output_protection() {
    let w = setup_bound(ABC_SHA256);
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("model.json")).unwrap()).unwrap();
    value["row_ids"] = serde_json::json!(["b", "a"]);
    w.write("model.json", &value.to_string());
    assert_eq!(w.run(&bound_args()).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());
    let w = setup_bound(ABC_SHA256);
    w.write("report.json", "do not overwrite");
    assert_eq!(w.run(&bound_args()).status.code(), Some(2));
    assert_eq!(fs::read(w.0.join("report.json")).unwrap(), b"do not overwrite");
}

#[test]
fn empty_query_is_a_valid_byte_artifact_and_missing_query_is_not() {
    let w = setup_bound("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    w.write("query.bin", "");
    assert!(w.run(&bound_args()).status.success());
    let w = setup_bound(ABC_SHA256);
    fs::remove_file(w.0.join("query.bin")).unwrap();
    assert_eq!(w.run(&bound_args()).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());
}

#[test]
fn v2_requires_one_nonnull_digest_and_bounded_query() {
    for digest in [None, Some(serde_json::Value::Null)] {
        let w = setup_bound(ABC_SHA256);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(w.0.join("model.json")).unwrap()).unwrap();
        value.as_object_mut().unwrap().remove("query_sha256");
        if let Some(digest) = digest {
            value["query_sha256"] = digest;
        }
        w.write("model.json", &value.to_string());
        assert_eq!(w.run(&bound_args()).status.code(), Some(2));
        assert!(!w.0.join("report.json").exists());
    }
    let w = setup_bound(ABC_SHA256);
    let original = fs::read_to_string(w.0.join("model.json")).unwrap();
    let duplicate = format!("{{\"query_sha256\":\"{ABC_SHA256}\",{}", &original[1..]);
    w.write("model.json", &duplicate);
    assert_eq!(w.run(&bound_args()).status.code(), Some(2));
    assert!(!w.0.join("report.json").exists());

    let w = setup_bound(ABC_SHA256);
    fs::File::create(w.0.join("query.bin"))
        .unwrap()
        .set_len(64 * 1024 * 1024 + 1)
        .unwrap();
    let output = w.run(&bound_args());
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("64 MiB"));
    assert!(!w.0.join("report.json").exists());
}

fn group_args() -> Vec<&'static str> {
    let mut args = ARGS.to_vec();
    args.extend(["--groups", "groups.json"]);
    args
}

fn mapping(ids: serde_json::Value, groups: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "schema_version": "clotho.declared_row_groups.v1", "cohort": "synthetic-1",
        "row_ids": ids, "group_ids": groups
    })
}

#[test]
fn declared_groups_show_pooled_macro_sign_reversal_without_ratios() {
    let w = setup();
    for (file, values) in [
        ("observed.json", serde_json::json!([0., 0., 0., 0., 0.])),
        ("model.json", serde_json::json!([0., 0., 0., 0., 3.])),
        ("baseline.json", serde_json::json!([2., 2., 2., 2., 0.])),
    ] {
        let mut value = series(values);
        value["row_ids"] = serde_json::json!(["a", "b", "c", "d", "e"]);
        w.write(file, &value.to_string());
    }
    w.write("groups.json", &mapping(
        serde_json::json!(["a", "b", "c", "d", "e"]),
        serde_json::json!(["large", "large", "large", "large", "small"]),
    ).to_string());
    assert!(w.run(&group_args()).status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
    let g = &report["grouped_summary"];
    assert_eq!(g["n_groups"], 2);
    assert_eq!(g["n_rows"], 5);
    assert!((g["pooled_row_mean_squared_error_delta"].as_f64().unwrap() - 1.4).abs() < 1e-14);
    assert_eq!(g["macro_equal_group_mean_squared_error_delta"], -2.5);
    assert_eq!(g["improved_groups"], 1);
    assert_eq!(g["worsened_groups"], 1);
    assert_eq!(g["baseline_perfect_groups"], 1);
    assert_eq!(g["groups"][1]["mean_squared_error_delta"], -9.);
    assert_eq!(g["groups"][1]["baseline_mse"], 0.);
    assert_eq!(g["groups"][1]["model_mse"], 9.);
    assert_eq!(g["groups"][1]["baseline_perfect"], true);
    assert_eq!(g["independence_verified"], false);
    assert_eq!(g["provenance_verified"], false);
    fs::remove_file(w.0.join("report.json")).unwrap();
    assert!(w.run(&ARGS).status.success());
    let pooled: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
    let mut stripped = report;
    stripped.as_object_mut().unwrap().remove("grouped_summary");
    assert_eq!(stripped, pooled);
}

#[test]
fn mapping_rejects_ambiguous_missing_duplicate_and_reordered_rows() {
    let valid = mapping(serde_json::json!(["a", "b"]), serde_json::json!(["one", "two"]));
    for (field, value) in [
        ("row_ids", serde_json::json!(["b", "a"])),
        ("row_ids", serde_json::json!(["a", "a"])),
        ("row_ids", serde_json::json!(["a"])),
        ("row_ids", serde_json::json!(["a", "b", "c"])),
        ("group_ids", serde_json::json!(["one"])),
        ("group_ids", serde_json::json!(["one", " "])),
        ("group_ids", serde_json::json!(["one", "x\ny"])),
        ("group_ids", serde_json::json!(["one", "x".repeat(257)])),
        ("cohort", serde_json::json!("other")),
        ("schema_version", serde_json::json!("unknown")),
        ("extra", serde_json::json!(true)),
    ] {
        let w = setup();
        let mut bad = valid.clone();
        bad[field] = value;
        w.write("groups.json", &bad.to_string());
        assert_eq!(w.run(&group_args()).status.code(), Some(2));
        assert!(!w.0.join("report.json").exists());
    }
    for text in [
        "[\"clotho.declared_row_groups.v1\",\"synthetic-1\",[\"a\",\"b\"],[\"one\",\"two\"]]".to_owned(),
        format!("{{\"cohort\":\"synthetic-1\",{}", &valid.to_string()[1..]),
    ] {
        let w = setup();
        w.write("groups.json", &text);
        assert_eq!(w.run(&group_args()).status.code(), Some(2));
        assert!(!w.0.join("report.json").exists());
    }
}

#[test]
fn groups_keep_perfect_and_tiny_baselines_honest_and_reject_square_range_loss() {
    for values in [serde_json::json!([1., 3.]), serde_json::json!([1e-150, 1e-150])] {
        let w = setup();
        w.write("baseline.json", &series(values.clone()).to_string());
        w.write("model.json", &series(values).to_string());
        w.write("groups.json", &mapping(
            serde_json::json!(["a", "b"]), serde_json::json!(["one", "two"]),
        ).to_string());
        assert!(w.run(&group_args()).status.success());
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
        assert_eq!(report["grouped_summary"]["tied_groups"], 2);
    }
    for magnitude in [1e200, 1e-200] {
        let w = setup();
        w.write("observed.json", &series(serde_json::json!([0., 0.])).to_string());
        w.write("model.json", &series(serde_json::json!([magnitude, magnitude])).to_string());
        w.write("baseline.json", &series(serde_json::json!([magnitude, magnitude])).to_string());
        w.write("groups.json", &mapping(
            serde_json::json!(["a", "b"]), serde_json::json!(["one", "one"]),
        ).to_string());
        assert_eq!(w.run(&group_args()).status.code(), Some(2));
        assert!(!w.0.join("report.json").exists());
        assert!(w.run(&ARGS).status.success());
    }
}

#[test]
fn grouped_v2_and_tiny_nonzero_errors_are_not_reported_as_perfect() {
    let w = setup_bound(ABC_SHA256);
    w.write("observed.json", &{
        let mut v = series(serde_json::json!([0., 0.]));
        v["schema_version"] = "clotho.declared_regression_series.v2".into();
        v["query_sha256"] = ABC_SHA256.into();
        v.to_string()
    });
    for (file, values) in [
        ("model.json", serde_json::json!([0., 0.])),
        ("baseline.json", serde_json::json!([1e-150, 1e-150])),
    ] {
        let mut v = series(values);
        v["schema_version"] = "clotho.declared_regression_series.v2".into();
        v["query_sha256"] = ABC_SHA256.into();
        w.write(file, &v.to_string());
    }
    w.write("groups.json", &mapping(
        serde_json::json!(["a", "b"]), serde_json::json!(["one", "two"]),
    ).to_string());
    let mut args = bound_args();
    args.extend(["--groups", "groups.json"]);
    assert!(w.run(&args).status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["query_bytes_verified"], true);
    assert_eq!(report["grouped_summary"]["baseline_perfect_groups"], 0);
    assert_eq!(report["grouped_summary"]["improved_groups"], 2);
    assert!(report["grouped_summary"]["macro_equal_group_mean_squared_error_delta"]
        .as_f64().unwrap() > 0.);
}

#[test]
fn grouped_compensated_mean_retains_small_paired_gain_after_cancellation() {
    let w = setup();
    for (file, values) in [
        ("observed.json", serde_json::json!([0., 0., 0.])),
        ("model.json", serde_json::json!([0., 0., 1e150])),
        ("baseline.json", serde_json::json!([1e150, 1., 0.])),
    ] {
        let mut value = series(values);
        value["row_ids"] = serde_json::json!(["a", "b", "c"]);
        w.write(file, &value.to_string());
    }
    w.write("groups.json", &mapping(
        serde_json::json!(["a", "b", "c"]),
        serde_json::json!(["one", "one", "one"]),
    ).to_string());
    assert!(w.run(&group_args()).status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("report.json")).unwrap()).unwrap();
    let delta = report["grouped_summary"]["pooled_row_mean_squared_error_delta"]
        .as_f64().unwrap();
    assert!((delta - 1. / 3.).abs() < 1e-14);
}


// Use literal JSON tokens: constructing serde_json::Value first would already
// have lost the distinction this file-boundary regression test protects.
fn token_series(token: &str, v2: bool) -> String {
    let version = if v2 { "v2" } else { "v1" };
    let digest = if v2 {
        // SHA-256 of the exact query bytes "abc".
        r#", "query_sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad""#
    } else {
        ""
    };
    format!(
        r#"{{"schema_version":"clotho.declared_regression_series.{version}","cohort":"1e-999","target":"y","unit":"u","horizon":"h","row_ids":["escaped\"1e-999\\tail"],"values":[{token}]{digest}}}"#
    )
}

#[test]
fn all_saved_input_roles_reject_nonzero_decimal_to_zero_in_v1_and_v2() {
    for v2 in [false, true] {
        for role in ["observed.json", "model.json", "baseline.json"] {
            for token in ["1e-999", "-1E-999", "2e-324", "0.000001e-999"] {
                let w = Workspace::new();
                w.write("query", "abc");
                for file in ["observed.json", "model.json", "baseline.json"] {
                    w.write(file, &token_series(if file == role { token } else { "0" }, v2));
                }
                let mut args = ARGS.to_vec();
                if v2 {
                    args.extend(["--query", "query"]);
                }
                let output = w.run(&args);
                assert!(!output.status.success(), "accepted {role} {token}");
                let error = String::from_utf8_lossy(&output.stderr);
                assert!(error.contains("underflows binary64"), "{error}");
                assert!(!error.contains(token));
                assert!(!w.0.join("report.json").exists());
            }
        }
    }
}

#[test]
fn saved_zero_subnormal_and_quoted_numeric_looking_ids_remain_valid() {
    for v2 in [false, true] {
        for token in ["0", "-0.0", "0e-999", "-0.000E+999", "5e-324", "-5e-324", "3e-324"] {
            let w = Workspace::new();
            w.write("query", "abc");
            // Identical representable inputs give exact zero errors, avoiding
            // any separate limitations of the squared-error arithmetic.
            for file in ["observed.json", "model.json", "baseline.json"] {
                w.write(file, &token_series(token, v2));
            }
            let mut args = ARGS.to_vec();
            if v2 {
                args.extend(["--query", "query"]);
            }
            let output = w.run(&args);
            assert!(output.status.success(), "{token}: {output:?}");
            let report: serde_json::Value = serde_json::from_slice(
                &fs::read(w.0.join("report.json")).unwrap(),
            ).unwrap();
            assert_eq!(report["model_rmse"], 0.0);
            assert_eq!(report["baseline_rmse"], 0.0);
            assert!(report["mse_skill"].is_null());
            assert_eq!(report["query_bytes_verified"], v2);
        }
    }
}

#[test]
fn subnormal_rmse_ratio_does_not_publish_a_false_skill_tie() {
    let w = Workspace::new();
    for (file, values) in [
        ("observed.json", "[0,0]"),
        ("model.json", "[5e-324,0]"),
        ("baseline.json", "[5e-324,5e-324]"),
    ] {
        let text = format!(
            r#"{{"schema_version":"clotho.declared_regression_series.v1","cohort":"synthetic","target":"y","unit":"u","horizon":"h","row_ids":["a","b"],"values":{values}}}"#
        );
        w.write(file, &text);
    }
    let output = w.run(&ARGS);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("Evaluation exceeds the supported floating-point range"));
    assert!(!w.0.join("report.json").exists());
}
