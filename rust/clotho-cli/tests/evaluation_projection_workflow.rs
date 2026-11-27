use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}, process::{Command, Output}, sync::atomic::{AtomicUsize, Ordering}};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Work(PathBuf);

impl Work {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("clotho-eval-projection-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn evaluate(work: &Work, binary: bool) -> PathBuf {
    let csv = work.0.join("source.csv");
    fs::write(&csv, concat!(
        "private_feature,private_target,private_group\n",
        "1,0,<private-a>\n2,0,<private-a>\n",
        "3,1,<private-b>\n4,1,<private-b>\n",
        "5,0,<private-c>\n6,1,<private-c>\n",
        "7,0,<private-d>\n8,1,<private-d>\n"
    )).unwrap();
    let bundle = work.0.join("bundle");
    let out = Command::new(env!("CARGO_BIN_EXE_clotho"))
        .args(["evaluate", "--csv"]).arg(csv)
        .args(["--features", "private_feature", "--target", "private_target", "--group", "private_group",
            "--model", if binary { "majority" } else { "mean" }, "--folds", "2", "--out"])
        .arg(&bundle).output().unwrap();
    assert!(out.status.success(), "{}",
        String::from_utf8_lossy(&out.stderr));
    bundle
}

fn inspect(bundle: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_clotho"))
        .args(["inspect-evaluation", "--bundle"]).arg(bundle).output().unwrap()
}

fn inspect_success(bundle: &Path) -> Value {
    let out = inspect(bundle);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let bytes = String::from_utf8(out.stdout).unwrap();
    for secret in ["private", "<", "row_index", "positive_probability", "majority", "\"model\"", "\"groups\""] {
        assert!(!bytes.contains(secret), "projection leaked {secret}");
    }
    serde_json::from_str(&bytes).unwrap()
}

fn assert_rejected(bundle: &Path) {
    let out = inspect(bundle);
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&out.stderr).contains("private"));
}

fn alter(bundle: &Path, edit: impl FnOnce(&mut Value)) {
    let path = bundle.join("metrics.json");
    let mut original = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    edit(&mut original);
    fs::write(path, serde_json::to_vec(&original).unwrap()).unwrap();
}

#[test]
fn native_regression_and_binary_producers_project_aggregates_only() {
    for binary in [false, true] {
        let work = Work::new();
        let bundle = evaluate(&work, binary);
        alter(&bundle, |v| v["extra_metadata"] = json!("<script>private</script>"));
        let proj = inspect_success(&bundle);
        let source: Value = serde_json::from_slice(&fs::read(bundle.join("metrics.json")).unwrap()).unwrap();
        assert_eq!(proj["schema_version"], "clotho.grouped_evaluation_projection.v1");
        assert_eq!(proj["task"], if binary { "binary" } else { "regression" });
        assert_eq!(proj["n_rows"], 8);
        assert_eq!(proj["n_groups"], 4);
        assert_eq!(proj["n_folds"], 2);
        assert_eq!(proj["pooled_row_weighted"], source["pooled_row_weighted"]);
        assert_eq!(proj["equal_group_weighted"], source["equal_group_weighted"]);
        assert_eq!(proj["folds"].as_array().unwrap().len(), 2);
        assert!(proj.get("threshold").is_some() == binary);
    }
}

#[test]
fn tampered_metrics_rosters_counts_and_nullability_fail_closed() {
    for mode in 0..8 {
        let work = Work::new();
        let binary = mode >= 5;
        let bundle = evaluate(&work, binary);
        alter(&bundle, |v| match mode {
            0 => v["pooled_row_weighted"]["mae"] = json!(2),
            1 => v["folds"][0]["n_test"] = json!(1),
            2 => { v["pooled_row_weighted"].as_object_mut().unwrap().remove("r2"); },
            3 => v["groups"][1]["group"] = v["groups"][0]["group"].clone(),
            4 => v["equal_group_weighted"]["mean_group_mae"] = json!(0.0),
            5 => v["groups"][0]["metrics"]["tp"] = json!(99),
            6 => v["positive_label"] = json!(0),
            _ => v["threshold"] = json!(2.0),
        });
        assert_rejected(&bundle);
    }
}

#[test]
fn malformed_or_incomplete_csv_and_unsupported_markers_fail_closed() {
    for mode in 0..7 {
        let work = Work::new();
        let bundle = evaluate(&work, mode == 6);
        let csv = bundle.join("predictions.csv");
        let mut text = fs::read_to_string(&csv).unwrap();
        match mode {
            0 => { fs::remove_file(bundle.join("COMPLETE")).unwrap(); },
            1 => { fs::write(bundle.join("COMPLETE"), "clotho.nested_regression.v1\n").unwrap(); },
            2 => { text = text.replace("target,prediction", "prediction,target"); fs::write(csv, text).unwrap(); },
            3 => { text = text.replace("0,0,0,", "3,0,0,"); fs::write(csv, text).unwrap(); },
            4 => { text.push_str("999,0,0,\"unterminated"); fs::write(csv, text).unwrap(); },
            5 => { fs::remove_file(csv).unwrap(); },
            _ => {
                let mut rows: Vec<_> = text.lines().map(str::to_owned).collect();
                rows[1].pop();
                rows[1].push('2');
                fs::write(csv, rows.join("\n")).unwrap();
            },
        }
        assert_rejected(&bundle);
    }
}

#[cfg(unix)]
#[test]
fn rejects_symlink_members_and_bundle_even_with_trailing_slash() {
    use std::os::unix::fs::symlink;
    for member in ["predictions.csv", "metrics.json", "COMPLETE"] {
        let work = Work::new();
        let bundle = evaluate(&work, false);
        let original = work.0.join("backing");
        fs::rename(bundle.join(member), &original).unwrap();
        symlink(original, bundle.join(member)).unwrap();
        assert_rejected(&bundle);
    }
    let work = Work::new();
    let bundle = evaluate(&work, false);
    let link = work.0.join("linked");
    symlink(bundle, &link).unwrap();
    assert_rejected(&link);
    assert_rejected(Path::new(&format!("{}/", link.display())));
}

#[test]
fn genuine_encoder_width_boundary_is_shared_with_the_producer() {
    for second_categories in [48, 49] {
        let work = Work::new();
        let csv = work.0.join("source.csv");
        let mut source = String::from("private_feature,private_second,private_target,private_group\n");
        for group in ["<private-a>", "<private-b>"] {
            for c in 0..50 {
                source.push_str(&format!("private-first-{c},private-second-{},{},{group}\n", c % second_categories, c % 11));
            }
        }
        fs::write(&csv, source).unwrap();
        let bundle = work.0.join("bundle");
        let out = Command::new(env!("CARGO_BIN_EXE_clotho"))
            .args(["evaluate", "--csv"]).arg(csv)
            .args(["--features", "private_feature,private_second", "--target", "private_target",
                "--group", "private_group", "--model", "mean", "--folds", "2",
                "--preprocess", "mixed", "--categorical", "private_feature,private_second", "--out"])
            .arg(&bundle).output().unwrap();
        if second_categories == 48 {
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            let projection = inspect_success(&bundle);
            assert_eq!(projection["folds"][0]["encoded_feature_count"], 100);
        } else {
            assert!(!out.status.success());
            assert!(!bundle.exists());
        }
    }
}

#[test]
fn fitted_models_round_trip_nontrivial_binary_and_regression_scales() {
    for model in ["ridge", "logistic"] {
        let work = Work::new();
        let csv = work.0.join("source.csv");
        let mut source = String::from("private_feature,private_target,private_group\n");
        for group in 0..6 {
            for i in 0..7 {
                let x = (group * 23 + i * 7) as f64 * 0.03125 - 1.7;
                let y = if model == "ridge" { 1e16 + 1e12 * x + (i as f64).sin() * 1e11 }
                    else if (group + i) % 3 == 0 { 0.0 } else { 1.0 };
                source.push_str(&format!("{x},{y},<private-{group}>\n"));
            }
        }
        fs::write(&csv, source).unwrap();
        let bundle = work.0.join("bundle");
        let out = Command::new(env!("CARGO_BIN_EXE_clotho"))
            .args(["evaluate", "--csv"]).arg(csv)
            .args(["--features", "private_feature", "--target", "private_target",
                "--group", "private_group", "--model", model, "--folds", "3", "--out"])
            .arg(&bundle).output().unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let projection = inspect_success(&bundle);
        assert_eq!(projection["n_rows"], 42);
    }
}

#[test]
fn genuine_sources_bind_encoded_width_to_preprocessing_declarations() {
    for binary in [false, true] {
        let mutations: [fn(&mut Value); 5] = [
            |v| v["folds"][0]["encoded_feature_count"] = json!(2),
            |v| v["features"][0] = v["target"].clone(),
            |v| v["preprocess"] = json!("not-a-processor"),
            |v| v["categorical_features"] = json!(["private_feature"]),
            |v| {
                v["preprocess"] = json!("mixed");
                v["categorical_features"] = json!(["private_absent_feature"]);
            },
        ];
        for mutate in mutations {
            let work = Work::new();
            let bundle = evaluate(&work, binary);
            let first = inspect_success(&bundle);
            // Numeric-only imputation retains one output per input.
            alter(&bundle, |v| v["preprocess"] = json!("mixed"));
            assert_eq!(inspect_success(&bundle), first);
            alter(&bundle, |v| v["preprocess"] = json!("numeric"));
            alter(&bundle, mutate);
            assert_rejected(&bundle);
        }
    }
}
