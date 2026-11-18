use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-legacy-json-{}-{}-{}",
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



fn artifact(x: &str, y: &str, alpha: &str, model: &str) -> String {
    format!(r#"{{"schema_version":"clotho.numeric.v3","model":"{model}","features":["x"],"target":"y","k":1,"alpha":{alpha},"training_x":[[{x}],[1]],"training_y":[{y},1]}}"#)
}
fn predict(w: &Workspace) -> Output {
    w.run(&["predict", "--artifact", "model.json", "--csv", "query.csv", "--out", "out.csv"])
}

#[test]
fn consumed_numeric_arrays_and_alpha_reject_loss_to_zero_without_output() {
    for model in ["mean", "ridge"] {
        for token in ["1e-999", "-1e-999", "2e-324", "2.4703282292062327e-324"] {
            for (x, y, alpha) in [(token, "0", "1"), ("0", token, "1"), ("0", "0", token)] {
                let w = Workspace::new();
                w.write("model.json", &artifact(x, y, alpha, model));
                w.write("query.csv", "x\n0\n");
                let result = predict(&w);
                assert!(!result.status.success(), "accepted {model} {token}");
                let error = String::from_utf8_lossy(&result.stderr);
                assert!(error.contains("underflows binary64"));
                assert!(!error.contains(token));
                assert!(!w.0.join("out.csv").exists());
            }
        }
    }
}

#[test]
fn zero_subnormal_and_quoted_names_preserve_actual_predictions() {
    for token in ["0", "-0.0", "0e-999", "5e-324", "3e-324", "2.4703282292062328e-324"] {
        let w = Workspace::new();
        // One-row Mean must return the actual serde-converted target bitwise.
        let text = format!(r#"{{"schema_version":"clotho.numeric.v3","model":"mean","features":["1e-999"],"target":"quoted\"1e-999\\name","k":1,"alpha":0e-999,"training_x":[[{token}]],"training_y":[{token}]}}"#);
        w.write("model.json", &text);
        w.write("query.csv", "1e-999\n0\n");
        let output = predict(&w);
        assert!(output.status.success(), "{token}: {output:?}");
        let csv = fs::read_to_string(w.0.join("out.csv")).unwrap();
        let predicted: f64 = csv.lines().nth(1).unwrap().split(',').nth(1).unwrap().parse().unwrap();
        // Mean normalizes signed zero to positive zero; nonzero fidelity matters.
        let expected = token.parse::<f64>().unwrap();
        if expected != 0.0 {
            assert_eq!(predicted.to_bits(), expected.to_bits());
        } else {
            assert_eq!(predicted, 0.0);
        }
    }
}

#[test]
fn authored_mean_ridge_roundtrip_and_malformed_schema_remain_checked() {
    for model in ["mean", "ridge"] {
        let w = Workspace::new();
        w.write("train.csv", "x,y\n0,0\n1,1\n2,2\n");
        w.write("query.csv", "x\n0\n1\n2\n");
        assert!(w.run(&[
            "fit", "--csv", "train.csv", "--features", "x", "--target", "y",
            "--model", model, "--alpha", "1", "--out", "model.json",
        ]).status.success());
        assert!(predict(&w).status.success());
        let csv = fs::read_to_string(w.0.join("out.csv")).unwrap();
        let values: Vec<f64> = csv.lines().skip(1).map(|s| s.split(',').nth(1).unwrap().parse().unwrap()).collect();
        let expected = if model == "mean" { [1.0, 1.0, 1.0] } else { [0.25, 1.0, 1.75] };
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
    for invalid in [
        artifact("0", "0", "1", "mean").replace("\"k\":1", "\"k\":1,\"k\":2"),
        artifact("0", "0", "1", "mean").replace("\"k\":1", "\"unknown\":1"),
        artifact("0", "0", "1", "mean").replace("\"training_y\":[0,1]", "\"training_y\":[0]"),
        artifact("0", "0", "1e999", "mean"),
    ] {
        let w = Workspace::new();
        w.write("model.json", &invalid);
        w.write("query.csv", "x\n0\n");
        assert!(!predict(&w).status.success());
        assert!(!w.0.join("out.csv").exists());
    }
}
