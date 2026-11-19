use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "clotho-knn-precision-{}-{}-{}",
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




#[test]
fn fitted_artifact_reload_preserves_exact_nearest_neighbor() {
    let w = Workspace::new();
    w.write("train.csv", "x,y\n7,0\n7.000000000000001,1\n12,2\n");
    w.write("query.csv", "x\n7.000000000000001\n7\n");
    assert!(w.run(&[
        "fit", "--csv", "train.csv", "--features", "x", "--target", "y",
        "--model", "knn", "--k", "1", "--out", "model.json",
    ]).status.success());
    assert!(w.run(&[
        "predict", "--artifact", "model.json", "--csv", "query.csv", "--out", "out.csv",
    ]).status.success());
    let text = fs::read_to_string(w.0.join("out.csv")).unwrap();
    let values: Vec<f64> = text.lines().skip(1)
        .map(|line| line.split(',').nth(1).unwrap().parse().unwrap()).collect();
    assert_eq!(values, vec![1.0, 0.0]);
}

#[test]
fn reloaded_neighbor_mean_preserves_representable_cancellation_residual() {
    let w = Workspace::new();
    w.write("train.csv", "x,y\n0,1e308\n0,-1e308\n0,1e-15\n");
    w.write("query.csv", "x\n0\n");
    assert!(w.run(&[
        "fit", "--csv", "train.csv", "--features", "x", "--target", "y",
        "--model", "knn", "--k", "3", "--out", "model.json",
    ]).status.success());
    assert!(w.run(&[
        "predict", "--artifact", "model.json", "--csv", "query.csv", "--out", "out.csv",
    ]).status.success());
    let csv = fs::read_to_string(w.0.join("out.csv")).unwrap();
    let value: f64 = csv.lines().nth(1).unwrap().split(',').nth(1).unwrap().parse().unwrap();
    assert_eq!(value.to_bits(), (1e-15_f64 / 3.0).to_bits());
}
