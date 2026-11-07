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
            "clotho-prepare-{}-{}-{}",
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
    fn write(&self, n: &str, s: &str) {
        fs::write(self.0.join(n), s).unwrap();
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
fn setup(w: &Workspace) {
    w.write(
        "features.csv",
        "lot,wafer,device,x,unused\nB,W,D,2,unused\nA,W,D,1,unused\n",
    );
    w.write(
        "partitions.csv",
        "lot,wafer,device,partition\nA,W,D,train\nB,W,D,predict\n",
    );
    w.write("targets.csv", "lot,wafer,device,target\nA,W,D,3\n");
}
fn prepare(w: &Workspace) -> Output {
    w.run(&[
        "prepare-measurements",
        "--csv",
        "features.csv",
        "--partitions",
        "partitions.csv",
        "--training-targets",
        "targets.csv",
        "--identity",
        "lot,wafer,device",
        "--features",
        "x",
        "--group-prefix",
        "1",
        "--out",
        "prepared",
    ])
}
#[test]
fn prepares_canonical_rows_and_can_fit_predict() {
    let w = Workspace::new();
    setup(&w);
    let r = prepare(&w);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let future = fs::read_to_string(w.0.join("prepared/future.csv")).unwrap();
    assert!(!future.lines().next().unwrap().contains("target"));
    let r = w.run(&[
        "fit",
        "--csv",
        "prepared/training.csv",
        "--target",
        "target",
        "--features",
        "x",
        "--model",
        "mean",
        "--out",
        "model.json",
    ]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let r = w.run(&[
        "predict",
        "--artifact",
        "model.json",
        "--csv",
        "prepared/future.csv",
        "--out",
        "pred.csv",
    ]);
    assert!(r.status.success());
    assert!(fs::read_to_string(w.0.join("pred.csv"))
        .unwrap()
        .contains('3'));
    let j: serde_json::Value =
        serde_json::from_slice(&fs::read(w.0.join("prepared/preparation.json")).unwrap()).unwrap();
    assert_eq!(j["ml_fits"], 0);
    assert_eq!(
        j["mapping"][0]["identity"],
        serde_json::json!(["A", "W", "D"])
    );
    assert!(!prepare(&w).status.success());
}
#[test]
fn future_outcomes_and_missing_endpoints_rejected() {
    for csv in [
        "lot,wafer,device,target\nA,W,D,3\nB,W,D,5\n",
        "lot,wafer,device,target\n",
    ] {
        let w = Workspace::new();
        setup(&w);
        w.write("targets.csv", csv);
        assert!(!prepare(&w).status.success());
        assert!(!w.0.join("prepared").exists());
    }
}
#[test]
fn exact_endpoint_schema() {
    let w = Workspace::new();
    setup(&w);
    w.write("targets.csv", "lot,wafer,device,target,hidden\nA,W,D,3,5\n");
    assert!(!prepare(&w).status.success());
    assert!(!w.0.join("prepared").exists());
}
#[test]
fn duplicate_identity_rejected() {
    let w = Workspace::new();
    setup(&w);
    w.write(
        "features.csv",
        "lot,wafer,device,x\nA,W,D,1\nA,W,D,2\nB,W,D,3\n",
    );
    assert!(!prepare(&w).status.success());
    assert!(!w.0.join("prepared").exists());
}
#[test]
fn underflow_endpoint_not_silently_zero() {
    let w = Workspace::new();
    setup(&w);
    w.write("targets.csv", "lot,wafer,device,target\nA,W,D,1e-999\n");
    assert!(!prepare(&w).status.success());
    assert!(!w.0.join("prepared").exists());
}
#[test]
fn unsafe_header_rejected() {
    let w = Workspace::new();
    setup(&w);
    w.write("features.csv", "lot,wafer,device,=1+1\nA,W,D,1\nB,W,D,2\n");
    let r = w.run(&[
        "prepare-measurements",
        "--csv",
        "features.csv",
        "--partitions",
        "partitions.csv",
        "--training-targets",
        "targets.csv",
        "--identity",
        "lot,wafer,device",
        "--features",
        "=1+1",
        "--group-prefix",
        "1",
        "--out",
        "prepared",
    ]);
    assert!(!r.status.success());
    assert!(!w.0.join("prepared").exists());
}
#[test]
fn directory_input_rejected() {
    let w = Workspace::new();
    setup(&w);
    fs::remove_file(w.0.join("features.csv")).unwrap();
    fs::create_dir(w.0.join("features.csv")).unwrap();
    assert!(!prepare(&w).status.success());
    assert!(!w.0.join("prepared").exists());
}
#[cfg(unix)]
#[test]
fn fifo_input_rejected_before_open() {
    let w = Workspace::new();
    setup(&w);
    fs::remove_file(w.0.join("features.csv")).unwrap();
    let status = Command::new("mkfifo")
        .arg(w.0.join("features.csv"))
        .status()
        .unwrap();
    assert!(status.success());
    assert!(!prepare(&w).status.success());
    assert!(!w.0.join("prepared").exists());
}
