//! Native numeric regression workflow. No Python process or runtime is used.
#![forbid(unsafe_code)]
use clotho_core::evaluation::{grouped_kfold, regression_metrics};
use clotho_core::knn::KnnRegressor;
use clotho_core::regression::{MeanRegressor, RidgeRegressor};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const HELP: &str = "Clotho native numeric regression (experimental)\n\nfit --csv FILE --target NAME --features A,B --model knn|ridge|mean --out MODEL.json [--k 5] [--alpha 1]\npredict --artifact MODEL.json --csv FILE --out PREDICTIONS.csv\nevaluate --csv FILE --target NAME --features A,B --group NAME --model knn|ridge|mean --folds 5 --out NEW_DIRECTORY [--k 5] [--alpha 1]\n\nFeatures must be explicit finite numeric columns. Evaluation holds out whole groups.\nNo Python, runtime downloads, telemetry, automatic tuning or categorical encoding.\nOutputs must not already exist. Model files contain training data: keep them private.\n";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Kind {
    Knn,
    Ridge,
    Mean,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    schema_version: String,
    model: Kind,
    features: Vec<String>,
    target: String,
    k: usize,
    alpha: f64,
    training_x: Vec<Vec<f64>>,
    training_y: Vec<f64>,
}
enum Model {
    Knn(KnnRegressor),
    Ridge(RidgeRegressor),
    Mean(MeanRegressor),
}
impl Artifact {
    fn fit(&self) -> Result<Model> {
        if self.schema_version != "clotho.numeric.v1" {
            return Err("Unsupported model schema".into());
        }
        validate_features(&self.features, &self.target, None)?;
        if self
            .training_x
            .iter()
            .any(|r| r.len() != self.features.len())
        {
            return Err("Artifact feature width mismatch".into());
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err("alpha must be finite and nonnegative".into());
        }
        Ok(match self.model {
            Kind::Knn => Model::Knn(KnnRegressor::fit(
                &self.training_x,
                &self.training_y,
                self.k,
            )?),
            Kind::Ridge => Model::Ridge(RidgeRegressor::fit(
                &self.training_x,
                &self.training_y,
                self.alpha,
            )?),
            Kind::Mean => Model::Mean(MeanRegressor::fit(&self.training_x, &self.training_y)?),
        })
    }
}
impl Model {
    fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        Ok(match self {
            Self::Knn(m) => m.predict(x)?,
            Self::Ridge(m) => m.predict(x)?,
            Self::Mean(m) => m.predict(x)?,
        })
    }
}
struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}
fn read_bounded(path: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("Input exceeds 64 MiB limit".into());
    }
    Ok(bytes)
}
fn table(path: &str) -> Result<Table> {
    let bytes = read_bounded(path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(false)
        .from_reader(bytes.as_slice());
    let headers: Vec<String> = rdr.headers()?.iter().map(str::to_owned).collect();
    let mut seen = HashSet::new();
    if headers.is_empty()
        || headers
            .iter()
            .any(|s| s.is_empty() || s.trim() != s || s.contains('\0') || !seen.insert(s.clone()))
    {
        return Err(
            "CSV headers must be nonempty, unique and without surrounding whitespace or NUL".into(),
        );
    }
    let mut rows = Vec::new();
    for record in rdr.records() {
        let record = record?;
        if rows.len() >= 20_000 {
            return Err("CSV exceeds 20,000 row limit".into());
        }
        rows.push(record.iter().map(str::to_owned).collect());
    }
    if rows.is_empty() {
        return Err("CSV must contain data rows".into());
    }
    Ok(Table { headers, rows })
}
impl Table {
    fn column(&self, name: &str) -> Result<usize> {
        self.headers
            .iter()
            .position(|s| s == name)
            .ok_or_else(|| "Required CSV column is missing".into())
    }
    fn numeric(&self, names: &[String]) -> Result<Vec<Vec<f64>>> {
        let cols = names
            .iter()
            .map(|s| self.column(s))
            .collect::<Result<Vec<_>>>()?;
        self.rows
            .iter()
            .map(|row| {
                cols.iter()
                    .map(|&i| {
                        let value = row[i].parse::<f64>().map_err(|_| {
                            "Numeric columns must contain explicit numbers, without missing values"
                        })?;
                        if !value.is_finite() {
                            return Err("Numeric columns must be finite".into());
                        }
                        Ok(value)
                    })
                    .collect()
            })
            .collect()
    }
    fn target(&self, name: &str) -> Result<Vec<f64>> {
        Ok(self
            .numeric(&[name.to_owned()])?
            .into_iter()
            .map(|r| r[0])
            .collect())
    }
    fn groups(&self, name: &str) -> Result<Vec<String>> {
        let i = self.column(name)?;
        self.rows
            .iter()
            .map(|r| {
                if r[i].trim().is_empty() || r[i].contains('\0') {
                    Err("Group IDs must be nonempty and contain no NUL".into())
                } else {
                    Ok(r[i].clone())
                }
            })
            .collect()
    }
}
fn validate_features(features: &[String], target: &str, group: Option<&str>) -> Result<()> {
    let mut seen = HashSet::new();
    if features.is_empty()
        || features.len() > 100
        || target.trim().is_empty()
        || features.iter().any(|f| {
            f.is_empty()
                || f.trim() != f
                || f.contains('\0')
                || f == target
                || group == Some(f.as_str())
                || !seen.insert(f)
        })
    {
        return Err("Supply 1-100 unique feature names excluding target and group".into());
    }
    if group == Some(target) {
        return Err("Target and group must differ".into());
    }
    Ok(())
}
fn args() -> Result<(String, HashMap<String, String>)> {
    let mut a = std::env::args().skip(1);
    let cmd = a.next().unwrap_or_else(|| "--help".to_owned());
    let mut options = HashMap::new();
    while let Some(key) = a.next() {
        if !key.starts_with("--") || options.contains_key(&key) {
            return Err("Invalid or duplicate option".into());
        }
        let value = a.next().ok_or("Each option requires a value")?;
        if value.starts_with("--") {
            return Err("Each option requires a value".into());
        }
        options.insert(key, value);
    }
    Ok((cmd, options))
}
fn required<'a>(opts: &'a HashMap<String, String>, key: &str) -> Result<&'a str> {
    opts.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("Missing required option {key}").into())
}
fn number<T: std::str::FromStr>(
    opts: &HashMap<String, String>,
    key: &str,
    default: &str,
) -> Result<T> {
    opts.get(key)
        .map(String::as_str)
        .unwrap_or(default)
        .parse()
        .map_err(|_| format!("Invalid numeric option {key}").into())
}
fn create_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut f = options.open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
fn predictions_bytes(values: &[f64]) -> Result<Vec<u8>> {
    let mut w = csv::Writer::from_writer(Vec::new());
    w.write_record(["row_index", "prediction"])?;
    for (i, y) in values.iter().enumerate() {
        w.write_record([i.to_string(), y.to_string()])?;
    }
    Ok(w.into_inner()?)
}
fn run() -> Result<()> {
    let (cmd, opts) = args()?;
    if cmd == "--help" || cmd == "help" {
        print!("{HELP}");
        return Ok(());
    }
    if cmd == "--version" {
        println!("clotho native {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let allowed: &[&str] = match cmd.as_str() {
        "fit" => &[
            "--csv",
            "--target",
            "--features",
            "--model",
            "--out",
            "--k",
            "--alpha",
        ],
        "predict" => &["--artifact", "--csv", "--out"],
        "evaluate" => &[
            "--csv",
            "--target",
            "--features",
            "--model",
            "--out",
            "--k",
            "--alpha",
            "--group",
            "--folds",
        ],
        _ => return Err("Unknown command; use --help".into()),
    };
    if opts.keys().any(|k| !allowed.contains(&k.as_str())) {
        return Err("Unknown option for command".into());
    }
    let output = Path::new(required(&opts, "--out")?);
    if output.symlink_metadata().is_ok() {
        return Err("Output already exists; choose a new path".into());
    }
    let data = table(required(&opts, "--csv")?)?;
    if cmd == "predict" {
        let artifact: Artifact =
            serde_json::from_slice(&read_bounded(required(&opts, "--artifact")?)?)?;
        let model = artifact.fit()?;
        let x = data.numeric(&artifact.features)?;
        create_file(output, &predictions_bytes(&model.predict(&x)?)?)?;
        println!("Wrote {} numeric predictions", x.len());
        return Ok(());
    }
    let features: Vec<String> = required(&opts, "--features")?
        .split(',')
        .map(|s| s.trim().to_owned())
        .collect();
    let target = required(&opts, "--target")?.to_owned();
    let group = opts.get("--group").map(String::as_str);
    validate_features(&features, &target, group)?;
    let model = match required(&opts, "--model")? {
        "knn" => Kind::Knn,
        "ridge" => Kind::Ridge,
        "mean" => Kind::Mean,
        _ => return Err("Model must be knn, ridge or mean".into()),
    };
    let mut artifact = Artifact {
        schema_version: "clotho.numeric.v1".to_owned(),
        model,
        features: features.clone(),
        target: target.clone(),
        k: number(&opts, "--k", "5")?,
        alpha: number(&opts, "--alpha", "1")?,
        training_x: data.numeric(&features)?,
        training_y: data.target(&target)?,
    };
    if cmd == "fit" {
        artifact.fit()?;
        create_file(output, &serde_json::to_vec_pretty(&artifact)?)?;
        println!(
            "Wrote model specification and private training data ({} rows)",
            artifact.training_x.len()
        );
        return Ok(());
    }
    let groups = data.groups(required(&opts, "--group")?)?;
    let n_folds = number(&opts, "--folds", "5")?;
    if !(2..=20).contains(&n_folds) {
        return Err("CLI folds must be between 2 and 20".into());
    }
    let folds = grouped_kfold(&groups, n_folds)?;
    let full_x = std::mem::take(&mut artifact.training_x);
    let full_y = std::mem::take(&mut artifact.training_y);
    let mut pred = vec![0.0; full_y.len()];
    let mut assigned = vec![false; full_y.len()];
    let mut fold_ids = vec![0; full_y.len()];
    let mut fold_metrics = Vec::new();
    for (i, fold) in folds.iter().enumerate() {
        artifact.training_x = fold
            .train_indices
            .iter()
            .map(|&j| full_x[j].clone())
            .collect();
        artifact.training_y = fold.train_indices.iter().map(|&j| full_y[j]).collect();
        let model = artifact.fit()?;
        let test_x: Vec<_> = fold
            .test_indices
            .iter()
            .map(|&j| full_x[j].clone())
            .collect();
        let test_y: Vec<_> = fold.test_indices.iter().map(|&j| full_y[j]).collect();
        let p = model.predict(&test_x)?;
        let m = regression_metrics(&test_y, &p)?;
        fold_metrics.push(serde_json::json!({"fold":i,"n_train":fold.train_indices.len(),"n_test":fold.test_indices.len(),"mae":m.mae,"rmse":m.rmse,"r2":m.r2}));
        for (&j, &value) in fold.test_indices.iter().zip(&p) {
            if assigned[j] {
                return Err("Internal duplicate held-out assignment".into());
            }
            pred[j] = value;
            assigned[j] = true;
            fold_ids[j] = i;
        }
    }
    if assigned.iter().any(|done| !*done) {
        return Err("Internal incomplete held-out assignment".into());
    }
    let m = regression_metrics(&full_y, &pred)?;
    let mut group_rows: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        group_rows.entry(g.as_str()).or_default().push(i);
    }
    let mut group_metrics = Vec::new();
    let mut group_maes = Vec::new();
    let mut group_rmses = Vec::new();
    let mut worst_mae = 0.0_f64;
    for (name, indices) in &group_rows {
        let truth: Vec<_> = indices.iter().map(|&i| full_y[i]).collect();
        let predicted: Vec<_> = indices.iter().map(|&i| pred[i]).collect();
        let gm = regression_metrics(&truth, &predicted)?;
        group_maes.push(gm.mae);
        group_rmses.push(gm.rmse);
        worst_mae = worst_mae.max(gm.mae);
        group_metrics.push(serde_json::json!({"group":name,"n_rows":indices.len(),"mae":gm.mae,"rmse":gm.rmse,"r2":gm.r2}));
    }
    // MAE against zeros is a stable nonnegative mean, using core scaling and
    // compensation rather than dividing subnormal group values before summing.
    let zeros = vec![0.0; group_rows.len()];
    let macro_mae = regression_metrics(&zeros, &group_maes)?.mae;
    let macro_rmse = regression_metrics(&zeros, &group_rmses)?.mae;
    let metrics = serde_json::json!({"schema_version":"clotho.grouped_evaluation.v1","model":artifact.model,"features":features,"target":target,"group":group,"n_rows":full_y.len(),"n_groups":group_rows.len(),"n_folds":folds.len(),"k":artifact.k,"alpha":artifact.alpha,"selection":"fixed hyperparameters; no tuning","split":"largest groups first; least-loaded fold; lexical tie break","pooled_row_weighted":{"mae":m.mae,"rmse":m.rmse,"r2":m.r2},"equal_group_weighted":{"mean_group_mae":macro_mae,"mean_group_rmse":macro_rmse,"worst_group_mae":worst_mae},"groups":group_metrics,"folds":fold_metrics});
    let mut w = csv::Writer::from_writer(Vec::new());
    w.write_record(["row_index", "fold", "target", "prediction"])?;
    for i in 0..pred.len() {
        w.write_record([
            i.to_string(),
            fold_ids[i].to_string(),
            full_y[i].to_string(),
            pred[i].to_string(),
        ])?;
    }
    let csv = w.into_inner()?;
    let json = serde_json::to_vec_pretty(&metrics)?;
    // All parsing, training and serialization complete before publishing anything.
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(output)?;
    create_file(&output.join("predictions.csv"), &csv)?;
    create_file(&output.join("metrics.json"), &json)?;
    create_file(&output.join("COMPLETE"), b"clotho.grouped_evaluation.v1\n")?;
    println!(
        "Wrote {} out-of-fold predictions; MAE={:.6}, RMSE={:.6}",
        pred.len(),
        m.mae,
        m.rmse
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}
