//! Native numeric and explicitly mixed modelling. No Python runtime is used.
#![forbid(unsafe_code)]
mod fet_command;
mod prepare_command;
mod report_command;
mod rram_command;
use clotho_core::classification::{LogisticClassifier, MajorityClassifier};
use clotho_core::classification_metrics::{binary_metrics, BinaryMetrics};
use clotho_core::evaluation::{grouped_kfold, regression_metrics};
use clotho_core::input_support::{FeatureColumn, FeatureValues};
use clotho_core::knn::KnnRegressor;
use clotho_core::preprocessing::Preprocessor;
use clotho_core::regression::{MeanRegressor, RidgeRegressor};
use clotho_core::selection::{nested_grouped_regression, RegressionCandidate};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const HELP: &str = "Clotho native modelling (experimental)\n\nreport --run EVALUATION_DIRECTORY --out NEW_REPORT.html\nprepare-measurements --csv FILE --partitions FILE --training-targets FILE --identity lot,wafer,device --features A,B --group-prefix 1 --out NEW_DIRECTORY\nextract-rram --csv FILE --max-bracket-v V --accept-assumptions true --out NEW_DIRECTORY [--feature-row-unit device|device-condition]\nextract-fet --csv FILE --criterion-a A --polarity positive|negative --max-bracket-v V --window-min-v V --window-max-v V --accept-assumptions true --out NEW_DIRECTORY\nfit --csv FILE --target NAME --features A,B --model knn|ridge|mean|logistic|majority --out MODEL.json [--k 5] [--alpha 1] [--preprocess mixed] [--categorical A,B]\npredict --artifact MODEL.json --csv FILE --out PREDICTIONS.csv [--threshold 0.5]\nevaluate --csv FILE --target NAME --features A,B --group NAME --model knn|ridge|mean|logistic|majority --folds 5 --out NEW_DIRECTORY [--k 5] [--alpha 1] [--threshold 0.5] [--preprocess mixed] [--categorical A,B]\ncompare --csv FILE --target NAME --features A,B --group NAME --candidates mean,ridge:1,knn:3 --outer-folds 5 --inner-folds 3 --max-fits 200 --out NEW_DIRECTORY\n\nStrict finite numeric features are the default. Mixed mode imputes empty numeric cells\nand encodes explicitly named categorical features using training rows only. All other\nfeatures remain numeric; NA and null are literal categorical strings, not missing.\nEvaluation holds out whole groups and fits preprocessing inside each training fold.\nAt most 100 encoded features per estimator. No Python, downloads or telemetry.\nOutputs must not already exist. Model files contain private raw training data.\n";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Kind {
    Knn,
    Ridge,
    Mean,
    Logistic,
    Majority,
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
    /// Raw selected feature cells, in `features` order. Empty cells are missing.
    /// Presence selects mixed preprocessing, including all-numeric imputation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    training_rows: Option<Vec<Vec<String>>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    categorical_features: Vec<String>,
}
enum Model {
    Knn(KnnRegressor),
    Ridge(RidgeRegressor),
    Mean(MeanRegressor),
    Logistic(LogisticClassifier),
    Majority(MajorityClassifier),
}
/// Source rows are split before any preprocessing is fitted.
enum SourceRows {
    Numeric(Vec<Vec<f64>>),
    Mixed(Vec<Vec<String>>),
}
impl SourceRows {
    fn select(&self, indices: &[usize]) -> Self {
        match self {
            Self::Numeric(rows) => {
                Self::Numeric(indices.iter().map(|&i| rows[i].clone()).collect())
            }
            Self::Mixed(rows) => Self::Mixed(indices.iter().map(|&i| rows[i].clone()).collect()),
        }
    }
}
struct Fitted {
    model: Model,
    preprocessing: Option<Preprocessor>,
    features: Vec<String>,
    categorical_features: Vec<String>,
}
impl Fitted {
    fn feature_count(&self) -> usize {
        self.preprocessing
            .as_ref()
            .map_or(self.features.len(), |p| p.feature_names().len())
    }
    fn predict(&self, rows: &SourceRows) -> Result<Vec<f64>> {
        match (&self.preprocessing, rows) {
            (None, SourceRows::Numeric(x)) => self.model.predict(x),
            (Some(preprocessing), SourceRows::Mixed(rows)) => {
                let columns = mixed_columns(rows, &self.features, &self.categorical_features)?;
                self.model.predict(&preprocessing.transform(&columns)?)
            }
            _ => Err("Prediction preprocessing mode differs from training".into()),
        }
    }
}
impl Artifact {
    fn is_binary(&self) -> bool {
        matches!(self.model, Kind::Logistic | Kind::Majority)
    }
    fn preprocessing_mode(&self) -> &'static str {
        if self.training_rows.is_some() {
            "mixed"
        } else {
            "numeric"
        }
    }
    fn take_rows(&mut self) -> SourceRows {
        match self.training_rows.take() {
            Some(rows) => SourceRows::Mixed(rows),
            None => SourceRows::Numeric(std::mem::take(&mut self.training_x)),
        }
    }
    fn set_rows(&mut self, rows: SourceRows) {
        match rows {
            SourceRows::Numeric(rows) => {
                self.training_x = rows;
                self.training_rows = None;
            }
            SourceRows::Mixed(rows) => {
                self.training_x.clear();
                self.training_rows = Some(rows);
            }
        }
    }
    fn fit(&self) -> Result<Fitted> {
        if ![
            "clotho.numeric.v1",
            "clotho.numeric.v2",
            "clotho.numeric.v3",
        ]
        .contains(&self.schema_version.as_str())
            || (self.schema_version == "clotho.numeric.v1" && self.is_binary())
        {
            return Err("Unsupported model schema".into());
        }
        validate_features(&self.features, &self.target, None)?;
        validate_categorical(
            &self.features,
            &self.categorical_features,
            self.training_rows.is_some(),
        )?;
        if self.schema_version != "clotho.numeric.v3" && self.training_rows.is_some() {
            return Err("Mixed preprocessing requires model schema clotho.numeric.v3".into());
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err("alpha must be finite and nonnegative".into());
        }
        let preprocessing;
        let transformed;
        let x = if let Some(rows) = &self.training_rows {
            if !self.training_x.is_empty() {
                return Err(
                    "Mixed artifacts must contain raw rows only, not training_x values".into(),
                );
            }
            let columns = mixed_columns(rows, &self.features, &self.categorical_features)?;
            let fitted = Preprocessor::fit(&columns)?;
            if fitted.feature_names().len() > 100 {
                return Err(
                    "Mixed preprocessing exceeds the estimator limit of 100 encoded features"
                        .into(),
                );
            }
            transformed = fitted.transform(&columns)?;
            preprocessing = Some(fitted);
            &transformed
        } else {
            if self
                .training_x
                .iter()
                .any(|r| r.len() != self.features.len())
            {
                return Err("Artifact feature width mismatch".into());
            }
            preprocessing = None;
            &self.training_x
        };
        let model = match self.model {
            Kind::Knn => Model::Knn(KnnRegressor::fit(x, &self.training_y, self.k)?),
            Kind::Ridge => Model::Ridge(RidgeRegressor::fit(x, &self.training_y, self.alpha)?),
            Kind::Mean => Model::Mean(MeanRegressor::fit(x, &self.training_y)?),
            Kind::Logistic => {
                Model::Logistic(LogisticClassifier::fit(x, &self.training_y, self.alpha)?)
            }
            Kind::Majority => Model::Majority(MajorityClassifier::fit(x, &self.training_y)?),
        };
        Ok(Fitted {
            model,
            preprocessing,
            features: self.features.clone(),
            categorical_features: self.categorical_features.clone(),
        })
    }
}
impl Model {
    fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        Ok(match self {
            Self::Knn(m) => m.predict(x)?,
            Self::Ridge(m) => m.predict(x)?,
            Self::Mean(m) => m.predict(x)?,
            Self::Logistic(m) => m.predict_proba(x)?,
            Self::Majority(m) => m.predict_proba(x)?,
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
    fn feature_rows(&self, artifact: &Artifact) -> Result<SourceRows> {
        if artifact.training_rows.is_some() {
            Ok(SourceRows::Mixed(self.raw_features(&artifact.features)?))
        } else {
            Ok(SourceRows::Numeric(self.numeric(&artifact.features)?))
        }
    }
    fn raw_features(&self, names: &[String]) -> Result<Vec<Vec<String>>> {
        let columns = names
            .iter()
            .map(|name| self.column(name))
            .collect::<Result<Vec<_>>>()?;
        Ok(self
            .rows
            .iter()
            .map(|row| columns.iter().map(|&i| row[i].clone()).collect())
            .collect())
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
fn validate_categorical(features: &[String], categorical: &[String], mixed: bool) -> Result<()> {
    if !mixed && !categorical.is_empty() {
        return Err("categorical features require --preprocess mixed".into());
    }
    let mut seen = HashSet::new();
    if categorical.iter().any(|name| {
        name.is_empty()
            || name.trim() != name
            || name.contains('\0')
            || !features.contains(name)
            || !seen.insert(name)
    }) {
        return Err(
            "Categorical names must be unique, nonempty members of the selected features".into(),
        );
    }
    Ok(())
}
fn preprocessing_options(
    opts: &HashMap<String, String>,
    features: &[String],
) -> Result<(bool, Vec<String>)> {
    let mixed = match opts.get("--preprocess").map(String::as_str) {
        None => false,
        Some("mixed") => true,
        _ => return Err("preprocess must be mixed; omit it for strict numeric mode".into()),
    };
    if opts.contains_key("--categorical") && !mixed {
        return Err("categorical features require --preprocess mixed".into());
    }
    let categorical: Vec<String> = opts
        .get("--categorical")
        .map(|names| names.split(',').map(|s| s.trim().to_owned()).collect())
        .unwrap_or_default();
    validate_categorical(features, &categorical, mixed)?;
    Ok((mixed, categorical))
}
fn mixed_columns(
    rows: &[Vec<String>],
    features: &[String],
    categorical: &[String],
) -> Result<Vec<FeatureColumn>> {
    if rows.iter().any(|row| row.len() != features.len()) {
        return Err("Artifact feature width mismatch".into());
    }
    features.iter().enumerate().map(|(i, name)| {
        let values = if categorical.contains(name) {
            FeatureValues::Categorical(rows.iter().map(|row| {
                if row[i].is_empty() { None } else { Some(row[i].clone()) }
            }).collect())
        } else {
            FeatureValues::Numeric(rows.iter().map(|row| {
                if row[i].is_empty() { return Ok(None); }
                let value = row[i].parse::<f64>().map_err(|_| "Mixed numeric columns must contain explicit numbers or empty cells; declare string columns with --categorical")?;
                if !value.is_finite() { return Err("Numeric columns must be finite".into()); }
                Ok(Some(value))
            }).collect::<Result<Vec<_>>>()?)
        };
        Ok(FeatureColumn { name: name.clone(), values })
    }).collect()
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
    if cmd == "report" {
        return report_command::run(&opts);
    }
    if cmd == "extract-rram" {
        return rram_command::run(&opts);
    }
    if cmd == "prepare-measurements" {
        return prepare_command::run(&opts);
    }
    if cmd == "extract-fet" {
        return fet_command::run(&opts);
    }
    if cmd == "--help" || cmd == "help" {
        print!("{HELP}");
        return Ok(());
    }
    if cmd == "--version" {
        println!("clotho native {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let allowed: &[&str] = match cmd.as_str() {
        "compare" => &[
            "--csv",
            "--target",
            "--features",
            "--group",
            "--candidates",
            "--outer-folds",
            "--inner-folds",
            "--max-fits",
            "--out",
        ],
        "fit" => &[
            "--csv",
            "--target",
            "--features",
            "--model",
            "--out",
            "--k",
            "--alpha",
            "--preprocess",
            "--categorical",
        ],
        "predict" => &["--artifact", "--csv", "--out", "--threshold"],
        "evaluate" => &[
            "--csv",
            "--target",
            "--features",
            "--model",
            "--out",
            "--k",
            "--alpha",
            "--preprocess",
            "--categorical",
            "--group",
            "--folds",
            "--threshold",
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
    if cmd == "compare" {
        return compare_regression(output, &data, &opts);
    }
    if cmd == "predict" {
        let artifact: Artifact =
            serde_json::from_slice(&read_bounded(required(&opts, "--artifact")?)?).map_err(
                |error| {
                    // Serde's Display may include private raw values or field names.
                    format!(
                        "Invalid model JSON at line {}, column {}",
                        error.line(),
                        error.column()
                    )
                },
            )?;
        let model = artifact.fit()?;
        let x = data.feature_rows(&artifact)?;
        let probabilities = model.predict(&x)?;
        if artifact.is_binary() {
            let threshold = threshold_option(&opts)?;
            let mut w = csv::Writer::from_writer(Vec::new());
            w.write_record(["row_index", "positive_probability", "predicted_label"])?;
            for (i, p) in probabilities.iter().enumerate() {
                w.write_record([
                    i.to_string(),
                    p.to_string(),
                    usize::from(*p >= threshold).to_string(),
                ])?;
            }
            create_file(output, &w.into_inner()?)?;
        } else {
            if opts.contains_key("--threshold") {
                return Err("threshold applies only to binary models".into());
            }
            create_file(output, &predictions_bytes(&probabilities)?)?;
        }
        println!("Wrote {} predictions", probabilities.len());
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
        "logistic" => Kind::Logistic,
        "majority" => Kind::Majority,
        _ => return Err("Model must be knn, ridge, mean, logistic or majority".into()),
    };
    let (mixed, categorical_features) = preprocessing_options(&opts, &features)?;
    let mut artifact = Artifact {
        schema_version: "clotho.numeric.v3".to_owned(),
        model,
        features: features.clone(),
        target: target.clone(),
        k: number(&opts, "--k", "5")?,
        alpha: number(&opts, "--alpha", "1")?,
        training_x: if mixed {
            Vec::new()
        } else {
            data.numeric(&features)?
        },
        training_y: data.target(&target)?,
        training_rows: if mixed {
            Some(data.raw_features(&features)?)
        } else {
            None
        },
        categorical_features,
    };
    if cmd == "fit" {
        artifact.fit()?;
        create_file(output, &serde_json::to_vec_pretty(&artifact)?)?;
        println!(
            "Wrote model specification and private training data ({} rows)",
            artifact.training_y.len()
        );
        return Ok(());
    }
    let groups = data.groups(required(&opts, "--group")?)?;
    let n_folds = number(&opts, "--folds", "5")?;
    if !(2..=20).contains(&n_folds) {
        return Err("CLI folds must be between 2 and 20".into());
    }
    let folds = grouped_kfold(&groups, n_folds)?;
    let full_x = artifact.take_rows();
    let full_y = std::mem::take(&mut artifact.training_y);
    if artifact.is_binary() {
        return evaluate_binary(
            output,
            artifact,
            &full_x,
            &full_y,
            GroupContext {
                name: required(&opts, "--group")?,
                values: &groups,
            },
            &folds,
            threshold_option(&opts)?,
        );
    }
    if opts.contains_key("--threshold") {
        return Err("threshold applies only to binary models".into());
    }
    let mut pred = vec![0.0; full_y.len()];
    let mut assigned = vec![false; full_y.len()];
    let mut fold_ids = vec![0; full_y.len()];
    let mut fold_metrics = Vec::new();
    for (i, fold) in folds.iter().enumerate() {
        artifact.set_rows(full_x.select(&fold.train_indices));
        artifact.training_y = fold.train_indices.iter().map(|&j| full_y[j]).collect();
        let model = artifact.fit()?;
        let test_x = full_x.select(&fold.test_indices);
        let test_y: Vec<_> = fold.test_indices.iter().map(|&j| full_y[j]).collect();
        let p = model.predict(&test_x)?;
        let m = regression_metrics(&test_y, &p)?;
        fold_metrics.push(serde_json::json!({"fold":i,"n_train":fold.train_indices.len(),"n_test":fold.test_indices.len(),"encoded_feature_count":model.feature_count(),"mae":m.mae,"rmse":m.rmse,"r2":m.r2}));
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
    let metrics = serde_json::json!({"schema_version":"clotho.grouped_evaluation.v1","preprocess":artifact.preprocessing_mode(),"categorical_features":artifact.categorical_features,"model":artifact.model,"features":features,"target":target,"group":group,"n_rows":full_y.len(),"n_groups":group_rows.len(),"n_folds":folds.len(),"k":artifact.k,"alpha":artifact.alpha,"selection":"fixed hyperparameters; no tuning","split":"largest groups first; least-loaded fold; lexical tie break","pooled_row_weighted":{"mae":m.mae,"rmse":m.rmse,"r2":m.r2},"equal_group_weighted":{"mean_group_mae":macro_mae,"mean_group_rmse":macro_rmse,"worst_group_mae":worst_mae},"groups":group_metrics,"folds":fold_metrics});
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

fn threshold_option(opts: &HashMap<String, String>) -> Result<f64> {
    let t: f64 = number(opts, "--threshold", "0.5")?;
    if !t.is_finite() || !(0.0..=1.0).contains(&t) {
        return Err("threshold must be finite and between 0 and 1".into());
    }
    Ok(t)
}
fn binary_json(m: &BinaryMetrics) -> serde_json::Value {
    serde_json::json!({"accuracy":m.accuracy,"balanced_accuracy":m.balanced_accuracy,"precision":m.precision,"recall":m.recall,"f1":m.f1,"log_loss":m.log_loss,"brier":m.brier,"roc_auc":m.roc_auc,"n_positive":m.n_positive,"n_negative":m.n_negative,"tp":m.tp,"fp":m.fp,"tn":m.tn,"fn":m.fn_})
}
struct GroupContext<'a> {
    name: &'a str,
    values: &'a [String],
}
fn evaluate_binary(
    output: &Path,
    mut artifact: Artifact,
    x: &SourceRows,
    y: &[f64],
    group_context: GroupContext<'_>,
    folds: &[clotho_core::evaluation::Fold],
    threshold: f64,
) -> Result<()> {
    let group = group_context.name;
    let groups = group_context.values;
    let mut pred = vec![0.0; y.len()];
    let mut assigned = vec![false; y.len()];
    let mut fold_ids = vec![0; y.len()];
    let mut fold_metrics = Vec::new();
    for (i, fold) in folds.iter().enumerate() {
        artifact.set_rows(x.select(&fold.train_indices));
        artifact.training_y = fold.train_indices.iter().map(|&j| y[j]).collect();
        let model = artifact.fit()?;
        let tx = x.select(&fold.test_indices);
        let ty: Vec<_> = fold.test_indices.iter().map(|&j| y[j]).collect();
        let p = model.predict(&tx)?;
        let metrics = binary_metrics(&ty, &p, threshold)?;
        fold_metrics.push(serde_json::json!({"fold":i,"n_train":fold.train_indices.len(),"n_test":fold.test_indices.len(),"encoded_feature_count":model.feature_count(),"metrics":binary_json(&metrics)}));
        for (&j, &value) in fold.test_indices.iter().zip(&p) {
            if assigned[j] {
                return Err("Internal duplicate held-out assignment".into());
            }
            pred[j] = value;
            assigned[j] = true;
            fold_ids[j] = i;
        }
    }
    if assigned.iter().any(|a| !*a) {
        return Err("Internal incomplete held-out assignment".into());
    }
    let metrics = binary_metrics(y, &pred, threshold)?;
    let mut group_rows: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        group_rows.entry(g.as_str()).or_default().push(i);
    }
    let mut group_metrics = Vec::new();
    let mut group_losses = Vec::new();
    let mut group_accuracies = Vec::new();
    let mut balanced = Vec::new();
    for (group, indices) in &group_rows {
        let gy: Vec<_> = indices.iter().map(|&i| y[i]).collect();
        let gp: Vec<_> = indices.iter().map(|&i| pred[i]).collect();
        let m = binary_metrics(&gy, &gp, threshold)?;
        group_losses.push(m.log_loss);
        group_accuracies.push(m.accuracy);
        if let Some(v) = m.balanced_accuracy {
            balanced.push(v);
        }
        group_metrics.push(
            serde_json::json!({"group":group,"n_rows":indices.len(),"metrics":binary_json(&m)}),
        );
    }
    let zeros = vec![0.0; group_rows.len()];
    let mean_loss = regression_metrics(&zeros, &group_losses)?.mae;
    let mean_accuracy = regression_metrics(&zeros, &group_accuracies)?.mae;
    let mean_balanced = if balanced.is_empty() {
        None
    } else {
        Some(regression_metrics(&vec![0.0; balanced.len()], &balanced)?.mae)
    };
    let report = serde_json::json!({"schema_version":"clotho.grouped_binary_evaluation.v1","preprocess":artifact.preprocessing_mode(),"categorical_features":artifact.categorical_features,"group":group,"model":artifact.model,"features":artifact.features,"target":artifact.target,"n_rows":y.len(),"n_groups":group_rows.len(),"n_folds":folds.len(),"alpha":artifact.alpha,"threshold":threshold,"positive_label":1,"selection":"fixed hyperparameters and threshold; no tuning","split":"largest groups first; least-loaded fold; lexical tie break; not class stratified","pooled_row_weighted":binary_json(&metrics),"equal_group_weighted":{"mean_group_log_loss":mean_loss,"mean_group_accuracy":mean_accuracy,"mean_defined_group_balanced_accuracy":mean_balanced,"balanced_accuracy_group_denominator":balanced.len()},"folds":fold_metrics,"groups":group_metrics});
    let mut w = csv::Writer::from_writer(Vec::new());
    w.write_record([
        "row_index",
        "fold",
        "target",
        "positive_probability",
        "predicted_label",
    ])?;
    for i in 0..y.len() {
        w.write_record([
            i.to_string(),
            fold_ids[i].to_string(),
            y[i].to_string(),
            pred[i].to_string(),
            usize::from(pred[i] >= threshold).to_string(),
        ])?;
    }
    let csv = w.into_inner()?;
    let json = serde_json::to_vec_pretty(&report)?;
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(output)?;
    create_file(&output.join("predictions.csv"), &csv)?;
    create_file(&output.join("metrics.json"), &json)?;
    create_file(
        &output.join("COMPLETE"),
        b"clotho.grouped_binary_evaluation.v1\n",
    )?;
    println!(
        "Wrote {} held-out binary probabilities; log loss={:.6}, Brier={:.6}",
        y.len(),
        metrics.log_loss,
        metrics.brier
    );
    Ok(())
}

fn candidate_json(candidate: &RegressionCandidate) -> serde_json::Value {
    match candidate {
        RegressionCandidate::Mean => serde_json::json!({"model":"mean"}),
        RegressionCandidate::Ridge { alpha } => serde_json::json!({"model":"ridge","alpha":alpha}),
        RegressionCandidate::Knn { k } => serde_json::json!({"model":"knn","k":k}),
    }
}
fn compare_regression(output: &Path, data: &Table, opts: &HashMap<String, String>) -> Result<()> {
    let features: Vec<String> = required(opts, "--features")?
        .split(',')
        .map(|s| s.trim().to_owned())
        .collect();
    let target = required(opts, "--target")?;
    let group = required(opts, "--group")?;
    validate_features(&features, target, Some(group))?;
    let mut candidates = Vec::new();
    for token in required(opts, "--candidates")?.split(',').map(str::trim) {
        let candidate = if token == "mean" {
            RegressionCandidate::Mean
        } else if let Some(alpha) = token.strip_prefix("ridge:") {
            RegressionCandidate::Ridge {
                alpha: alpha.parse().map_err(|_| "Invalid ridge candidate alpha")?,
            }
        } else if let Some(k) = token.strip_prefix("knn:") {
            RegressionCandidate::Knn {
                k: k.parse().map_err(|_| "Invalid kNN candidate k")?,
            }
        } else {
            return Err("Candidates must be mean, ridge:ALPHA or knn:K".into());
        };
        candidates.push(candidate);
    }
    let x = data.numeric(&features)?;
    let y = data.target(target)?;
    let groups = data.groups(group)?;
    let outer = number(opts, "--outer-folds", "5")?;
    let inner = number(opts, "--inner-folds", "3")?;
    let max_fits = number(opts, "--max-fits", "200")?;
    let report = nested_grouped_regression(&x, &y, &groups, outer, inner, &candidates, max_fits)?;
    let mut fold_ids = vec![0; y.len()];
    let mut folds = Vec::new();
    for (i, fold) in report.outer_folds.iter().enumerate() {
        for &row in &fold.test_indices {
            fold_ids[row] = i;
        }
        let scores:Vec<_>=fold.inner_candidate_scores.iter().map(|score|{
            let failure=score.failure.as_ref().map(|f|serde_json::json!({"inner_fold":f.inner_fold_index,"stage":format!("{:?}",f.stage),"error":f.error.to_string()}));
            serde_json::json!({"candidate_index":score.candidate_index,"pooled_inner_rmse":score.pooled_rmse,"failure":failure,"attempted_fits":score.fit_count})
        }).collect();
        let truth: Vec<_> = fold.test_indices.iter().map(|&j| y[j]).collect();
        let pred: Vec<_> = fold
            .test_indices
            .iter()
            .map(|&j| report.out_of_fold_predictions[j])
            .collect();
        let m = regression_metrics(&truth, &pred)?;
        folds.push(serde_json::json!({"outer_fold":i,"n_train":fold.train_indices.len(),"n_test":fold.test_indices.len(),"selected_candidate_index":fold.selected_candidate_index,"selected_candidate":candidate_json(&candidates[fold.selected_candidate_index]),"inner_candidate_scores":scores,"attempted_fits":fold.fit_count,"outer_metrics":{"mae":m.mae,"rmse":m.rmse,"r2":m.r2}}));
    }
    let m = &report.pooled_metrics;
    let manifest = serde_json::json!({"schema_version":"clotho.nested_regression.v1","features":features,"target":target,"group":group,"n_rows":y.len(),"n_groups":groups.iter().collect::<HashSet<_>>().len(),"outer_folds":outer,"inner_folds":inner,"candidates":candidates.iter().map(candidate_json).collect::<Vec<_>>(),"selection":"pooled inner out-of-fold RMSE; exact ties first candidate","preprocessing":"strict numeric; scaling fitted independently by each estimator","split":"deterministic group-disjoint largest-first","max_fits":max_fits,"planned_fits":report.planned_fit_count,"attempted_fits":report.fit_count,"pooled_outer_metrics":{"mae":m.mae,"rmse":m.rmse,"r2":m.r2},"folds":folds});
    let mut w = csv::Writer::from_writer(Vec::new());
    w.write_record(["row_index", "outer_fold", "target", "prediction"])?;
    for i in 0..y.len() {
        w.write_record([
            i.to_string(),
            fold_ids[i].to_string(),
            y[i].to_string(),
            report.out_of_fold_predictions[i].to_string(),
        ])?;
    }
    let csv = w.into_inner()?;
    let json = serde_json::to_vec_pretty(&manifest)?;
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(output)?;
    create_file(&output.join("predictions.csv"), &csv)?;
    create_file(&output.join("selection.json"), &json)?;
    create_file(&output.join("COMPLETE"), b"clotho.nested_regression.v1\n")?;
    println!(
        "Wrote {} nested held-out predictions; {} of {} planned fits attempted",
        y.len(),
        report.fit_count,
        report.planned_fit_count
    );
    Ok(())
}
