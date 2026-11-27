//! Validate native grouped evaluation bundles and expose aggregates only.
//! Pooled and fold metrics are recomputed from the CSV. Group metrics can
//! only be checked for internal consistency: v1 CSV has no group row keys.
//! These are stable-file checks, not protection from concurrent replacement,
//! producer authentication, or proof of scientific group independence.
use crate::{numeric_json::reject_numeric_underflow, Result, MAX_BYTES, MAX_ESTIMATOR_FEATURES};
use clotho_core::{
    classification_metrics::{binary_metrics, BinaryMetrics, LOG_LOSS_EPSILON},
    evaluation::{regression_metrics, RegressionMetrics, MAX_GROUPED_ROWS, MAX_GROUPED_SPLITS},
};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};
const REG: &str = "clotho.grouped_evaluation.v1";
const BIN: &str = "clotho.grouped_binary_evaluation.v1";
const SCHEMA: &str = "clotho.grouped_evaluation_projection.v1";

#[derive(Serialize)]
pub(crate) struct EvaluationProjection {
    schema_version: &'static str,
    n_rows: usize,
    n_groups: usize,
    n_folds: usize,
    #[serde(flatten)]
    task: Aggregate,
}

#[derive(Serialize)]
#[serde(tag = "task", rename_all = "snake_case")]
enum Aggregate {
    Regression {
        pooled_row_weighted: Reg,
        equal_group_weighted: RegMacro,
        folds: Vec<FoldProjection<Reg>>,
    },
    Binary {
        threshold: f64,
        pooled_row_weighted: Bin,
        equal_group_weighted: BinMacro,
        folds: Vec<FoldProjection<Bin>>,
    },
}

#[derive(Serialize)]
struct FoldProjection<M> {
    fold: usize,
    n_train: usize,
    n_test: usize,
    encoded_feature_count: usize,
    metrics: M,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Reg {
    mae: f64,
    rmse: f64,
    #[serde(deserialize_with = "nullable")]
    r2: Option<f64>,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Bin {
    accuracy: f64,
    #[serde(deserialize_with = "nullable")]
    balanced_accuracy: Option<f64>,
    #[serde(deserialize_with = "nullable")]
    precision: Option<f64>,
    #[serde(deserialize_with = "nullable")]
    recall: Option<f64>,
    #[serde(deserialize_with = "nullable")]
    f1: Option<f64>,
    log_loss: f64,
    brier: f64,
    #[serde(deserialize_with = "nullable")]
    roc_auc: Option<f64>,
    n_positive: usize,
    n_negative: usize,
    tp: usize,
    fp: usize,
    tn: usize,
    #[serde(rename = "fn")]
    fn_: usize,
}

#[derive(Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RegMacro {
    mean_group_mae: f64,
    mean_group_rmse: f64,
    worst_group_mae: f64,
}

#[derive(Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct BinMacro {
    mean_group_log_loss: f64,
    mean_group_accuracy: f64,
    #[serde(deserialize_with = "nullable")]
    mean_defined_group_balanced_accuracy: Option<f64>,
    balanced_accuracy_group_denominator: usize,
}

fn nullable<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<f64>, D::Error> {
    Option::deserialize(d)
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ProducerModel { Knn, Ridge, Mean, Logistic, Majority }

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ProducerPreprocess { Numeric, Mixed }


#[derive(Deserialize)]
struct Manifest<M, G, F, A> {
    schema_version: String,
    model: ProducerModel,
    preprocess: ProducerPreprocess,
    features: Vec<String>,
    categorical_features: Vec<String>,
    target: String,
    group: String,
    n_rows: usize,
    n_groups: usize,
    n_folds: usize,
    pooled_row_weighted: M,
    equal_group_weighted: A,
    groups: Vec<G>,
    folds: Vec<F>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegGroup {
    group: String,
    n_rows: usize,
    mae: f64,
    rmse: f64,
    #[serde(deserialize_with = "nullable")]
    r2: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BinGroup {
    group: String,
    n_rows: usize,
    metrics: Bin,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegFold {
    fold: usize,
    n_train: usize,
    n_test: usize,
    encoded_feature_count: usize,
    mae: f64,
    rmse: f64,
    #[serde(deserialize_with = "nullable")]
    r2: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BinFold {
    fold: usize,
    n_train: usize,
    n_test: usize,
    encoded_feature_count: usize,
    metrics: Bin,
}

#[derive(Deserialize)]
struct BinaryHeader {
    threshold: f64,
    positive_label: usize,
}

fn require(ok: bool, message: &'static str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}

fn read_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
    // symlink_metadata also rejects FIFOs and other special paths before open.
    let check = || fs::symlink_metadata(path).map(|m| m.is_file() && m.len() <= limit);
    require(check().unwrap_or(false), "Expected a bounded regular evaluation file")?;
    let file = File::open(path).map_err(|_| "Cannot open evaluation file")?;
    require(
        file.metadata().map(|m| m.is_file()).unwrap_or(false) && check().unwrap_or(false),
        "Expected a bounded regular evaluation file",
    )?;
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes).map_err(|_| "Cannot read evaluation file")?;
    require(bytes.len() as u64 <= limit, "Evaluation file exceeds the size limit")?;
    Ok(bytes)
}

pub(crate) fn load(bundle: &Path) -> Result<EvaluationProjection> {
    // Collect components before checking the path so a trailing slash or
    // terminal '.' cannot make symlink_metadata follow a directory link.
    let bundle: PathBuf = bundle.components().collect();
    require(!bundle.as_os_str().is_empty(), "Expected an evaluation bundle directory")?;
    for ancestor in bundle.ancestors().filter(|p| !p.as_os_str().is_empty()) {
        require(
            fs::symlink_metadata(ancestor).map(|m| m.is_dir()).unwrap_or(false),
            "Expected a real evaluation bundle directory",
        )?;
    }
    let marker = read_file(&bundle.join("COMPLETE"), 128)?;
    let kind = match marker.as_slice() {
        b"clotho.grouped_evaluation.v1\n" => REG,
        b"clotho.grouped_binary_evaluation.v1\n" => BIN,
        _ => return Err("Unsupported evaluation completion marker".into()),
    };
    let json = read_file(&bundle.join("metrics.json"), MAX_BYTES)?;
    reject_numeric_underflow(&json)?;
    let csv = read_file(&bundle.join("predictions.csv"), MAX_BYTES)?;
    if kind == REG {
        let manifest = decode(&json)?;
        project_reg(manifest, &csv)
    } else {
        let header: BinaryHeader = decode(&json)?;
        let manifest = decode(&json)?;
        project_bin(manifest, header, &csv)
    }
}

fn decode<T: serde::de::DeserializeOwned>(json: &[u8]) -> Result<T> {
    serde_json::from_slice(json).map_err(|_| "Invalid evaluation metrics JSON".into())
}

pub(super) fn run(opts: &HashMap<String, String>) -> Result<()> {
    require(opts.len() == 1 && opts.contains_key("--bundle"), "Expected --bundle DIRECTORY")?;
    let projection = load(Path::new(&opts["--bundle"]))?;
    let mut output = serde_json::to_vec_pretty(&projection)?;
    output.push(b'\n');
    io::stdout().lock().write_all(&output)?;
    Ok(())
}

struct Rows {
    observed: Vec<f64>,
    predicted: Vec<f64>,
    folds: Vec<(Vec<f64>, Vec<f64>)>,
}

fn parse_rows(bytes: &[u8], rows: usize, folds: usize, threshold: Option<f64>) -> Result<Rows> {
    require(crate::device_csv::has_complete_csv_quoting(bytes), "Malformed evaluation CSV framing")?;
    let mut csv = csv::Reader::from_reader(bytes);
    let binary = threshold.is_some();
    let expected: &[&str] = if binary {
        &["row_index", "fold", "target", "positive_probability", "predicted_label"]
    } else {
        &["row_index", "fold", "target", "prediction"]
    };
    require(
        csv.headers().map_err(|_| "Invalid evaluation CSV header")?.iter().eq(expected.iter().copied()),
        "Unexpected evaluation CSV header",
    )?;
    let mut data = Rows { observed: Vec::with_capacity(rows), predicted: Vec::with_capacity(rows),
        folds: (0..folds).map(|_| (Vec::new(), Vec::new())).collect() };
    for (i, record) in csv.records().enumerate() {
        let record = record.map_err(|_| "Invalid evaluation CSV record")?;
        let idx: usize = record[0].parse().map_err(|_| "Invalid evaluation row index")?;
        let fold: usize = record[1].parse().map_err(|_| "Invalid evaluation fold index")?;
        require(i < rows && idx == i && fold < folds, "Inconsistent evaluation row or fold index")?;
        let actual = number(&record[2])?;
        let predicted = number(&record[3])?;
        if let Some(threshold) = threshold {
            require(
                (actual == 0.0 || actual == 1.0) && (0.0..=1.0).contains(&predicted)
                && &record[4] == if predicted >= threshold { "1" } else { "0" },
                "Invalid binary evaluation row",
            )?;
        }
        data.observed.push(actual);
        data.predicted.push(predicted);
        data.folds[fold].0.push(actual);
        data.folds[fold].1.push(predicted);
    }
    require(data.observed.len() == rows, "Inconsistent evaluation row count")?;
    Ok(data)
}

fn number(cell: &str) -> Result<f64> {
    let value: f64 = cell.parse().map_err(|_| "Invalid evaluation number")?;
    require(value.is_finite(), "Invalid evaluation number")?;
    // A CSV producer emits binary64 values; never accept invisible nonzero→0.
    if value == 0.0 {
        let mantissa = cell.split(['e', 'E']).next().unwrap_or("");
        require(!mantissa.bytes().any(|v| matches!(v, b'1'..=b'9')), "Invalid evaluation number")?;
    }
    Ok(value)
}

fn check_manifest<M, G, F, A>(m: &Manifest<M, G, F, A>, schema: &str) -> Result<()> {
    require(m.schema_version == schema, "Evaluation marker and metrics schema disagree")?;
    let mixed = matches!(m.preprocess, ProducerPreprocess::Mixed);
    crate::validate_features(&m.features, &m.target, Some(&m.group))
        .map_err(|_| "Invalid evaluation feature declaration")?;
    crate::validate_categorical(&m.features, &m.categorical_features, mixed)
        .map_err(|_| "Invalid evaluation categorical declaration")?;
    let binary_model = matches!(m.model, ProducerModel::Logistic | ProducerModel::Majority);
    require(binary_model == (schema == BIN), "Evaluation model and schema disagree")?;
    require(
        (2..=MAX_GROUPED_ROWS).contains(&m.n_rows)
        && (2..=m.n_rows).contains(&m.n_groups)
        && (2..=m.n_groups.min(MAX_GROUPED_SPLITS)).contains(&m.n_folds)
        && m.groups.len() == m.n_groups && m.folds.len() == m.n_folds,
        "Inconsistent evaluation counts",
    )
}

fn check_encoded_width<M, G, F, A>(
    m: &Manifest<M, G, F, A>, encoded: usize,
) -> Result<()> {
    let declared = m.features.len();
    let plausible = if m.categorical_features.is_empty() {
        encoded == declared
    } else {
        (declared..=MAX_ESTIMATOR_FEATURES).contains(&encoded)
    };
    require(plausible, "Evaluation encoded width contradicts the feature declaration")
}

fn projection<M, G, F, A>(m: &Manifest<M, G, F, A>, task: Aggregate) -> EvaluationProjection {
    EvaluationProjection { schema_version: SCHEMA, n_rows: m.n_rows, n_groups: m.n_groups,
        n_folds: m.n_folds, task }
}

fn check_group_rows<'a>(
    rows: &Rows, groups: impl Iterator<Item = (&'a str, usize)>,
) -> Result<()> {
    let n_rows = rows.observed.len();
    let mut total = 0_usize;
    let mut seen = HashSet::new();
    let mut sizes = Vec::new();
    for (id, n) in groups {
        require(seen.insert(id) && (1..=n_rows).contains(&n), "Inconsistent evaluation group roster")?;
        total = total.checked_add(n).ok_or("Invalid evaluation group count")?;
        sizes.push(n);
    }
    require(total == n_rows, "Inconsistent evaluation group counts")?;

    // Largest group first, then lowest-index lightest fold. Producer
    // lexical tie-breaking among equal-sized groups cannot affect the
    // resulting loads. This neither reconstructs nor asserts row-to-group IDs.
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let mut loads: BinaryHeap<_> = (0..rows.folds.len()).map(|i| Reverse((0, i))).collect();
    for size in sizes {
        let mut lightest = loads.peek_mut().ok_or("Missing evaluation folds")?;
        let Reverse((load, i)) = *lightest;
        *lightest = Reverse((load + size, i));
    }
    require(loads.into_iter().all(|Reverse((load, i))| rows.folds[i].0.len() == load),
        "Evaluation fold sizes disagree with the declared group size partition")
}

fn fold<M>(
    index: usize, declared: usize, train: usize, test: usize, encoded: usize,
    total: usize, actual: usize, metrics: M,
) -> Result<FoldProjection<M>> {
    require(index == declared && test == actual && test > 0 && train > 0
        && train.checked_add(test) == Some(total) && (1..=MAX_ESTIMATOR_FEATURES).contains(&encoded),
        "Inconsistent evaluation fold counts")?;
    Ok(FoldProjection { fold: declared, n_train: train, n_test: test,
        encoded_feature_count: encoded, metrics })
}

fn reg(m: RegressionMetrics) -> Reg {
    Reg { mae: m.mae, rmse: m.rmse, r2: m.r2 }
}

fn bin(m: BinaryMetrics) -> Bin {
    Bin { accuracy: m.accuracy, balanced_accuracy: m.balanced_accuracy, precision: m.precision,
        recall: m.recall, f1: m.f1, log_loss: m.log_loss, brier: m.brier, roc_auc: m.roc_auc,
        n_positive: m.n_positive, n_negative: m.n_negative, tp: m.tp, fp: m.fp, tn: m.tn, fn_: m.fn_ }
}

fn mean(values: &[f64]) -> Result<f64> {
    Ok(regression_metrics(&vec![0.0; values.len()], values)
        .map_err(|_| "Cannot verify aggregate evaluation metrics")?.mae)
}

fn compare<T: PartialEq>(a: T, b: T) -> Result<()> {
    require(a == b, "Evaluation metrics disagree with bundle data")
}

fn project_reg(m: Manifest<Reg, RegGroup, RegFold, RegMacro>, csv: &[u8]) -> Result<EvaluationProjection> {
    check_manifest(&m, REG)?;
    let rows = parse_rows(csv, m.n_rows, m.n_folds, None)?;
    compare(m.pooled_row_weighted, reg(regression_metrics(&rows.observed, &rows.predicted)?))?;
    check_group_rows(&rows, m.groups.iter().map(|g| (g.group.as_str(), g.n_rows)))?;
    let mut maes = Vec::with_capacity(m.n_groups);
    let mut rmses = Vec::with_capacity(m.n_groups);
    for g in &m.groups {
        require(g.mae >= 0.0 && g.rmse >= 0.0 && g.r2.map_or(true, |r| r <= 1.0 && g.n_rows > 1),
            "Invalid regression group metrics")?;
        maes.push(g.mae);
        rmses.push(g.rmse);
    }
    let equal_group_weighted = RegMacro { mean_group_mae: mean(&maes)?, mean_group_rmse: mean(&rmses)?,
        worst_group_mae: maes.into_iter().fold(0.0, f64::max) };
    require(equal_group_weighted == m.equal_group_weighted, "Inconsistent regression group summary")?;
    let mut folds = Vec::with_capacity(m.n_folds);
    for (i, (f, (y, p))) in m.folds.iter().zip(&rows.folds).enumerate() {
        check_encoded_width(&m, f.encoded_feature_count)?;
        let measured = reg(regression_metrics(y, p)?);
        compare(Reg { mae: f.mae, rmse: f.rmse, r2: f.r2 }, measured)?;
        folds.push(fold(i, f.fold, f.n_train, f.n_test, f.encoded_feature_count, m.n_rows, y.len(), measured)?);
    }
    Ok(projection(&m, Aggregate::Regression { pooled_row_weighted: m.pooled_row_weighted, equal_group_weighted, folds }))
}

fn binary_group(m: Bin, n: usize) -> Result<()> {
    let parts = [m.n_positive, m.n_negative, m.tp, m.fp, m.tn, m.fn_];
    require(parts.iter().all(|&v| v <= n), "Invalid binary group counts")?;
    require(m.n_positive + m.n_negative == n && m.tp + m.fn_ == m.n_positive
        && m.tn + m.fp == m.n_negative, "Invalid binary group confusion totals")?;
    let ratio = |a: f64, b: f64| (b != 0.0).then(|| a / b);
    let recall = ratio(m.tp as f64, m.n_positive as f64);
    let specificity = ratio(m.tn as f64, m.n_negative as f64);
    let epsilon = LOG_LOSS_EPSILON;
    let high = 1.0 - epsilon;
    // Group averages of clipped losses stay between the endpoint losses.
    // Allow binary64 rounding in the compensated sum and final division.
    let low_loss = (-high.ln()).min(-(-epsilon).ln_1p()) * (1.0 - 8.0 * f64::EPSILON);
    let high_loss = (-epsilon.ln()).max(-(-high).ln_1p()) * (1.0 + 8.0 * f64::EPSILON);
    require(
        m.accuracy == (m.tp + m.tn) as f64 / n as f64
        && m.balanced_accuracy == recall.zip(specificity).map(|(s, t)| (s + t) * 0.5)
        && m.precision == ratio(m.tp as f64, (m.tp + m.fp) as f64)
        && m.recall == recall
        && m.f1 == ratio(2.0 * m.tp as f64, 2.0 * m.tp as f64 + m.fp as f64 + m.fn_ as f64)
        && (low_loss..=high_loss).contains(&m.log_loss) && (0.0..=1.0).contains(&m.brier)
        && (match m.roc_auc {
            Some(v) => m.n_positive != 0 && m.n_negative != 0 && (0.0..=1.0).contains(&v),
            None => m.n_positive == 0 || m.n_negative == 0,
        }),
        "Invalid binary group metrics",
    )
}

fn project_bin(m: Manifest<Bin, BinGroup, BinFold, BinMacro>, header: BinaryHeader, csv: &[u8]) -> Result<EvaluationProjection> {
    check_manifest(&m, BIN)?;
    let threshold = header.threshold;
    require((0.0..=1.0).contains(&threshold) && header.positive_label == 1, "Invalid binary evaluation metadata")?;
    let rows = parse_rows(csv, m.n_rows, m.n_folds, Some(threshold))?;
    compare(m.pooled_row_weighted, bin(binary_metrics(&rows.observed, &rows.predicted, threshold)?))?;
    check_group_rows(&rows, m.groups.iter().map(|g| (g.group.as_str(), g.n_rows)))?;
    let mut losses = Vec::new();
    let mut accuracies = Vec::new();
    let mut balanced = Vec::new();
    let mut confusion = [0; 4];
    for g in &m.groups {
        let r = g.metrics;
        binary_group(r, g.n_rows)?;
        losses.push(r.log_loss);
        accuracies.push(r.accuracy);
        balanced.extend(r.balanced_accuracy);
        for (sum, n) in confusion.iter_mut().zip([r.tp, r.fp, r.tn, r.fn_]) { *sum += n; }
    }
    let pooled = m.pooled_row_weighted;
    compare(confusion, [pooled.tp, pooled.fp, pooled.tn, pooled.fn_])?;
    let equal_group_weighted = BinMacro { mean_group_log_loss: mean(&losses)?, mean_group_accuracy: mean(&accuracies)?,
        mean_defined_group_balanced_accuracy: if balanced.is_empty() { None } else { Some(mean(&balanced)?) },
        balanced_accuracy_group_denominator: balanced.len() };
    require(equal_group_weighted == m.equal_group_weighted, "Inconsistent binary group summary")?;
    let mut folds = Vec::with_capacity(m.n_folds);
    for (i, (f, (y, p))) in m.folds.iter().zip(&rows.folds).enumerate() {
        check_encoded_width(&m, f.encoded_feature_count)?;
        let measured = bin(binary_metrics(y, p, threshold)?);
        compare(f.metrics, measured)?;
        folds.push(fold(i, f.fold, f.n_train, f.n_test, f.encoded_feature_count, m.n_rows, y.len(), measured)?);
    }
    Ok(projection(&m, Aggregate::Binary { threshold, pooled_row_weighted: pooled, equal_group_weighted, folds }))
}
