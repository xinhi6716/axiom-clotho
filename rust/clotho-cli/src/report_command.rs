//! Offline, read-only results UI. Never deserializes or loads a fitted model.
use crate::{create_file, required, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Read;
use std::path::Path;
const LIMIT: u64 = 16 * 1024 * 1024;

fn read(path: &Path) -> Result<Vec<u8>> {
    if !std::fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("Report inputs must be regular files, not symlinks".into());
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(LIMIT + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > LIMIT {
        return Err("Report input exceeds 16 MiB".into());
    }
    Ok(bytes)
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
fn display(v: &Value) -> String {
    match v {
        Value::Null => "Not defined".into(),
        Value::String(s) => escape(s),
        _ => escape(&v.to_string()),
    }
}
fn table(v: &Value) -> String {
    let mut s = String::from(
        "<table><thead><tr><th scope=col>Measure</th><th scope=col>Value</th></tr></thead><tbody>",
    );
    if let Some(map) = v.as_object() {
        for (k, v) in map {
            let _ = write!(
                s,
                "<tr><th scope=row>{}</th><td>{}</td></tr>",
                escape(k),
                display(v)
            );
        }
    }
    s.push_str("</tbody></table>");
    s
}

fn count(v: &Value) -> Result<u64> {
    v.as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(|| "Missing or invalid positive count".into())
}
fn metric(v: &Value, key: &str, nullable: bool) -> Result<()> {
    let value = v.get(key).ok_or("Missing metric field")?;
    if nullable && value.is_null() {
        return Ok(());
    }
    if !value.as_f64().is_some_and(|x| x.is_finite()) {
        return Err("Invalid metric field".into());
    }
    Ok(())
}
fn validate_metrics(v: &Value, binary: bool) -> Result<()> {
    if binary {
        for k in ["accuracy", "log_loss", "brier"] {
            metric(v, k, false)?;
        }
        for k in ["balanced_accuracy", "precision", "recall", "f1", "roc_auc"] {
            metric(v, k, true)?;
        }
        for k in ["n_positive", "n_negative", "tp", "fp", "tn", "fn"] {
            if v[k].as_u64().is_none() {
                return Err("Missing binary metric count".into());
            }
        }
    } else {
        metric(v, "mae", false)?;
        metric(v, "rmse", false)?;
        metric(v, "r2", true)?;
    }
    Ok(())
}
fn validate_shape(m: &Value, binary: bool, n: u64) -> Result<Vec<u64>> {
    let ng = count(&m["n_groups"])?;
    let nf = count(&m["n_folds"])?;
    if ng > n || nf < 2 || nf > ng {
        return Err("Invalid group/fold counts".into());
    }
    for key in [
        "model",
        "target",
        "group",
        "selection",
        "split",
        "preprocess",
    ] {
        if m[key].as_str().filter(|s| !s.is_empty()).is_none() {
            return Err("Missing protocol field".into());
        }
    }
    let features = m["features"].as_array().ok_or("Missing features")?;
    if features.is_empty()
        || features
            .iter()
            .any(|v| v.as_str().filter(|s| !s.is_empty()).is_none())
    {
        return Err("Invalid features".into());
    }
    validate_metrics(&m["pooled_row_weighted"], binary)?;
    let equal = &m["equal_group_weighted"];
    if binary {
        metric(equal, "mean_group_log_loss", false)?;
        metric(equal, "mean_group_accuracy", false)?;
        metric(equal, "mean_defined_group_balanced_accuracy", true)?;
        if equal["balanced_accuracy_group_denominator"]
            .as_u64()
            .filter(|d| *d <= ng)
            .is_none()
        {
            return Err("Invalid group metric denominator".into());
        }
    } else {
        for k in ["mean_group_mae", "mean_group_rmse", "worst_group_mae"] {
            metric(equal, k, false)?;
        }
    }
    let groups = m["groups"].as_array().ok_or("Missing groups")?;
    let folds = m["folds"].as_array().ok_or("Missing folds")?;
    if groups.len() as u64 != ng || folds.len() as u64 != nf {
        return Err("Group/fold list size differs from counts".into());
    }
    let mut seen = std::collections::HashSet::new();
    let mut total = 0u64;
    for g in groups {
        let name = g["group"].as_str().ok_or("Missing group identity")?;
        if !seen.insert(name) {
            return Err("Duplicate group identity".into());
        }
        total = total
            .checked_add(count(&g["n_rows"])?)
            .ok_or("Group count overflow")?;
        validate_metrics(if binary { &g["metrics"] } else { g }, binary)?;
    }
    if total != n {
        return Err("Group row counts do not total n_rows".into());
    }
    let mut counts = Vec::new();
    total = 0;
    for (i, f) in folds.iter().enumerate() {
        if f["fold"].as_u64() != Some(i as u64) {
            return Err("Fold identities must be ordered and unique".into());
        }
        let nt = count(&f["n_test"])?;
        let train = count(&f["n_train"])?;
        if nt.checked_add(train) != Some(n) {
            return Err("Fold train/test sizes do not total n_rows".into());
        }
        total = total.checked_add(nt).ok_or("Fold count overflow")?;
        counts.push(nt);
        count(&f["encoded_feature_count"])?;
        validate_metrics(if binary { &f["metrics"] } else { f }, binary)?;
    }
    if total != n {
        return Err("Fold held-out counts do not total n_rows".into());
    }
    Ok(counts)
}

pub fn run(opts: &HashMap<String, String>) -> Result<()> {
    if opts
        .keys()
        .any(|k| !["--run", "--out"].contains(&k.as_str()))
    {
        return Err("Unknown report option".into());
    }
    let dir = Path::new(required(opts, "--run")?);
    let output = Path::new(required(opts, "--out")?);
    let bytes = read(&dir.join("metrics.json"))?;
    let m: Value = serde_json::from_slice(&bytes)?;
    let schema = m["schema_version"]
        .as_str()
        .ok_or("Missing report schema")?;
    if ![
        "clotho.grouped_evaluation.v1",
        "clotho.grouped_binary_evaluation.v1",
    ]
    .contains(&schema)
    {
        return Err("Report supports native grouped regression/binary evaluation v1 only".into());
    }
    let complete = read(&dir.join("COMPLETE"))?;
    if complete != format!("{schema}\n").as_bytes() {
        return Err("Missing or mismatched COMPLETE marker".into());
    }
    let n = m["n_rows"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("Invalid row count")?;
    if !m["pooled_row_weighted"].is_object() || !m["equal_group_weighted"].is_object() {
        return Err("Missing metric sections".into());
    }
    let binary = schema.contains("binary");
    let expected_fold_counts = validate_shape(&m, binary, n)?;
    let mut actual_fold_counts = vec![0u64; expected_fold_counts.len()];
    if binary && m["positive_label"].as_u64() != Some(1) {
        return Err("Binary schema requires positive_label 1".into());
    }
    let threshold = if binary {
        m["threshold"]
            .as_f64()
            .filter(|t| t.is_finite() && (0.0..=1.0).contains(t))
            .ok_or("Missing or invalid binary threshold")?
    } else {
        0.5
    };
    let csv = read(&dir.join("predictions.csv"))?;
    let mut reader = csv::Reader::from_reader(csv.as_slice());
    let expected = if binary {
        vec![
            "row_index",
            "fold",
            "target",
            "positive_probability",
            "predicted_label",
        ]
    } else {
        vec!["row_index", "fold", "target", "prediction"]
    };
    if reader.headers()?.iter().collect::<Vec<_>>() != expected {
        return Err("Prediction header does not match schema".into());
    }
    let mut points = Vec::new();
    for (i, row) in reader.records().enumerate() {
        let row = row?;
        if row[0].parse::<usize>()? != i {
            return Err("Prediction row indices must be contiguous and ordered".into());
        }
        let fold = row[1].parse::<u64>()?;
        if fold >= m["n_folds"].as_u64().ok_or("Missing fold count")? {
            return Err("Prediction fold out of range".into());
        }
        actual_fold_counts[fold as usize] += 1;
        let y: f64 = row[2].parse()?;
        let p: f64 = row[3].parse()?;
        if !y.is_finite() || !p.is_finite() {
            return Err("Nonfinite prediction or target".into());
        }
        if binary && (!(y == 0.0 || y == 1.0) || !(0.0..=1.0).contains(&p)) {
            return Err("Invalid binary target/probability".into());
        }
        if binary {
            let label: u8 = row[4].parse()?;
            if label > 1 || label != u8::from(p >= threshold) {
                return Err("Predicted label disagrees with saved probability/threshold".into());
            }
        }
        points.push((y, p));
    }
    if points.len() as u64 != n {
        return Err("Prediction count differs from metrics".into());
    }
    if actual_fold_counts != expected_fold_counts {
        return Err("Prediction fold sizes disagree with metrics".into());
    }
    let mut html = String::from(include_str!("report_shell.html"));
    let title = if binary {
        "Binary classification"
    } else {
        "Regression"
    };
    let mut body = format!("<header><p class=eyebrow>CLOTHO / NATIVE RESULTS</p><h1>{title} workspace</h1><p>Local, offline evaluation explorer · Rust-generated · no Python runtime</p></header><nav aria-label=Sections><a href=#overview>Overview</a><a href=#metrics>Metrics</a><a href=#predictions>Predictions</a><a href=#details>Protocol &amp; details</a></nav><main id=main><section id=overview><h2>Evaluation overview</h2><div class=cards>");
    for (label, key) in [
        ("Held-out rows", "n_rows"),
        ("Groups held out", "n_groups"),
        ("Folds", "n_folds"),
        ("Model", "model"),
    ] {
        let _ = write!(
            body,
            "<article class=card><p>{label}</p><strong>{}</strong></article>",
            display(&m[key])
        );
    }
    body.push_str("</div><p class=notice>These are saved evaluation outputs, not a new training run. This viewer checks file shape and row alignment; it does not independently reproduce fits or certify scientific validity. Keep reports private if source names or predictions are sensitive.</p></section><section id=metrics><h2>Metrics</h2><div class=columns><article><h3>Pooled, row weighted</h3>");
    body.push_str(&table(&m["pooled_row_weighted"]));
    body.push_str("</article><article><h3>Equal group weighted</h3>");
    body.push_str(&table(&m["equal_group_weighted"]));
    body.push_str("</article></div><p>");
    body.push_str(if binary { "Class-dependent metrics can be undefined in single-class groups; use the saved denominator for mean defined group balanced accuracy. " } else { "Mean group RMSE is not pooled RMSE. " });
    body.push_str("Undefined metrics remain undefined, never silently replaced with zero.</p></section><section id=predictions><h2>Held-out predictions</h2>");
    body.push_str(&plot(&points, binary));
    body.push_str("<details><summary>Preview first 100 prediction rows</summary><div class=scroll><table><thead><tr><th scope=col>Row index</th><th scope=col>Observed target</th>");
    body.push_str(if binary {
        "<th scope=col>Positive probability</th><th scope=col>Predicted label</th>"
    } else {
        "<th scope=col>Prediction</th>"
    });
    body.push_str("</tr></thead><tbody>");
    for (i, (y, p)) in points.iter().take(100).enumerate() {
        let _ = write!(body, "<tr><th scope=row>{i}</th><td>{y}</td><td>{p}</td>");
        if binary {
            let _ = write!(body, "<td>{}</td>", u8::from(*p >= threshold));
        }
        body.push_str("</tr>");
    }
    body.push_str("</tbody></table></div></details></section><section id=details><h2>Protocol &amp; details</h2>");
    if binary {
        let _ = write!(body,"<p><strong>Positive label:</strong> 1</p><p><strong>Decision threshold:</strong> {threshold} (predict 1 when probability &ge; threshold)</p>");
    }
    for (label, key) in [
        ("Target", "target"),
        ("Grouping column", "group"),
        ("Feature allowlist", "features"),
        ("Selection rule", "selection"),
        ("Split rule", "split"),
        ("Preprocessing", "preprocess"),
    ] {
        let _ = write!(
            body,
            "<p><strong>{label}:</strong> {}</p>",
            display(&m[key])
        );
    }
    for (label, key) in [
        ("Per-group results", "groups"),
        ("Per-fold results", "folds"),
    ] {
        let _ = write!(body, "<details><summary>{label}</summary>");
        if let Some(rows) = m[key].as_array() {
            for row in rows {
                body.push_str(&table(row));
            }
        }
        body.push_str("</details>");
    }
    let _ = write!(body,"<details><summary>Full saved metrics (escaped JSON)</summary><pre>{}</pre></details><p>Schema: {}</p></section></main><footer>Clotho · Experimental native workflow. No telemetry, external assets, scripts, model loading, or network requests.</footer></body></html>",escape(&serde_json::to_string_pretty(&m)?),escape(schema));
    html.push_str(&body);
    create_file(output, html.as_bytes())?;
    println!("Wrote offline results UI for {n} rows (no fitting performed)");
    Ok(())
}

fn plot(points: &[(f64, f64)], binary: bool) -> String {
    // Divide before subtracting to avoid overflow even for +/-f64::MAX.
    let max = points
        .iter()
        .flat_map(|(y, p)| [y.abs(), p.abs()])
        .fold(1.0_f64, f64::max);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (y, p) in points {
        lo = lo.min(y / max).min(p / max);
        hi = hi.max(y / max).max(p / max);
    }
    if binary {
        lo = 0.0;
        hi = 1.0;
    }
    if hi == lo {
        lo = (lo - 0.5).max(-1.0);
        hi = (hi + 0.5).min(1.0);
    }
    let scale = |v: f64| (v / max - lo) / (hi - lo);
    let mut s=String::from("<figure><svg viewBox='0 0 640 420' role=img aria-labelledby='plot-title plot-desc'><title id=plot-title>Observed versus predicted</title><desc id=plot-desc>Scatter preview, at most the first 2000 rows in file order. Horizontal axis is observed target; vertical axis is prediction. Full values are in predictions.csv.</desc><path class=axis d='M60 20V360H610'/>");
    if !binary {
        s.push_str("<path class=reference d='M60 360L610 20'/>");
    }
    for (y, p) in points.iter().take(2000) {
        let _ = write!(
            s,
            "<circle cx='{:.3}' cy='{:.3}' r='3'/>",
            60.0 + 550.0 * scale(*y),
            360.0 - 340.0 * scale(*p)
        );
    }
    let _=write!(s,"<text x=60 y=382>{:.3e}</text><text x=550 y=382>{:.3e}</text><text x=260 y=410>Observed target</text><text x=65 y=16>Prediction {}</text></svg><figcaption>Preview: {} of {} rows; file-order subset, not a representative sample. {}</figcaption></figure>",lo*max,hi*max,if binary {"(positive probability)"} else {""},points.len().min(2000),points.len(),if binary {"Binary outcomes and probabilities are different quantities."} else {"Dashed line: prediction equals observed."});
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn escapes_all_html_boundaries() {
        assert_eq!(
            escape("<script a=\"&'\">"),
            "&lt;script a=&quot;&amp;&#39;&quot;&gt;"
        );
    }
    #[test]
    fn null_is_not_zero() {
        assert_eq!(display(&Value::Null), "Not defined");
    }
    #[test]
    fn extreme_points_have_finite_coordinates() {
        let p = plot(&[(-f64::MAX, f64::MAX), (f64::MAX, -f64::MAX)], false);
        assert!(!p.contains("NaN"));
        assert!(!p.contains("inf"));
    }
}
