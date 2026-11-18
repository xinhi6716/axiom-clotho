//! Descriptive pooling of caller-supplied scalar reports, not roster verification.
use super::{create_file, required, Result};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MAX_REPORTS: usize = 64;
const MAX_BYTES: u64 = 64 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    reports: Vec<String>,
    model_id: String,
    baseline_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    schema_version: String,
    cohort: String,
    target: String,
    unit: String,
    horizon: String,
    n_rows: usize,
    model_rmse: f64,
    baseline_rmse: f64,
    mse_skill: serde_json::Value,
    validation: String,
    provenance_verified: bool,
    query_bytes_verified: bool,
    fit_performed: bool,
    query_sha256: String,
}

fn read_object<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let mut bytes = Vec::new();
    File::open(path)?.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("Pooling input exceeds 64 KiB limit".into());
    }
    if bytes.iter().copied().find(|b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r')) != Some(b'{') {
        return Err("Pooling input must be a JSON object".into());
    }
    reject_numeric_underflow(&bytes)?;
    serde_json::from_slice(&bytes).map_err(|_| "Invalid strict pooling JSON".into())
}

// serde_json can round a nonzero decimal token such as 1e-999 to zero. Do not
// let such an edited report silently become a perfect baseline. Structural and
// number grammar validation is still performed by serde, including duplicates.
fn reject_numeric_underflow(bytes: &[u8]) -> Result<()> {
    let (mut i, mut in_string) = (0, false);
    while i < bytes.len() {
        if in_string {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == b'"' { in_string = false; }
        } else if bytes[i] == b'"' {
            in_string = true;
        } else if bytes[i].is_ascii_digit() || bytes[i] == b'-' {
            let start = i;
            while i < bytes.len() && matches!(bytes[i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                i += 1;
            }
            let token = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
            if let Ok(value) = token.parse::<f64>() {
                let mantissa = token.split(['e', 'E']).next().unwrap_or("");
                if value == 0.0 && mantissa.bytes().any(|b| (b'1'..=b'9').contains(&b)) {
                    return Err("Nonzero pooling input number underflows binary64".into());
                }
            }
            continue;
        }
        i += 1;
    }
    Ok(())
}

fn valid_text(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}

fn skill(model: f64, baseline: f64) -> Result<Option<f64>> {
    if baseline == 0.0 {
        return Ok(None);
    }
    if model == 0.0 {
        return Ok(Some(1.0));
    }
    // Match the core range restriction, including caller-supplied reports.
    if model.is_subnormal() || baseline.is_subnormal() {
        return Err("Pooled skill is outside the supported binary64 range".into());
    }
    let ratio = model / baseline;
    let result = 1.0 - ratio * ratio;
    if !result.is_finite() {
        return Err("Pooled skill is outside the supported binary64 range".into());
    }
    Ok(Some(result))
}

fn pooled_rmse(reports: &[Report], baseline: bool, total: usize) -> Result<f64> {
    let value = |r: &Report| if baseline { r.baseline_rmse } else { r.model_rmse };
    let scale = reports.iter().map(value).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(0.0);
    }
    let (mut sum, mut correction) = (0.0_f64, 0.0_f64);
    for report in reports {
        let ratio = value(report) / scale;
        let term = ratio * ratio * (report.n_rows as f64 / total as f64);
        if value(report) != 0.0 && term == 0.0 {
            return Err("Report RMSE range is too wide to pool without underflow".into());
        }
        let next = sum + term;
        correction += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
    let result = (sum + correction).sqrt() * scale;
    if !result.is_finite() || result == 0.0 {
        return Err("Pooled RMSE is outside the supported binary64 range".into());
    }
    Ok(result)
}

pub(super) fn run(opts: &HashMap<String, String>) -> Result<()> {
    if opts.keys().any(|key| !["--manifest", "--out"].contains(&key.as_str())) {
        return Err("Unknown option for pool-saved".into());
    }
    let output = Path::new(required(opts, "--out")?);
    if output.symlink_metadata().is_ok() {
        return Err("Output already exists; choose a new path".into());
    }
    let manifest_path = Path::new(required(opts, "--manifest")?);
    let manifest: Manifest = read_object(manifest_path)?;
    if manifest.schema_version != "clotho.saved_report_manifest.v1"
        || manifest.reports.is_empty() || manifest.reports.len() > MAX_REPORTS
        || !valid_text(&manifest.model_id) || !valid_text(&manifest.baseline_id)
        || manifest.model_id == manifest.baseline_id
    {
        return Err("Manifest requires 1 to 64 reports and distinct bounded model/baseline declarations".into());
    }
    let directory = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let (mut cohorts, mut digests, mut paths) = (HashSet::new(), HashSet::new(), HashSet::new());
    let mut reports: Vec<Report> = Vec::new();
    let mut total = 0;
    for path in &manifest.reports {
        if path.is_empty() || !paths.insert(path) {
            return Err("Report paths must be nonempty and distinct".into());
        }
        let report: Report = read_object(&directory.join(path))?;
        if report.schema_version != "clotho.declared_regression_comparison.v2"
            || report.n_rows == 0 || report.n_rows > 20_000
            || [&report.cohort, &report.target, &report.unit, &report.horizon]
                .iter().any(|s| !valid_text(s))
            || ![report.model_rmse, report.baseline_rmse].iter().all(|v| v.is_finite() && *v >= 0.0)
            || report.provenance_verified || !report.query_bytes_verified || report.fit_performed
            || report.validation != "matching declared contracts, ordered unique row IDs and supplied query byte digest only"
            || report.query_sha256.len() != 64
            || !report.query_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("Unsupported or invalid saved comparison report".into());
        }
        let expected = skill(report.model_rmse, report.baseline_rmse)?;
        let consistent = match expected {
            None => report.mse_skill.is_null(),
            Some(value) => report.mse_skill.as_f64() == Some(value),
        };
        if !consistent {
            return Err("Report skill is inconsistent with its RMSE values".into());
        }
        if !cohorts.insert(report.cohort.clone()) || !digests.insert(report.query_sha256.clone()) {
            return Err("Reports must declare distinct cohorts and query digests".into());
        }
        if let Some(first) = reports.first() {
            if (report.target.as_str(), report.unit.as_str(), report.horizon.as_str())
                != (first.target.as_str(), first.unit.as_str(), first.horizon.as_str())
            {
                return Err("Report target, unit and horizon must match exactly".into());
            }
        }
        total += report.n_rows;
        reports.push(report);
    }
    let model = pooled_rmse(&reports, false, total)?;
    let baseline = pooled_rmse(&reports, true, total)?;
    let sources: Vec<_> = reports.iter().map(|r| serde_json::json!({
        "cohort": r.cohort, "query_sha256": r.query_sha256, "n_rows": r.n_rows,
        "model_rmse": r.model_rmse, "baseline_rmse": r.baseline_rmse
    })).collect();
    let result = serde_json::json!({
        "schema_version": "clotho.declared_report_pool.v1",
        "target": reports[0].target, "unit": reports[0].unit, "horizon": reports[0].horizon,
        "n_reports": reports.len(), "n_rows": total,
        "model_id": manifest.model_id, "baseline_id": manifest.baseline_id,
        "identities_verified": false, "membership_verified": false,
        "model_rmse": model, "baseline_rmse": baseline, "mse_skill": skill(model, baseline)?,
        "aggregation": "row-count-weighted squared RMSE from supplied scalar reports; not average skill",
        "query_bytes_verified": false, "roster_completeness_verified": false,
        "disjointness_verified": false, "provenance_verified": false, "fit_performed": false,
        "sources": sources
    });
    create_file(output, &serde_json::to_vec_pretty(&result)?)?;
    println!("Pooled {} declared reports; no roster or provenance verification", reports.len());
    Ok(())
}
