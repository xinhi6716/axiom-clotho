//! Fit-free file adapter for explicitly declared ordered regression series.
use super::{create_file, read_bounded, required, Result};
use clotho_core::saved_comparison::{
    compare_saved_regression, CohortContract, SavedRegressionSeries,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedSeries {
    schema_version: String,
    cohort: String,
    target: String,
    unit: String,
    horizon: String,
    row_ids: Vec<String>,
    values: Vec<f64>,
}

impl SavedSeries {
    fn load(path: &str) -> Result<Self> {
        // Do not echo untrusted file contents (or private row IDs) in errors.
        let bytes = read_bounded(path)?;
        // A derived struct can also deserialize positionally from a JSON array.
        // Require a named object without losing serde's duplicate-field checks.
        if bytes
            .iter()
            .copied()
            .find(|byte| !matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
            != Some(b'{')
        {
            return Err("Invalid saved-series JSON; expected a JSON object".into());
        }
        let series: Self = serde_json::from_slice(&bytes)
            .map_err(|_| "Invalid saved-series JSON; expected the strict regression-series schema")?;
        if series.schema_version != "clotho.declared_regression_series.v1" {
            return Err("Unsupported saved-series schema".into());
        }
        Ok(series)
    }

    fn borrowed<'a>(&'a self, ids: &'a [&'a str]) -> SavedRegressionSeries<'a> {
        SavedRegressionSeries {
            contract: CohortContract {
                cohort: &self.cohort,
                target: &self.target,
                unit: &self.unit,
                horizon: &self.horizon,
            },
            row_ids: ids,
            values: &self.values,
        }
    }
}

pub fn run(opts: &HashMap<String, String>) -> Result<()> {
    const ALLOWED: [&str; 4] = ["--observed", "--model", "--baseline", "--out"];
    if opts.keys().any(|key| !ALLOWED.contains(&key.as_str())) {
        return Err("Unknown option for compare-saved".into());
    }
    let output = Path::new(required(opts, "--out")?);
    let observed = SavedSeries::load(required(opts, "--observed")?)?;
    let model = SavedSeries::load(required(opts, "--model")?)?;
    let baseline = SavedSeries::load(required(opts, "--baseline")?)?;
    let observed_ids: Vec<_> = observed.row_ids.iter().map(String::as_str).collect();
    let model_ids: Vec<_> = model.row_ids.iter().map(String::as_str).collect();
    let baseline_ids: Vec<_> = baseline.row_ids.iter().map(String::as_str).collect();
    let result = compare_saved_regression(
        observed.borrowed(&observed_ids),
        model.borrowed(&model_ids),
        baseline.borrowed(&baseline_ids),
    )?;
    let report = serde_json::json!({
        "schema_version": "clotho.declared_regression_comparison.v1",
        "cohort": result.contract.cohort,
        "target": result.contract.target,
        "unit": result.contract.unit,
        "horizon": result.contract.horizon,
        "n_rows": result.n_rows,
        "model_rmse": result.model_rmse,
        "baseline_rmse": result.baseline_rmse,
        "mse_skill": result.mse_skill,
        "validation": "matching declared contracts and ordered unique row IDs only",
        "provenance_verified": false,
        "query_bytes_verified": false,
        "fit_performed": false
    });
    // All validation and serialization finish before creating the private output.
    let bytes = serde_json::to_vec_pretty(&report)?;
    create_file(output, &bytes)?;
    println!(
        "Wrote declared comparison for {} rows; no fitting or provenance verification",
        result.n_rows
    );
    Ok(())
}
