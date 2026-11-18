//! Fit-free file adapter for explicitly declared ordered regression series.
use super::{create_file, read_bounded, required, Result};
use super::numeric_json::reject_numeric_underflow;
use clotho_core::saved_comparison::{
    compare_saved_regression, CohortContract, SavedRegressionSeries,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
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
    #[serde(default, deserialize_with = "present_digest")]
    query_sha256: Option<String>,
}

// Absence is allowed only for v1. An explicit null must not bypass the v1
// unknown-field boundary or substitute for the required v2 digest.
fn present_digest<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error> {
    String::deserialize(deserializer).map(Some)
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
        reject_numeric_underflow(&bytes)?;
        let series: Self = serde_json::from_slice(&bytes)
            .map_err(|_| "Invalid saved-series JSON; expected the strict regression-series schema")?;
        match (series.schema_version.as_str(), &series.query_sha256) {
            ("clotho.declared_regression_series.v1", None) => {}
            ("clotho.declared_regression_series.v2", Some(digest))
                if digest.len() == 64
                    && digest
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) => {}
            _ => return Err("Unsupported saved-series schema or invalid query digest".into()),
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
    const ALLOWED: [&str; 6] = [
        "--observed", "--model", "--baseline", "--out", "--query", "--groups",
    ];
    if opts.keys().any(|key| !ALLOWED.contains(&key.as_str())) {
        return Err("Unknown option for compare-saved".into());
    }
    let output = Path::new(required(opts, "--out")?);
    let observed = SavedSeries::load(required(opts, "--observed")?)?;
    let model = SavedSeries::load(required(opts, "--model")?)?;
    let baseline = SavedSeries::load(required(opts, "--baseline")?)?;
    // The file is read once. This binds its bytes at read time, not its path,
    // later contents, physical rows, or the process that produced predictions.
    let query_digest = if let Some(query) = opts.get("--query") {
        if [&observed, &model, &baseline]
            .iter()
            .any(|series| series.query_sha256.is_none())
        {
            return Err("--query requires v2 series for all three inputs".into());
        }
        let digest = format!("{:x}", Sha256::digest(read_bounded(query)?));
        if [&observed, &model, &baseline]
            .iter()
            .any(|series| series.query_sha256.as_deref() != Some(digest.as_str()))
        {
            return Err("Saved-series query digest does not match the supplied query bytes".into());
        }
        Some(digest)
    } else {
        if [&observed, &model, &baseline]
            .iter()
            .any(|series| series.query_sha256.is_some())
        {
            return Err("v2 saved series require --query".into());
        }
        None
    };
    let observed_ids: Vec<_> = observed.row_ids.iter().map(String::as_str).collect();
    let model_ids: Vec<_> = model.row_ids.iter().map(String::as_str).collect();
    let baseline_ids: Vec<_> = baseline.row_ids.iter().map(String::as_str).collect();
    let result = compare_saved_regression(
        observed.borrowed(&observed_ids),
        model.borrowed(&model_ids),
        baseline.borrowed(&baseline_ids),
    )?;
    let mut report = serde_json::json!({
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
    if let Some(digest) = query_digest {
        report["schema_version"] = "clotho.declared_regression_comparison.v2".into();
        report["query_sha256"] = digest.into();
        report["query_bytes_verified"] = true.into();
        report["validation"] = concat!(
            "matching declared contracts, ordered unique row IDs ",
            "and supplied query byte digest only"
        )
        .into();
    }
    if let Some(path) = opts.get("--groups") {
        report["grouped_summary"] = super::saved_groups::summarize(
            path,
            &observed.cohort,
            &observed.row_ids,
            &observed.values,
            &model.values,
            &baseline.values,
        )?;
    }
    // All validation and serialization finish before creating the private output.
    let bytes = serde_json::to_vec_pretty(&report)?;
    create_file(output, &bytes)?;
    println!(
        "Wrote declared comparison for {} rows; no fitting or provenance verification",
        result.n_rows
    );
    Ok(())
}


#[cfg(test)]
mod numeric_token_tests {
    use super::*;

    #[test]
    fn accepted_subnormal_tokens_survive_the_actual_serde_conversion() {
        for (token, expected_bits) in [
            ("3e-324", 1),
            ("5e-324", 1),
            ("2.4703282292062328e-324", 1),
            ("-3e-324", (1_u64 << 63) | 1),
            ("0e-999", 0),
            ("-0.0e-999", 1_u64 << 63),
        ] {
            let bytes = format!(
                r#"{{"schema_version":"clotho.declared_regression_series.v1","cohort":"c","target":"y","unit":"u","horizon":"h","row_ids":["a"],"values":[{token}]}}"#
            );
            reject_numeric_underflow(bytes.as_bytes()).unwrap();
            let parsed: SavedSeries = serde_json::from_slice(bytes.as_bytes()).unwrap();
            assert_eq!(parsed.values[0].to_bits(), expected_bits, "{token}");
        }
        for token in ["2e-324", "2.4703282292062327e-324", "-2.4703282292062327e-324"] {
            assert!(reject_numeric_underflow(token.as_bytes()).is_err());
        }
    }
}
