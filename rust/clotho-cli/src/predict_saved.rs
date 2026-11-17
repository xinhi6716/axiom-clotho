//! Explicit regression export into the existing saved-series v2 protocol.
use super::{read_bounded, Artifact, Result, Table};
use clotho_core::saved_comparison::MAX_DECLARATION_BYTES;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Contract {
    schema_version: String,
    cohort: String,
    target: String,
    unit: String,
    horizon: String,
}

pub(super) struct Export {
    contract: Contract,
    row_ids: Vec<String>,
}

fn valid_text(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_DECLARATION_BYTES
        && !value.chars().any(char::is_control)
}

impl Export {
    pub(super) fn prepare(path: &str, id_column: &str, data: &Table, artifact: &Artifact) -> Result<Self> {
        if artifact.is_binary() {
            return Err("Saved-series export supports regression models only".into());
        }
        if artifact.features.iter().any(|feature| feature == id_column)
            || artifact.target == id_column
        {
            return Err("Row ID column must be separate from model features and target".into());
        }
        let export = Self::from_table(path, id_column, data)?;
        if export.contract.target != artifact.target {
            return Err("Saved prediction contract target differs from model artifact".into());
        }
        Ok(export)
    }

    fn from_table(path: &str, id_column: &str, data: &Table) -> Result<Self> {
        let bytes = read_bounded(path)?;
        if bytes
            .iter()
            .copied()
            .find(|b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
            != Some(b'{')
        {
            return Err("Saved prediction contract must be a JSON object".into());
        }
        let contract: Contract = serde_json::from_slice(&bytes)
            .map_err(|_| "Invalid saved prediction contract JSON")?;
        if contract.schema_version != "clotho.saved_prediction_contract.v1"
            || [&contract.cohort, &contract.target, &contract.unit, &contract.horizon]
                .iter().any(|value| !valid_text(value))
        {
            return Err("Invalid saved regression contract".into());
        }
        let column = data.column(id_column)?;
        let mut seen = HashSet::new();
        let mut row_ids = Vec::with_capacity(data.rows.len());
        for row in &data.rows {
            let id = &row[column];
            if !valid_text(id) || !seen.insert(id) {
                return Err("Saved prediction row IDs must be unique, bounded, nonblank text without controls".into());
            }
            row_ids.push(id.clone());
        }
        Ok(Self { contract, row_ids })
    }

    pub(super) fn observed(
        path: &str,
        id_column: &str,
        query: &Table,
        labels: &Table,
        query_bytes: &[u8],
    ) -> Result<Vec<u8>> {
        let export = Self::from_table(path, id_column, query)?;
        let target = &export.contract.target;
        if target == id_column || query.headers.iter().any(|header| header == target) {
            return Err("Observed export requires a query without the declared target column".into());
        }
        if labels.headers.len() != 2 {
            return Err("Observed CSV must contain exactly the ID and declared target columns".into());
        }
        let id = labels.column(id_column)?;
        let value = labels.column(target)?;
        let mut observed = HashMap::new();
        for row in &labels.rows {
            let key = &row[id];
            if !valid_text(key) || observed.contains_key(key.as_str()) {
                return Err("Observed IDs must be unique, bounded, nonblank text without controls".into());
            }
            let text = &row[value];
            let number: f64 = text
                .parse()
                .map_err(|_| "Observed targets must be explicit finite numbers")?;
            let mantissa = text.split(['e', 'E']).next().unwrap_or("");
            if !number.is_finite()
                || (number == 0.0 && mantissa.bytes().any(|b| (b'1'..=b'9').contains(&b)))
            {
                return Err("Observed target is outside the supported binary64 range".into());
            }
            observed.insert(key.as_str(), number);
        }
        if observed.len() != export.row_ids.len() {
            return Err("Observed IDs must match the complete query roster".into());
        }
        let values = export
            .row_ids
            .iter()
            .map(|id| {
                observed.get(id.as_str()).copied()
                    .ok_or_else(|| "Observed IDs must match the complete query roster".into())
            })
            .collect::<Result<Vec<f64>>>()?;
        export.bytes(&values, query_bytes)
    }

    pub(super) fn bytes(&self, predictions: &[f64], query: &[u8]) -> Result<Vec<u8>> {
        if predictions.len() != self.row_ids.len() || predictions.iter().any(|v| !v.is_finite()) {
            return Err("Invalid saved prediction values".into());
        }
        let value = serde_json::json!({
            "schema_version": "clotho.declared_regression_series.v2",
            "cohort": self.contract.cohort,
            "target": self.contract.target,
            "unit": self.contract.unit,
            "horizon": self.contract.horizon,
            "row_ids": self.row_ids,
            "values": predictions,
            "query_sha256": format!("{:x}", Sha256::digest(query))
        });
        Ok(serde_json::to_vec_pretty(&value)?)
    }
}
