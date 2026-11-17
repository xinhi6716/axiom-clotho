//! Explicit regression export into the existing saved-series v2 protocol.
use super::{read_bounded, Artifact, Result, Table};
use clotho_core::saved_comparison::MAX_DECLARATION_BYTES;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

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
            || contract.target != artifact.target
            || [&contract.cohort, &contract.target, &contract.unit, &contract.horizon]
                .iter().any(|value| !valid_text(value))
        {
            return Err("Invalid saved prediction contract or target differs from model artifact".into());
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
