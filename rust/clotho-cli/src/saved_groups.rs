//! Optional declared grouping; no inference of physical identity or independence.
use super::{read_bounded, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Mapping {
    schema_version: String,
    cohort: String,
    row_ids: Vec<String>,
    group_ids: Vec<String>,
}

// Scaled accumulation avoids overflowing the sum. Reject representational
// underflow rather than reporting a false exact zero for a nonzero mean.
fn mean(values: impl ExactSizeIterator<Item = f64> + Clone) -> Result<f64> {
    let len = values.len();
    let scale = values.clone().map(f64::abs).fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(0.0);
    }
    let (mut sum, mut correction) = (0.0_f64, 0.0_f64);
    for value in values {
        let term = value / scale;
        if value != 0.0 && term == 0.0 {
            return Err("Grouped squared-error range is too wide".into());
        }
        let next = sum + term;
        correction += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
    let total = sum + correction;
    let normalized = total / len as f64;
    if total != 0.0 && normalized == 0.0 {
        return Err("Grouped squared-error mean is outside the supported range".into());
    }
    let result = normalized * scale;
    if !result.is_finite() || (normalized != 0.0 && result == 0.0) {
        return Err("Grouped squared-error mean is outside the supported range".into());
    }
    Ok(result)
}

fn square_error(observed: f64, predicted: f64) -> Result<f64> {
    let error = observed - predicted;
    let square = error * error;
    if !square.is_finite() || (error != 0.0 && square == 0.0) {
        return Err("Grouped squared errors are outside the supported range".into());
    }
    Ok(square)
}

pub fn summarize(
    path: &str,
    cohort: &str,
    row_ids: &[String],
    observed: &[f64],
    model: &[f64],
    baseline: &[f64],
) -> Result<serde_json::Value> {
    let bytes = read_bounded(path)?;
    if bytes
        .iter()
        .copied()
        .find(|b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        != Some(b'{')
    {
        return Err("Group mapping must be a JSON object".into());
    }
    let mapping: Mapping = serde_json::from_slice(&bytes)
        .map_err(|_| "Invalid declared group mapping JSON")?;
    // The caller has already validated the unique, bounded observed roster.
    // Literal equality therefore also rejects duplicates and missing/extra rows.
    if mapping.schema_version != "clotho.declared_row_groups.v1"
        || mapping.cohort != cohort
        || mapping.row_ids != row_ids
        || mapping.group_ids.len() != row_ids.len()
        || mapping.group_ids.iter().any(|id| {
            id.trim().is_empty() || id.len() > 256 || id.chars().any(char::is_control)
        })
    {
        return Err("Group mapping must match the complete ordered cohort with valid labels".into());
    }
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, group) in mapping.group_ids.iter().enumerate() {
        groups.entry(group).or_default().push(i);
    }
    let mut deltas = Vec::with_capacity(row_ids.len());
    let mut baseline_squares = Vec::with_capacity(row_ids.len());
    let mut model_squares = Vec::with_capacity(row_ids.len());
    for i in 0..row_ids.len() {
        let b = square_error(observed[i], baseline[i])?;
        let m = square_error(observed[i], model[i])?;
        baseline_squares.push(b);
        model_squares.push(m);
        deltas.push(b - m);
    }
    let mut rows = Vec::with_capacity(groups.len());
    let mut means = Vec::with_capacity(groups.len());
    let (mut improved, mut tied, mut worsened, mut perfect) = (0, 0, 0, 0);
    for (group, indices) in &groups {
        // Each mean traverses the same ordered rows for scale and summation.
        // Borrow the values by index instead of copying three group-sized buffers.
        let (delta, baseline_mse, model_mse) = if let &[i] = indices.as_slice() {
            // A singleton mean is its value, except the existing mean returns
            // positive zero for an all-zero input.
            let canonical = |value| if value == 0.0 { 0.0 } else { value };
            (
                canonical(deltas[i]),
                baseline_squares[i],
                model_squares[i],
            )
        } else {
            (
                mean(indices.iter().map(|&i| deltas[i]))?,
                mean(indices.iter().map(|&i| baseline_squares[i]))?,
                mean(indices.iter().map(|&i| model_squares[i]))?,
            )
        };
        let baseline_perfect = indices.iter().all(|&i| baseline_squares[i] == 0.0);
        if delta > 0.0 {
            improved += 1;
        } else if delta < 0.0 {
            worsened += 1;
        } else {
            tied += 1;
        }
        if baseline_perfect {
            perfect += 1;
        }
        means.push(delta);
        rows.push(serde_json::json!({
            "group_id": group, "n_rows": indices.len(),
            "mean_squared_error_delta": delta, "baseline_perfect": baseline_perfect,
            "baseline_mse": baseline_mse, "model_mse": model_mse
        }));
    }
    Ok(serde_json::json!({
        "schema_version": "clotho.declared_group_comparison.v1",
        "delta_definition": "mean(baseline squared error - model squared error); positive favors model",
        "n_groups": groups.len(), "n_rows": row_ids.len(),
        "pooled_row_mean_squared_error_delta": mean(deltas.iter().copied())?,
        "macro_equal_group_mean_squared_error_delta": mean(means.iter().copied())?,
        "improved_groups": improved, "tied_groups": tied, "worsened_groups": worsened,
        "baseline_perfect_groups": perfect,
        "independence_verified": false, "provenance_verified": false,
        "groups": rows
    }))
}
