//! Thin optional Python adapter. All payload access uses immutable PyBytes.
#![forbid(unsafe_code)]

use clotho_core::input_support::{FeatureCounts, MAX_FEATURES, MAX_ROWS};
use clotho_core::numeric_bytes::{prediction_input_support_bytes, NumericBytesColumn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

type FeatureTuple = (usize, usize, usize, Option<usize>, Option<usize>);
type ReportTuple = (
    usize,
    usize,
    usize,
    (usize, usize, usize, usize),
    Vec<FeatureTuple>,
);

fn columns<'py>(value: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyBytes>>> {
    let tuple = value
        .cast_exact::<PyTuple>()
        .map_err(|_| PyValueError::new_err("Expected an exact tuple of immutable byte columns."))?;
    if tuple.is_empty() || tuple.len() > MAX_FEATURES {
        return Err(PyValueError::new_err("Expected 1 to 100 byte columns."));
    }
    tuple
        .iter()
        .map(|item| {
            item.cast_exact::<PyBytes>()
                .cloned()
                .map_err(|_| PyValueError::new_err("Every column must be exact bytes."))
        })
        .collect()
}

#[pyfunction]
fn counts(training: &Bound<'_, PyAny>, incoming: &Bound<'_, PyAny>) -> PyResult<ReportTuple> {
    // Retain all owners before borrowing their payloads. No callback, detach,
    // NumPy buffer access, pointer casts, or references retained after this call.
    let training = columns(training)?;
    let incoming = columns(incoming)?;
    if training.len() != incoming.len() {
        return Err(PyValueError::new_err("Column counts must match."));
    }
    // The Python wrapper already aligns names. Opaque ASCII names avoid any
    // lossy Python-str -> Rust-String conversion, including lone surrogates.
    let names: Vec<_> = (0..training.len()).map(|i| format!("f{i}")).collect();
    let source: Vec<_> = training
        .iter()
        .zip(&names)
        .map(|(value, name)| NumericBytesColumn {
            name,
            values: value.as_bytes(),
        })
        .collect();
    let query: Vec<_> = incoming
        .iter()
        .zip(&names)
        .map(|(value, name)| NumericBytesColumn {
            name,
            values: value.as_bytes(),
        })
        .collect();
    let report = prediction_input_support_bytes(&source, &query)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut features = Vec::with_capacity(report.features.len());
    for feature in report.features {
        let FeatureCounts::Numeric {
            below_training_min_count,
            above_training_max_count,
        } = feature.counts
        else {
            return Err(PyValueError::new_err(
                "Unexpected numeric support protocol.",
            ));
        };
        features.push((
            feature.training_missing_count,
            feature.prediction_missing_count,
            feature.prediction_non_missing_without_training_values_count,
            below_training_min_count,
            above_training_max_count,
        ));
    }
    let flags = report.row_flag_counts;
    Ok((
        report.n_training_rows,
        report.n_prediction_rows,
        report.rows_with_any_flag,
        (
            flags.missing_value,
            flags.numeric_outside_training_range,
            flags.unseen_category,
            flags.non_missing_value_in_all_missing_training_feature,
        ),
        features,
    ))
}

#[pymodule(gil_used = true)]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("PROTOCOL", "numeric-source-support-v1")?;
    module.add("MAX_ROWS", MAX_ROWS)?;
    module.add("MAX_FEATURES", MAX_FEATURES)?;
    module.add_function(wrap_pyfunction!(counts, module)?)?;
    Ok(())
}
