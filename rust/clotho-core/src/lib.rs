//! Incremental Rust core: source diagnostics, mixed-feature preprocessing,
//! regression, binary classification and group-disjoint evaluation. The native
//! CLI uses these algorithms without Python; legacy workflows are not fully
//! migrated or replaced. See `docs/RUST_NATIVE_WORKFLOW.md`,
//! `docs/RUST_BINARY_AND_MIXED.md` and `docs/RUST_MIGRATION_STATUS.md`.
#![forbid(unsafe_code)]

pub mod evaluation;
pub mod input_support;
pub mod knn;
pub mod numeric_bytes;
pub mod regression;

pub mod classification_metrics;
pub mod preprocessing;

pub mod classification;
