//! Incremental Rust core: source-feature diagnostics and a standalone numeric
//! k-nearest-neighbor regression baseline. The existing Python workflow remains
//! authoritative; the Rust estimator is not wired into it. See `docs/RUST_CORE.md`
//! and `docs/RUST_KNN_BASELINE.md` for scope and validation.

#![forbid(unsafe_code)]

pub mod input_support;

pub mod numeric_bytes;

/// Standalone numeric regression baseline; not wired into the Python workflow.
pub mod knn;
