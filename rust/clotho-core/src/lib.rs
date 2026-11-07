//! Incremental Rust core: source-feature diagnostics, numeric regression, and
//! group-disjoint evaluation. The additive native CLI uses these algorithms
//! without Python; legacy Python workflows are not fully migrated or replaced.
//! See `docs/RUST_NATIVE_WORKFLOW.md` and `docs/RUST_MIGRATION_STATUS.md`.

#![forbid(unsafe_code)]

pub mod evaluation;
pub mod input_support;
pub mod knn;
pub mod numeric_bytes;
pub mod regression;
