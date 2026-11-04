//! Incremental Rust core. The existing Python workflow remains authoritative.
//!
//! This first port compares already-normalized source features, without fitting,
//! imputation, inference, file access or network access. See `docs/RUST_CORE.md`.

#![forbid(unsafe_code)]

pub mod input_support;

pub mod numeric_bytes;
