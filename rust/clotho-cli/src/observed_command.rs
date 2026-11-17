//! Label-separated, exact-roster observed-series preparation, without fitting.
use super::{create_file, predict_saved::Export, read_bounded, required, table_bytes, Result};
use std::collections::HashMap;
use std::path::Path;

pub(super) fn run(opts: &HashMap<String, String>) -> Result<()> {
    const ALLOWED: [&str; 5] = ["--query", "--observed", "--row-id", "--saved-contract", "--out"];
    if opts.keys().any(|key| !ALLOWED.contains(&key.as_str())) {
        return Err("Unknown option for prepare-observed".into());
    }
    let output = Path::new(required(opts, "--out")?);
    if output.symlink_metadata().is_ok() {
        return Err("Output already exists; choose a new path".into());
    }
    let query_bytes = read_bounded(required(opts, "--query")?)?;
    validate_csv(&query_bytes)?;
    let query = table_bytes(&query_bytes)?;
    let label_bytes = read_bounded(required(opts, "--observed")?)?;
    validate_csv(&label_bytes)?;
    let labels = table_bytes(&label_bytes)?;
    let bytes = Export::observed(
        required(opts, "--saved-contract")?,
        required(opts, "--row-id")?,
        &query,
        &labels,
        &query_bytes,
    )?;
    create_file(output, &bytes)?;
    println!("Wrote observed series aligned to the declared query roster; no provenance verification");
    Ok(())
}

// csv::Reader deliberately tolerates malformed quoting and skips empty records.
// This command must not silently repair an ID or discard a record. Validate the
// bounded byte buffer before parsing it, without changing bytes used for hashing.
fn validate_csv(bytes: &[u8]) -> Result<()> {
    std::str::from_utf8(bytes).map_err(|_| "Observed workflow CSV must be valid UTF-8")?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    #[derive(Clone, Copy)]
    enum State { Start, Plain, Quoted, Closed }
    let mut state = State::Start;
    let mut record_started = false;
    let mut i = 0;
    while i < bytes.len() {
        let byte = bytes[i];
        if matches!(state, State::Quoted) {
            if byte == b'"' {
                state = State::Closed;
            }
            i += 1;
            continue;
        }
        if byte == b'\n' || byte == b'\r' {
            if !record_started {
                return Err("Blank CSV records are not allowed in the observed workflow".into());
            }
            if byte == b'\r' {
                if bytes.get(i + 1) != Some(&b'\n') {
                    return Err("CSV record endings must be LF or CRLF".into());
                }
                i += 1;
            }
            state = State::Start;
            record_started = false;
        } else if byte == b',' {
            state = State::Start;
            record_started = true;
        } else {
            state = match (state, byte) {
                (State::Start, b'"') | (State::Closed, b'"') => State::Quoted,
                (State::Plain, b'"') | (State::Closed, _) => {
                    return Err("Malformed CSV quoting in observed workflow".into());
                }
                _ => State::Plain,
            };
            record_started = true;
        }
        i += 1;
    }
    if matches!(state, State::Quoted) {
        return Err("Unclosed CSV quote in observed workflow".into());
    }
    Ok(())
}
