//! Opt-in raw allocation guard, before the CSV parser materializes String cells.
use super::{table_bytes_with_limit, Result, Table};
use clotho_core::regression::MAX_TRAINING_ROWS;
use std::fs::File;
use std::io::Read;

const MAX_BYTES: u64 = 16 * 1024 * 1024;
const MAX_FIELDS: usize = 128;
const MAX_CELLS: usize = 1_000_000;

pub(super) fn load(path: &str) -> Result<Table> {
    let mut bytes = Vec::new();
    File::open(path)?.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("Large-training input exceeds 16 MiB raw byte limit".into());
    }
    preflight(&bytes)?;
    table_bytes_with_limit(&bytes, MAX_TRAINING_ROWS)
}

// Strict lexical scan uses constant auxiliary memory. BOM/quoted field decoding
// remains the CSV parser's job; the scan never copies or normalizes cells.
fn preflight(bytes: &[u8]) -> Result<()> {
    std::str::from_utf8(bytes).map_err(|_| "Large-training CSV must be UTF-8")?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    #[derive(Clone, Copy)]
    enum State { Start, Plain, Quoted, Closed }
    let mut state = State::Start;
    let (mut records, mut cells, mut fields, mut started, mut i) = (0, 0, 1, false, 0);
    while i < bytes.len() {
        let byte = bytes[i];
        if matches!(state, State::Quoted) {
            if byte == b'"' { state = State::Closed; }
            i += 1;
            continue;
        }
        if byte == b'\n' || byte == b'\r' {
            if !started { return Err("Blank records are not allowed in large-training CSV".into()); }
            if byte == b'\r' {
                if bytes.get(i + 1) != Some(&b'\n') {
                    return Err("Large-training record endings must be LF or CRLF".into());
                }
                i += 1;
            }
            finish(fields, &mut records, &mut cells)?;
            fields = 1;
            state = State::Start;
            started = false;
        } else if byte == b',' {
            fields += 1;
            if fields > MAX_FIELDS { return Err("Large-training CSV exceeds 128 raw fields per record".into()); }
            state = State::Start;
            started = true;
        } else {
            state = match (state, byte) {
                (State::Start, b'"') | (State::Closed, b'"') => State::Quoted,
                (State::Plain, b'"') | (State::Closed, _) => return Err("Malformed large-training CSV quoting".into()),
                _ => State::Plain,
            };
            started = true;
        }
        i += 1;
    }
    if matches!(state, State::Quoted) { return Err("Unclosed large-training CSV quote".into()); }
    if started { finish(fields, &mut records, &mut cells)?; }
    Ok(())
}

fn finish(fields: usize, records: &mut usize, cells: &mut usize) -> Result<()> {
    *records += 1;
    *cells += fields;
    if *records > MAX_TRAINING_ROWS + 1 || *cells > MAX_CELLS {
        return Err("Large-training CSV exceeds raw row/cell budgets".into());
    }
    Ok(())
}
