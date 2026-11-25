//! Wire framing shared by explicit-SI native device CSV workflows.

// The CSV reader is intentionally permissive about quote syntax. Require
// complete wire framing before interpreting or hashing a measurement capture.
pub(super) fn has_complete_csv_quoting(raw: &[u8]) -> bool {
    #[derive(Clone, Copy)]
    enum Field {
        Start,
        Bare,
        Quoted,
        AfterQuote,
    }
    use Field::*;
    let mut field = Start;
    for &byte in raw.strip_prefix(b"\xef\xbb\xbf").unwrap_or(raw) {
        field = match (field, byte) {
            (Quoted, b'"') => AfterQuote,
            (Quoted, _) => Quoted,
            (Start | AfterQuote, b'"') => Quoted,
            (_, b',' | b'\r' | b'\n') => Start,
            (AfterQuote, _) | (Bare, b'"') => return false,
            _ => Bare,
        };
    }
    !matches!(field, Quoted)
}
