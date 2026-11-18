//! Guard against silent decimal nonzero-to-zero conversion before serde.
use super::Result;

// Quote-aware check before serde conversion. Serde remains responsible for
// JSON grammar, strict fields and duplicate keys; no raw values are echoed.
pub(super) fn reject_numeric_underflow(bytes: &[u8]) -> Result<()> {
    let (mut i, mut in_string) = (0, false);
    while i < bytes.len() {
        if in_string {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == b'"' {
                in_string = false;
            }
        } else if bytes[i] == b'"' {
            in_string = true;
        } else if bytes[i].is_ascii_digit() || bytes[i] == b'-' {
            let start = i;
            while i < bytes.len()
                && matches!(bytes[i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
            {
                i += 1;
            }
            let token = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
            if let Ok(value) = token.parse::<f64>() {
                let mantissa = token.split(['e', 'E']).next().unwrap_or("");
                if value == 0.0 && mantissa.bytes().any(|b| (b'1'..=b'9').contains(&b)) {
                    return Err("Nonzero JSON number underflows binary64".into());
                }
            }
            continue;
        }
        i += 1;
    }
    Ok(())
}
