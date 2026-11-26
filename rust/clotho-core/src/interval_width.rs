//! Compare a bounded binary64 acquisition width before returning its rounded value.

/// Inputs are prevalidated bounded finite values, lo <= hi, positive limit.
pub(crate) fn width_at_most(lo: f64, hi: f64, limit: f64) -> Option<f64> {
    let width = hi - lo;
    let too_wide = if width == limit {
        // Recover the exact subtraction residual (TwoDiff); a rounded
        // equality does not establish that the represented width fits.
        let lo_virtual = hi - width;
        let hi_virtual = width + lo_virtual;
        (hi - hi_virtual) + (lo_virtual - lo) > 0.0
    } else {
        width > limit
    };
    if too_wide {
        None
    } else {
        Some(width)
    }
}
