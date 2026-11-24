//! Construct finite group-max prediction intervals on a declared query roster.
//! Group IDs are declarations, not authenticated physical identity. Coverage
//! requires the assumptions in [crate::group_max_calibration], including
//! consistent roster construction; the code cannot verify those assumptions.
//! Finite intervals conservatively contain every finite binary64 observation
//! accepted by the rounded residual-score threshold; they may be wider than
//! that set.
use crate::group_max_calibration::{
    calibrate_group_max, valid_key, CalibrationRow, GroupMaxCutoff, GroupMaxError, RowKey,
    MAX_CALIBRATION_ROWS,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct QueryCenter<'a> {
    pub key: RowKey<'a>,
    pub center: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryInterval<'a> {
    pub key: RowKey<'a>,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GroupMaxIntervals<'a> {
    Finite {
        groups: usize,
        rank: usize,
        radius: f64,
        intervals: Vec<QueryInterval<'a>>,
    },
    InsufficientFinite { groups: usize, rank: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryIntervalError {
    Calibration(GroupMaxError),
    QueryCapacity,
    EmptyQueryRoster,
    InvalidQueryKey,
    QueryRosterMismatch,
    GroupOverlap,
    NonfiniteCenter,
    NonfiniteBound,
}

fn step(x: f64, upward: bool) -> f64 {
    if x == 0.0 {
        let subnormal = f64::from_bits(1);
        return if upward { subnormal } else { -subnormal };
    }
    let bits = x.to_bits();
    f64::from_bits(if (x > 0.0) == upward { bits + 1 } else { bits - 1 })
}

// With |a| >= |b|, finite round-to-nearest s, and gradual underflow,
// FastTwoSum gives the exact rounding residual b - (s - a). All arithmetic
// here is binary64, with no fast-math or fused operation.
fn outward_sum(x: f64, y: f64, upward: bool) -> Result<f64, QueryIntervalError> {
    let (a, b) = if x.abs() >= y.abs() { (x, y) } else { (y, x) };
    let s = a + b;
    if !s.is_finite() {
        return Err(QueryIntervalError::NonfiniteBound);
    }
    let error = b - (s - a);
    let bound = if (upward && error > 0.0) || (!upward && error < 0.0) {
        step(s, upward)
    } else {
        s
    };
    if bound.is_finite() { Ok(bound) } else { Err(QueryIntervalError::NonfiniteBound) }
}

/// Validate an independently declared query roster and return bounds in that
/// roster's order. Query inputs have no observations. Calibration and query
/// group strings must be disjoint, even if their row components differ.
/// Other separations (such as model-training groups) remain caller obligations.
/// Check all query keys and centers even when the rank has no finite cutoff.
pub fn calibrate_group_max_intervals<'a>(
    calibration_roster: &[RowKey<'_>],
    calibration_rows: &[CalibrationRow<'_>],
    query_roster: &[RowKey<'a>],
    query_centers: &[QueryCenter<'_>],
    numerator: u64,
    denominator: u64,
) -> Result<GroupMaxIntervals<'a>, QueryIntervalError> {
    if query_roster.len() > MAX_CALIBRATION_ROWS || query_centers.len() > MAX_CALIBRATION_ROWS {
        return Err(QueryIntervalError::QueryCapacity);
    }
    if query_roster.is_empty() {
        return Err(QueryIntervalError::EmptyQueryRoster);
    }
    if !query_roster.iter().copied().chain(query_centers.iter().map(|r| r.key)).all(valid_key) {
        return Err(QueryIntervalError::InvalidQueryKey);
    }
    let expected: BTreeSet<_> = query_roster.iter().copied().collect();
    let supplied: BTreeMap<_, _> = query_centers.iter().map(|r| (r.key, r.center)).collect();
    if expected.len() != query_roster.len() || supplied.len() != query_centers.len()
        || !expected.iter().copied().eq(supplied.keys().copied()) {
        return Err(QueryIntervalError::QueryRosterMismatch);
    }
    if !supplied.values().all(|x| x.is_finite()) {
        return Err(QueryIntervalError::NonfiniteCenter);
    }
    let cutoff = calibrate_group_max(calibration_roster, calibration_rows, numerator, denominator)
        .map_err(QueryIntervalError::Calibration)?;
    let calibration_groups: BTreeSet<_> = calibration_roster.iter().map(|r| r.group).collect();
    if query_roster.iter().any(|r| calibration_groups.contains(r.group)) {
        return Err(QueryIntervalError::GroupOverlap);
    }
    match cutoff {
        GroupMaxCutoff::InsufficientFinite { groups, rank } => {
            Ok(GroupMaxIntervals::InsufficientFinite { groups, rank })
        }
        GroupMaxCutoff::Finite { groups, rank, radius } => {
            // A positive finite rounded score <= radius implies the exact
            // absolute residual is below the next greater binary64 number.
            // Use that conservative radius, and outward-round both endpoints.
            // For zero, distinct finite binary64 values differ by at least
            // the minimum subnormal, so their rounded subtraction is nonzero.
            let guard = if radius == 0.0 { radius } else { step(radius, true) };
            if !guard.is_finite() {
                return Err(QueryIntervalError::NonfiniteBound);
            }
            let intervals = query_roster.iter().copied().map(|key| {
                let center = supplied[&key];
                // Preserve even a signed-zero center for a point cutoff.
                let (lower, upper) = if guard == 0.0 {
                    (center, center)
                } else {
                    (outward_sum(center, -guard, false)?, outward_sum(center, guard, true)?)
                };
                Ok(QueryInterval { key, lower, upper })
            }).collect::<Result<_, _>>()?;
            Ok(GroupMaxIntervals::Finite { groups, rank, radius, intervals })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: RowKey<'static> = RowKey { group: "train", row: "1" };
    const B: RowKey<'static> = RowKey { group: "held", row: "1" };
    const C: RowKey<'static> = RowKey { group: "held", row: "2" };
    const CAL: CalibrationRow<'static> = CalibrationRow {
        key: A, observed: f64::EPSILON / 4.0, prediction: 0.0,
    };

    #[test]
    fn matches_query_keys_and_emits_roster_order_with_outward_bounds() {
        let q = [QueryCenter { key: C, center: -1.0 }, QueryCenter { key: B, center: 1.0 }];
        let result = calibrate_group_max_intervals(&[A], &[CAL], &[B, C], &q, 1, 2).unwrap();
        let plus = f64::from_bits(1.0f64.to_bits() + 1);
        let minus = f64::from_bits(1.0f64.to_bits() - 1);
        assert_eq!(result, GroupMaxIntervals::Finite {
            groups: 1, rank: 1, radius: f64::EPSILON / 4.0,
            intervals: vec![
                QueryInterval { key: B, lower: minus, upper: plus },
                QueryInterval { key: C, lower: -plus, upper: -minus },
            ],
        });
        assert_eq!(outward_sum(1.0, 1.0, true), Ok(2.0));
        assert_eq!(outward_sum(0.0, f64::from_bits(1), true), Ok(f64::from_bits(1)));
    }

    #[test]
    fn insufficient_still_validates_binding_and_group_overlap() {
        let q = QueryCenter { key: B, center: 1.0 };
        assert_eq!(calibrate_group_max_intervals(&[A], &[CAL], &[B], &[q], 19, 20),
            Ok(GroupMaxIntervals::InsufficientFinite { groups: 1, rank: 2 }));
        assert_eq!(calibrate_group_max_intervals(&[A], &[CAL], &[B, C], &[q,q], 19, 20),
            Err(QueryIntervalError::QueryRosterMismatch));
        let repeated_group = RowKey { group: A.group, row: "different" };
        let r = QueryCenter { key: repeated_group, center: 1.0 };
        assert_eq!(calibrate_group_max_intervals(&[A], &[CAL], &[repeated_group], &[r], 19, 20),
            Err(QueryIntervalError::GroupOverlap));
        let bad = QueryCenter { center: f64::NAN, ..q };
        assert_eq!(calibrate_group_max_intervals(&[A], &[CAL], &[B], &[bad], 19, 20),
            Err(QueryIntervalError::NonfiniteCenter));
    }

    #[test]
    fn finite_rounded_sum_can_have_no_finite_outward_bound() {
        let q = QueryCenter { key: B, center: f64::MAX };
        assert_eq!(calibrate_group_max_intervals(&[A], &[CAL], &[B], &[q], 1, 2),
            Err(QueryIntervalError::NonfiniteBound));
        let zero_cal = CalibrationRow { observed: 0.0, ..CAL };
        let z = QueryCenter { key: B, center: -0.0 };
        let result = calibrate_group_max_intervals(&[A], &[zero_cal], &[B], &[z], 1, 2).unwrap();
        match result {
            GroupMaxIntervals::Finite { intervals, .. } => {
                assert!(intervals[0].lower.is_sign_negative());
                assert!(intervals[0].upper.is_sign_negative());
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn contains_rounded_tie_hidden_by_cancellation_and_refuses_max_guard() {
        let cal = CalibrationRow { observed: 1.0, ..CAL };
        let q = QueryCenter { key: B, center: -1.0 };
        let y = f64::EPSILON / 4.0;
        assert_eq!((y - q.center).abs(), 1.0);
        let result = calibrate_group_max_intervals(&[A], &[cal], &[B], &[q], 1, 2).unwrap();
        match result {
            GroupMaxIntervals::Finite { intervals, radius, .. } => {
                assert_eq!(radius, 1.0);
                assert!(intervals[0].lower <= y && y <= intervals[0].upper);
            }
            other => panic!("unexpected result: {other:?}"),
        }
        let max = CalibrationRow { observed: f64::MAX, ..CAL };
        assert_eq!(calibrate_group_max_intervals(&[A], &[max], &[B], &[q], 1, 2),
            Err(QueryIntervalError::NonfiniteBound));
        let smallest = f64::from_bits(1);
        assert_ne!((smallest - 0.0).abs(), 0.0);
        assert_ne!((f64::MIN_POSITIVE - f64::from_bits(f64::MIN_POSITIVE.to_bits()-1)).abs(), 0.0);
    }
}
