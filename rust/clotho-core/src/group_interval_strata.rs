//! Observed interval coverage by exact declared group size.
//! No pooling across sizes or inferential claims.
use crate::group_interval_diagnostics::{
    evaluate_group_intervals_by_group, GroupIntervalDiagnostic, IntervalDiagnosticError, IntervalObservation,
};
use crate::group_max_calibration::RowKey;
use crate::group_max_intervals::GroupMaxIntervals;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupSizeCoverage {
    pub rows_per_group: usize,
    pub groups: usize,
    pub groups_fully_covered: usize,
    pub rows_covered: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StratifiedIntervalCoverage {
    pub overall: GroupIntervalDiagnostic,
    /// Sorted by exact group size. Absent when no finite cutoff was returned.
    pub strata: Option<Vec<GroupSizeCoverage>>,
}

/// Describe coverage on the same complete roster as the main diagnostic.
/// Group size is the number of declared rows, not the count of covered rows.
pub fn stratify_group_interval_coverage(
    roster: &[RowKey<'_>],
    prediction: &GroupMaxIntervals<'_>,
    observations: &[IntervalObservation<'_>],
) -> Result<StratifiedIntervalCoverage, IntervalDiagnosticError> {
    // Small equal-size group rosters are common on bounded evaluation
    // cohorts. Count those sizes directly; retain sparse storage for large
    // groups so one unusual group never allocates a dense histogram.
    const DIRECT_SIZES: usize = 16;
    let mut direct = [[0_usize; 3]; DIRECT_SIZES];
    let mut overflow = BTreeMap::new();
    let overall = evaluate_group_intervals_by_group(roster, prediction, observations, |len, covered| {
        let slot = if len <= DIRECT_SIZES {
            &mut direct[len - 1]
        } else {
            overflow.entry(len).or_insert([0_usize; 3])
        };
        slot[0] += 1;
        slot[1] += usize::from(len == covered);
        slot[2] += covered;
    })?;
    let strata = match overall {
        GroupIntervalDiagnostic::InsufficientFinite { .. } => None,
        GroupIntervalDiagnostic::Scored { .. } => {
            let direct_count = direct.iter().filter(|count| count[0] != 0).count();
            let mut result = Vec::with_capacity(direct_count + overflow.len());
            result.extend(
                direct.into_iter().enumerate()
                    .filter_map(|(i, count)| (count[0] != 0).then_some((i + 1, count)))
                    .chain(overflow)
                    .map(|(rows_per_group, [groups, groups_fully_covered, rows_covered])| GroupSizeCoverage {
                        rows_per_group, groups, groups_fully_covered, rows_covered,
                    }),
            );
            Some(result)
        }
    };
    Ok(StratifiedIntervalCoverage { overall, strata })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group_max_intervals::QueryInterval;

    #[test]
    fn direct_and_sparse_sizes_share_exact_order_and_coverage_counts() {
        let mut owned = Vec::new();
        for (group, size) in [("z", 17), ("y", 1), ("x", 16), ("w", 17), ("v", 40)] {
            owned.extend((0..size).map(|i| (group, format!("{i:03}"))));
        }
        let keys: Vec<_> = owned.iter().map(|(group, row)| RowKey { group, row }).collect();
        let observations: Vec<_> = keys.iter().rev().map(|&key| IntervalObservation {
            key,
            value: if key.group == "w" && key.row == "016" { 2.0 } else { 0.0 },
        }).collect();
        let prediction = GroupMaxIntervals::Finite {
            groups: 1, rank: 1, radius: 1.0,
            intervals: keys.iter().copied().map(|key| QueryInterval { key, lower: 0.0, upper: 1.0 }).collect(),
        };
        let actual = stratify_group_interval_coverage(&keys, &prediction, &observations).unwrap();
        assert_eq!(actual.strata.unwrap(), [
            GroupSizeCoverage { rows_per_group: 1, groups: 1, groups_fully_covered: 1, rows_covered: 1 },
            GroupSizeCoverage { rows_per_group: 16, groups: 1, groups_fully_covered: 1, rows_covered: 16 },
            GroupSizeCoverage { rows_per_group: 17, groups: 2, groups_fully_covered: 1, rows_covered: 33 },
            GroupSizeCoverage { rows_per_group: 40, groups: 1, groups_fully_covered: 1, rows_covered: 40 },
        ]);
    }

    #[test]
    fn shared_scoring_keeps_sizes_sides_and_endpoints_on_permuted_rosters() {
        let a = RowKey { group: "a:b", row: "c" };
        let b = RowKey { group: "a", row: "b:c" };
        let c = RowKey { group: "a", row: "d" };
        let d = RowKey { group: "z", row: "one" };
        let e = RowKey { group: "z", row: "two" };
        let roster = [d, a, c, b, e];
        let observations = [
            IntervalObservation { key: e, value: 1.0 },
            IntervalObservation { key: b, value: -1.0 },
            IntervalObservation { key: a, value: -2.0 },
            IntervalObservation { key: d, value: -1.0 },
            IntervalObservation { key: c, value: 2.0 },
        ];
        let mut prediction = GroupMaxIntervals::Finite {
            groups: 4,
            rank: 4,
            radius: 1.0,
            intervals: [a, e, b, d, c].into_iter().map(|key| QueryInterval {
                key, lower: -1.0, upper: 1.0,
            }).collect(),
        };
        let result = stratify_group_interval_coverage(&roster, &prediction, &observations).unwrap();
        assert_eq!(result.overall, GroupIntervalDiagnostic::Scored {
            rows: 5, rows_covered: 3, groups: 3, groups_fully_covered: 1,
            below_rows: 1, above_rows: 1,
        });
        assert_eq!(result.strata, Some(vec![
            GroupSizeCoverage { rows_per_group: 1, groups: 1, groups_fully_covered: 0, rows_covered: 0 },
            GroupSizeCoverage { rows_per_group: 2, groups: 2, groups_fully_covered: 1, rows_covered: 3 },
        ]));
        let GroupMaxIntervals::Finite { intervals, .. } = &mut prediction else {
            unreachable!()
        };
        intervals[0].lower = f64::NAN;
        let mut invalid = observations;
        invalid[4].value = f64::INFINITY;
        assert_eq!(
            stratify_group_interval_coverage(&roster, &prediction, &invalid),
            Err(IntervalDiagnosticError::NonfiniteObservation),
        );
        assert_eq!(
            stratify_group_interval_coverage(&roster, &prediction, &observations),
            Err(IntervalDiagnosticError::InvalidBounds),
        );
    }

    #[test]
    fn one_miss_in_large_group_preserves_size_and_denominators() {
        let a = RowKey { group: "short", row: "1" };
        let b = RowKey { group: "long", row: "1" };
        let c = RowKey { group: "long", row: "2" };
        let rows = [a, b, c];
        let observations = [
            IntervalObservation { key: c, value: 2.0 },
            IntervalObservation { key: b, value: 1.0 },
            IntervalObservation { key: a, value: 0.0 },
        ];
        let prediction = GroupMaxIntervals::Finite {
            groups: 1, rank: 1, radius: 1.0,
            intervals: rows.into_iter().rev().map(|key| QueryInterval {
                key, lower: 0.0, upper: 1.0,
            }).collect(),
        };
        let result = stratify_group_interval_coverage(&rows, &prediction, &observations).unwrap();
        assert_eq!(result.strata, Some(vec![
            GroupSizeCoverage { rows_per_group: 1, groups: 1, groups_fully_covered: 1, rows_covered: 1 },
            GroupSizeCoverage { rows_per_group: 2, groups: 1, groups_fully_covered: 0, rows_covered: 1 },
        ]));
        assert_eq!(stratify_group_interval_coverage(&rows, &prediction, &observations[..2]),
            Err(IntervalDiagnosticError::RosterMismatch));
        let insufficient = GroupMaxIntervals::InsufficientFinite { groups: 1, rank: 2 };
        assert_eq!(stratify_group_interval_coverage(&rows, &insufficient, &observations).unwrap().strata, None);
    }
}
