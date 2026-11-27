//! Exact composite-identity joins for measurement features and separately
//! supplied training endpoints. No model fitting or inferred/missing labels.
use std::collections::{BTreeMap, HashMap};
use std::fmt;
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceFeatures {
    pub identity: Vec<String>,
    pub values: Vec<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    Train,
    Predict,
}
#[derive(Clone, Debug)]
pub struct DevicePartition {
    pub identity: Vec<String>,
    pub partition: Partition,
}
#[derive(Clone, Debug)]
pub struct TrainingEndpoint {
    pub identity: Vec<String>,
    pub target: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedRow {
    pub identity: Vec<String>,
    pub group: Vec<String>,
    pub values: Vec<f64>,
    pub target: Option<f64>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedMeasurements {
    pub training: Vec<PreparedRow>,
    pub prediction: Vec<PreparedRow>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinError {
    ResourceLimit,
    InvalidIdentity,
    DuplicateIdentity,
    InvalidFeatures,
    InvalidEndpoint,
    PartitionMismatch,
    EndpointMismatch,
    EmptyPartition,
    GroupOverlap,
    InvalidGrouping,
}
impl fmt::Display for JoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Measurement join rejected: {:?}", self)
    }
}
impl std::error::Error for JoinError {}
fn valid_id(key: &[String], width: usize) -> bool {
    key.len() == width
        && key.iter().all(|s| {
            !s.is_empty()
                && s.trim() == s
                && s.chars().count() <= 128
                && !s.chars().any(char::is_control)
        })
}
enum OrderedRoster<'a, T> {
    Borrowed(&'a [T], bool),
    Indexed(BTreeMap<&'a [String], &'a T>),
}

enum RosterIter<'i, 'a, T> {
    Borrowed(std::slice::Iter<'a, T>),
    Reversed(std::iter::Rev<std::slice::Iter<'a, T>>),
    Indexed(std::collections::btree_map::Values<'i, &'a [String], &'a T>),
}

impl<'a, T> Iterator for RosterIter<'_, 'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Borrowed(rows) => rows.next(),
            Self::Reversed(rows) => rows.next(),
            Self::Indexed(rows) => rows.next().copied(),
        }
    }
}

impl<'a, T> OrderedRoster<'a, T> {
    fn iter(&self) -> RosterIter<'_, 'a, T> {
        match self {
            Self::Borrowed(rows, false) => RosterIter::Borrowed(rows.iter()),
            Self::Borrowed(rows, true) => RosterIter::Reversed(rows.iter().rev()),
            Self::Indexed(rows) => RosterIter::Indexed(rows.values()),
        }
    }
}

fn checked_roster<'a, T>(
    rows: &'a [T],
    key: impl Fn(&T) -> &[String],
    check: impl Fn(&T) -> Result<(), JoinError>,
) -> Result<OrderedRoster<'a, T>, JoinError> {
    let descending = rows.first().map(&key) > rows.last().map(&key);
    if rows.windows(2).all(|w| {
        if descending { key(&w[0]) > key(&w[1]) } else { key(&w[0]) < key(&w[1]) }
    }) {
        for row in rows {
            check(row)?;
        }
        Ok(OrderedRoster::Borrowed(rows, descending))
    } else {
        let mut index = BTreeMap::new();
        for row in rows {
            // Validate and check duplicates in arrival order. Doing all the
            // value checks before indexing would change error precedence.
            check(row)?;
            if index.insert(key(row), row).is_some() {
                return Err(JoinError::DuplicateIdentity);
            }
        }
        Ok(OrderedRoster::Indexed(index))
    }
}

/// Output order is canonical lexicographic composite identity order within each
/// partition. `group_prefix` uses exact tuple components, never delimiter joins.
/// No acquisition, blinding, causal independence or source authenticity is
/// established by these identity checks. Caller supplies physically valid keys.
pub fn prepare_measurements(
    features: &[DeviceFeatures],
    partitions: &[DevicePartition],
    endpoints: &[TrainingEndpoint],
    group_prefix: usize,
) -> Result<PreparedMeasurements, JoinError> {
    if features.is_empty()
        || features.len() > 20000
        || partitions.len() > 20000
        || endpoints.len() > 20000
    {
        return Err(JoinError::ResourceLimit);
    }
    let width = features[0].identity.len();
    if width == 0 || width > 6 || group_prefix == 0 || group_prefix > width {
        return Err(JoinError::InvalidGrouping);
    }
    let n_features = features[0].values.len();
    if n_features == 0 || n_features > 100 {
        return Err(JoinError::InvalidFeatures);
    }
    // Strictly ordered tuple keys prove uniqueness without building trees.
    // Independently unordered rosters retain the original ordered index.
    let by_feature = checked_roster(features, |row| &row.identity, |row| {
        if !valid_id(&row.identity, width) {
            return Err(JoinError::InvalidIdentity);
        }
        if row.values.len() != n_features || row.values.iter().any(|v| !v.is_finite()) {
            return Err(JoinError::InvalidFeatures);
        }
        Ok(())
    })?;
    let by_partition = checked_roster(partitions, |row| &row.identity, |row| {
        if !valid_id(&row.identity, width) {
            return Err(JoinError::InvalidIdentity);
        }
        Ok(())
    })?;
    if by_partition.iter().map(|r| &r.identity).ne(by_feature.iter().map(|r| &r.identity)) {
        return Err(JoinError::PartitionMismatch);
    }
    let training_keys = || {
        by_partition.iter().filter_map(|row| {
            (row.partition == Partition::Train).then_some(row.identity.as_slice())
        })
    };
    let train_count = training_keys().count();
    if train_count == 0 || train_count == partitions.len() {
        return Err(JoinError::EmptyPartition);
    }
    // Lexicographic tuple order keeps every exact prefix contiguous. A
    // mixed group must contain an adjacent Train/Predict transition.
    let mut previous = None;
    for row in by_partition.iter() {
        let prefix = &row.identity[..group_prefix];
        if let Some((last_prefix, last_partition)) = previous {
            if last_partition != row.partition && last_prefix == prefix {
                return Err(JoinError::GroupOverlap);
            }
        }
        previous = Some((prefix, row.partition));
    }
    let targets = checked_roster(endpoints, |row| &row.identity, |row| {
        if !valid_id(&row.identity, width) {
            return Err(JoinError::InvalidIdentity);
        }
        if !row.target.is_finite() {
            return Err(JoinError::InvalidEndpoint);
        }
        Ok(())
    })?;
    if targets.iter().map(|r| r.identity.as_slice()).ne(training_keys()) {
        return Err(JoinError::EndpointMismatch);
    }
    let mut training = Vec::with_capacity(train_count);
    let mut prediction = Vec::with_capacity(partitions.len() - train_count);
    // The validated targets contain exactly the ordered train subsequence.
    let mut target_rows = targets.iter();
    for (row, partition) in by_feature.iter().zip(by_partition.iter()) {
        let key = &row.identity;
        debug_assert_eq!(key, &partition.identity);
        let target = match partition.partition {
            Partition::Train => {
                let target = target_rows.next().ok_or(JoinError::EndpointMismatch)?;
                debug_assert_eq!(key, &target.identity);
                Some(target.target)
            }
            Partition::Predict => None,
        };
        let prepared = PreparedRow {
            group: key[..group_prefix].to_vec(),
            values: row.values.clone(),
            target,
            identity: key.clone(),
        };
        match partition.partition {
            Partition::Train => training.push(prepared),
            Partition::Predict => prediction.push(prepared),
        }
    }
    debug_assert!(target_rows.next().is_none());
    Ok(PreparedMeasurements {
        training,
        prediction,
    })
}

/// One device's separately declared feature names and values.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedDeviceFeatures {
    pub identity: Vec<String>,
    pub values: Vec<(String, f64)>,
}

/// Align every row to the caller's frozen feature order before identity joining.
/// A name match does not establish units, measurement meaning or provenance.
pub fn prepare_named_measurements(
    feature_order: &[String],
    features: &[NamedDeviceFeatures],
    partitions: &[DevicePartition],
    endpoints: &[TrainingEndpoint],
    group_prefix: usize,
) -> Result<PreparedMeasurements, JoinError> {
    if features.is_empty()
        || features.len() > 20000
        || partitions.len() > 20000
        || endpoints.len() > 20000
    {
        return Err(JoinError::ResourceLimit);
    }
    let width = feature_order.len();
    if width == 0 || width > 100 {
        return Err(JoinError::InvalidFeatures);
    }
    let mut positions = HashMap::with_capacity(width);
    for (i, name) in feature_order.iter().enumerate() {
        if !valid_id(std::slice::from_ref(name), 1) || positions.insert(name.as_str(), i).is_some()
        {
            return Err(JoinError::InvalidFeatures);
        }
    }
    let identity_width = features[0].identity.len();
    if identity_width == 0
        || identity_width > 6
        || group_prefix == 0
        || group_prefix > identity_width
    {
        return Err(JoinError::InvalidGrouping);
    }
    let mut aligned = Vec::with_capacity(features.len());
    for row in features {
        if !valid_id(&row.identity, identity_width) {
            return Err(JoinError::InvalidIdentity);
        }
        if row.values.len() != width {
            return Err(JoinError::InvalidFeatures);
        }
        let mut values = vec![0.0; width];
        let mut assigned = 0_u128;
        for (declared, (name, value)) in row.values.iter().enumerate() {
            // An exact match to an already validated frozen name needs no
            // per-row text validation; matching the position avoids a lookup.
            let i = if name == &feature_order[declared] {
                declared
            } else {
                *positions.get(name.as_str()).ok_or(JoinError::InvalidFeatures)?
            };
            let bit = 1_u128 << i; // width is capped at 100 above.
            if !value.is_finite() || assigned & bit != 0 {
                return Err(JoinError::InvalidFeatures);
            }
            assigned |= bit;
            values[i] = *value;
        }
        // Exactly width distinct assignments into width slots cover all slots.
        aligned.push(DeviceFeatures {
            identity: row.identity.clone(),
            values,
        });
    }
    prepare_measurements(&aligned, partitions, endpoints, group_prefix)
}
