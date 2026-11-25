//! Exact composite-identity joins for measurement features and separately
//! supplied training endpoints. No model fitting or inferred/missing labels.
use std::collections::{BTreeMap, BTreeSet};
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
    // Keep borrowed tuple keys while checking the rosters. Only the returned
    // rows need to own their identities and group prefixes.
    let mut by_feature = BTreeMap::new();
    for row in features {
        if !valid_id(&row.identity, width) {
            return Err(JoinError::InvalidIdentity);
        }
        if row.values.len() != n_features || row.values.iter().any(|v| !v.is_finite()) {
            return Err(JoinError::InvalidFeatures);
        }
        if by_feature.insert(row.identity.as_slice(), row).is_some() {
            return Err(JoinError::DuplicateIdentity);
        }
    }
    let mut by_partition = BTreeMap::new();
    for row in partitions {
        if !valid_id(&row.identity, width) {
            return Err(JoinError::InvalidIdentity);
        }
        if by_partition
            .insert(row.identity.as_slice(), row.partition)
            .is_some()
        {
            return Err(JoinError::DuplicateIdentity);
        }
    }
    if by_partition.keys().ne(by_feature.keys()) {
        return Err(JoinError::PartitionMismatch);
    }
    let training_keys = || {
        by_partition
            .iter()
            .filter_map(|(&key, &partition)| (partition == Partition::Train).then_some(key))
    };
    let train_count = training_keys().count();
    if train_count == 0 || train_count == by_partition.len() {
        return Err(JoinError::EmptyPartition);
    }
    let train_groups: BTreeSet<_> = training_keys().map(|key| &key[..group_prefix]).collect();
    if by_partition.iter().any(|(&key, &partition)| {
        partition == Partition::Predict && train_groups.contains(&key[..group_prefix])
    }) {
        return Err(JoinError::GroupOverlap);
    }
    let mut targets = BTreeMap::new();
    for row in endpoints {
        if !valid_id(&row.identity, width) {
            return Err(JoinError::InvalidIdentity);
        }
        if !row.target.is_finite() {
            return Err(JoinError::InvalidEndpoint);
        }
        if targets.insert(row.identity.as_slice(), row.target).is_some() {
            return Err(JoinError::DuplicateIdentity);
        }
    }
    if targets.keys().copied().ne(training_keys()) {
        return Err(JoinError::EndpointMismatch);
    }
    let mut training = Vec::with_capacity(train_count);
    let mut prediction = Vec::with_capacity(by_partition.len() - train_count);
    for (key, row) in by_feature {
        let prepared = PreparedRow {
            group: key[..group_prefix].to_vec(),
            values: row.values.clone(),
            target: targets.get(key).copied(),
            identity: key.to_vec(),
        };
        match by_partition[key] {
            Partition::Train => training.push(prepared),
            Partition::Predict => prediction.push(prepared),
        }
    }
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
    let mut positions = BTreeMap::new();
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
        let mut assigned = vec![None; width];
        for (name, value) in &row.values {
            if !valid_id(std::slice::from_ref(name), 1) {
                return Err(JoinError::InvalidFeatures);
            }
            let i = *positions
                .get(name.as_str())
                .ok_or(JoinError::InvalidFeatures)?;
            if !value.is_finite() || assigned[i].replace(*value).is_some() {
                return Err(JoinError::InvalidFeatures);
            }
        }
        let values = assigned
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(JoinError::InvalidFeatures)?;
        aligned.push(DeviceFeatures {
            identity: row.identity.clone(),
            values,
        });
    }
    prepare_measurements(&aligned, partitions, endpoints, group_prefix)
}
