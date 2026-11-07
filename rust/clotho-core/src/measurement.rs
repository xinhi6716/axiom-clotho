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
    let mut by_feature = BTreeMap::new();
    for row in features {
        if !valid_id(&row.identity, width) {
            return Err(JoinError::InvalidIdentity);
        }
        if row.values.len() != n_features || row.values.iter().any(|v| !v.is_finite()) {
            return Err(JoinError::InvalidFeatures);
        }
        if by_feature.insert(row.identity.clone(), row).is_some() {
            return Err(JoinError::DuplicateIdentity);
        }
    }
    let mut by_partition = BTreeMap::new();
    for row in partitions {
        if !valid_id(&row.identity, width) {
            return Err(JoinError::InvalidIdentity);
        }
        if by_partition
            .insert(row.identity.clone(), row.partition)
            .is_some()
        {
            return Err(JoinError::DuplicateIdentity);
        }
    }
    if by_partition.keys().ne(by_feature.keys()) {
        return Err(JoinError::PartitionMismatch);
    }
    let train_keys: BTreeSet<_> = by_partition
        .iter()
        .filter(|(_, p)| **p == Partition::Train)
        .map(|(k, _)| k.clone())
        .collect();
    let prediction_keys: BTreeSet<_> = by_partition
        .iter()
        .filter(|(_, p)| **p == Partition::Predict)
        .map(|(k, _)| k.clone())
        .collect();
    if train_keys.is_empty() || prediction_keys.is_empty() {
        return Err(JoinError::EmptyPartition);
    }
    let train_groups: BTreeSet<_> = train_keys
        .iter()
        .map(|k| k[..group_prefix].to_vec())
        .collect();
    if prediction_keys
        .iter()
        .any(|k| train_groups.contains(&k[..group_prefix]))
    {
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
        if targets.insert(row.identity.clone(), row.target).is_some() {
            return Err(JoinError::DuplicateIdentity);
        }
    }
    if targets.keys().ne(train_keys.iter()) {
        return Err(JoinError::EndpointMismatch);
    }
    let mut training = Vec::new();
    let mut prediction = Vec::new();
    for (key, row) in by_feature {
        let prepared = PreparedRow {
            group: key[..group_prefix].to_vec(),
            values: row.values.clone(),
            target: targets.get(&key).copied(),
            identity: key.clone(),
        };
        match by_partition[&key] {
            Partition::Train => training.push(prepared),
            Partition::Predict => prediction.push(prepared),
        }
    }
    Ok(PreparedMeasurements {
        training,
        prediction,
    })
}
