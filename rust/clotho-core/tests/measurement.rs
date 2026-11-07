use clotho_core::measurement::*;
fn key(l: &str, d: &str) -> Vec<String> {
    vec![l.into(), "W".into(), d.into()]
}
fn fixture() -> (
    Vec<DeviceFeatures>,
    Vec<DevicePartition>,
    Vec<TrainingEndpoint>,
) {
    (
        vec![
            DeviceFeatures {
                identity: key("B", "D"),
                values: vec![2.0],
            },
            DeviceFeatures {
                identity: key("A", "D"),
                values: vec![1.0],
            },
        ],
        vec![
            DevicePartition {
                identity: key("A", "D"),
                partition: Partition::Train,
            },
            DevicePartition {
                identity: key("B", "D"),
                partition: Partition::Predict,
            },
        ],
        vec![TrainingEndpoint {
            identity: key("A", "D"),
            target: 3.0,
        }],
    )
}
#[test]
fn exact_join_and_future_target_absence() {
    let (f, p, t) = fixture();
    let r = prepare_measurements(&f, &p, &t, 1).unwrap();
    assert_eq!(r.training[0].identity, key("A", "D"));
    assert_eq!(r.training[0].values, vec![1.0]);
    assert_eq!(r.training[0].target, Some(3.0));
    assert_eq!(r.prediction[0].target, None);
}
#[test]
fn row_permutations_are_invariant() {
    let (mut f, mut p, t) = fixture();
    let a = prepare_measurements(&f, &p, &t, 1).unwrap();
    f.reverse();
    p.reverse();
    assert_eq!(a, prepare_measurements(&f, &p, &t, 1).unwrap());
}
#[test]
fn future_endpoints_forbidden() {
    let (f, p, mut t) = fixture();
    t.push(TrainingEndpoint {
        identity: key("B", "D"),
        target: 7.0,
    });
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::EndpointMismatch
    );
}
#[test]
fn missing_endpoint_forbidden() {
    let (f, p, _) = fixture();
    assert_eq!(
        prepare_measurements(&f, &p, &[], 1).unwrap_err(),
        JoinError::EndpointMismatch
    );
}
#[test]
fn orphan_partition_forbidden() {
    let (f, mut p, t) = fixture();
    p[1].identity = key("C", "D");
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::PartitionMismatch
    );
}
#[test]
fn duplicate_sources_forbidden() {
    let (mut f, p, t) = fixture();
    f.push(f[0].clone());
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::DuplicateIdentity
    );
    let (f, mut p, t) = fixture();
    p.push(p[0].clone());
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::DuplicateIdentity
    );
    let (f, p, mut t) = fixture();
    t.push(t[0].clone());
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::DuplicateIdentity
    );
}
#[test]
fn physical_group_overlap_forbidden() {
    let (mut f, mut p, t) = fixture();
    f[0].identity = key("A", "D2");
    p[1].identity = key("A", "D2");
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::GroupOverlap
    );
    assert!(prepare_measurements(&f, &p, &t, 3).is_ok());
}
#[test]
fn exact_tuple_not_delimiter_concatenation() {
    let (mut f, mut p, mut t) = fixture();
    let a = vec!["A|B".into(), "C".into(), "D".into()];
    let b = vec!["A".into(), "B|C".into(), "D".into()];
    f[1].identity = a.clone();
    p[0].identity = a.clone();
    t[0].identity = a;
    f[0].identity = b.clone();
    p[1].identity = b;
    assert!(prepare_measurements(&f, &p, &t, 2).is_ok());
}
#[test]
fn nonfinite_features_and_targets_fail() {
    let (mut f, p, t) = fixture();
    f[0].values[0] = f64::NAN;
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::InvalidFeatures
    );
    let (f, p, mut t) = fixture();
    t[0].target = f64::INFINITY;
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::InvalidEndpoint
    );
}
#[test]
fn identity_width_and_padding_fail() {
    let (mut f, p, t) = fixture();
    f[0].identity.push("extra".into());
    assert!(prepare_measurements(&f, &p, &t, 1).is_err());
    let (mut f, p, t) = fixture();
    f[0].identity[0] = " secret ".into();
    let e = prepare_measurements(&f, &p, &t, 1).unwrap_err();
    assert_eq!(e, JoinError::InvalidIdentity);
    assert!(!e.to_string().contains("secret"));
}
#[test]
fn missing_partition_or_empty_cohort_fail() {
    let (f, mut p, t) = fixture();
    p.pop();
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::PartitionMismatch
    );
    assert_eq!(
        prepare_measurements(&[], &[], &[], 1).unwrap_err(),
        JoinError::ResourceLimit
    );
}
#[test]
fn ragged_feature_matrix_rejected() {
    let (mut f, p, t) = fixture();
    f[0].values.push(2.0);
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::InvalidFeatures
    );
}
#[test]
fn canonical_order_is_invariant_with_multiple_rows_per_partition() {
    let (mut f, mut p, mut t) = fixture();
    for (lot, device, partition, target) in [
        ("A", "D0", Partition::Train, Some(8.0)),
        ("B", "D0", Partition::Predict, None),
    ] {
        f.push(DeviceFeatures {
            identity: key(lot, device),
            values: vec![4.0],
        });
        p.push(DevicePartition {
            identity: key(lot, device),
            partition,
        });
        if let Some(target) = target {
            t.push(TrainingEndpoint {
                identity: key(lot, device),
                target,
            });
        }
    }
    let expected = prepare_measurements(&f, &p, &t, 1).unwrap();
    assert_eq!(
        expected
            .training
            .iter()
            .map(|r| r.identity[2].as_str())
            .collect::<Vec<_>>(),
        vec!["D", "D0"]
    );
    assert_eq!(
        expected
            .prediction
            .iter()
            .map(|r| r.identity[2].as_str())
            .collect::<Vec<_>>(),
        vec!["D", "D0"]
    );
    f.reverse();
    p.reverse();
    t.reverse();
    assert_eq!(expected, prepare_measurements(&f, &p, &t, 1).unwrap());
}
#[test]
fn grouping_and_empty_partitions_are_rejected() {
    let (f, mut p, t) = fixture();
    for prefix in [0, 4, usize::MAX] {
        assert_eq!(
            prepare_measurements(&f, &p, &t, prefix).unwrap_err(),
            JoinError::InvalidGrouping
        );
    }
    p[1].partition = Partition::Train;
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::EmptyPartition
    );
    p[0].partition = Partition::Predict;
    p[1].partition = Partition::Predict;
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::EmptyPartition
    );
}
#[test]
fn padded_partition_and_endpoint_keys_are_never_normalized() {
    let (f, mut p, t) = fixture();
    p[0].identity[0] = " A".into();
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::InvalidIdentity
    );
    let (f, p, mut t) = fixture();
    t[0].identity[0] = "A ".into();
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::InvalidIdentity
    );
}
#[test]
fn unrelated_endpoint_cannot_substitute_for_missing_training_endpoint() {
    let (f, p, mut t) = fixture();
    t[0].identity = key("C", "D");
    assert_eq!(
        prepare_measurements(&f, &p, &t, 1).unwrap_err(),
        JoinError::EndpointMismatch
    );
}
