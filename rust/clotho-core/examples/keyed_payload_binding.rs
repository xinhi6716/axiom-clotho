//! Invented same-device join counterexample. No model fitting or external inputs.
//! cargo run --manifest-path rust/Cargo.toml -p clotho-core --example keyed_payload_binding
use std::collections::BTreeMap;

type Readout = (u8, u8); // (declared device key, invented endpoint)
const REFERENCE: [Readout; 5] = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)];

fn keyed(rows: &[Readout]) -> Option<BTreeMap<u8, u8>> {
    let map: BTreeMap<_, _> = rows.iter().copied().collect();
    if rows.len() != 5 || !map.keys().copied().eq(1..=5) {
        return None;
    }
    Some(map)
}

fn joined(rows: &[Readout]) -> Option<(u32, u32, bool)> {
    let observed = keyed(rows)?;
    let reference = keyed(&REFERENCE)?;
    let persistence_loss = observed.iter().map(|(&i, &v)| u32::from(i.abs_diff(v))).sum();
    let fixed_loss = observed.values().map(|&v| u32::from(3u8.abs_diff(v))).sum();
    // The committed full tuple relation is assumed trusted; equality only
    // checks identity against it. It cannot authenticate a self-consistent
    // reference created after the alleged misjoin.
    Some((persistence_loss, fixed_loss, observed == reference))
}

fn visit_permutations(values: &mut [u8; 5], i: usize, counts: &mut [u32; 13]) {
    if i == values.len() {
        let rows: Vec<_> = (1..=5).zip(values.iter().copied()).collect();
        let (persistence, fixed, _) = joined(&rows).unwrap();
        assert_eq!(fixed, 6);
        counts[persistence as usize] += 1;
        return;
    }
    for j in i..values.len() {
        values.swap(i, j);
        visit_permutations(values, i + 1, counts);
        values.swap(i, j);
    }
}

fn main() {
    let reversed_payload = [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)];
    for (label, data) in [("trusted relation", REFERENCE), ("misjoined values", reversed_payload)] {
        let (persistence, fixed, same) = joined(&data).unwrap();
        println!("{label}: roster valid, total absolute errors {persistence} vs {fixed}, reference match={same}");
    }
    let mut counts = [0; 13];
    visit_permutations(&mut [1, 2, 3, 4, 5], 0, &mut counts);
    println!("120 invented permutations: {} lower, {} tied, {} higher persistence loss",
             counts[..6].iter().sum::<u32>(), counts[6], counts[7..].iter().sum::<u32>());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_is_not_payload_reassignment() {
        let mut reordered = REFERENCE;
        reordered.reverse();
        assert_eq!(joined(&reordered), Some((0, 6, true)));
        assert_eq!(joined(&[(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]), Some((12, 6, false)));
        assert_eq!(joined(&[(1, 5), (1, 4), (3, 3), (4, 2), (5, 1)]), None);
    }

    #[test]
    fn exact_permutation_distribution() {
        let mut counts = [0; 13];
        visit_permutations(&mut [1, 2, 3, 4, 5], 0, &mut counts);
        assert_eq!(counts, [1, 0, 4, 0, 12, 0, 24, 0, 35, 0, 24, 0, 20]);
    }
}
