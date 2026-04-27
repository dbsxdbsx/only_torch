use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::evolution::gene::*;
use crate::nn::evolution::mutation::{GrowHiddenSizeMutation, Mutation, SizeConstraints};
use crate::nn::evolution::net2net::{
    counts_of, gather_along_axis, gather_along_axis_scaled, widening_mapping,
};
use crate::nn::evolution::node_ops::repair_param_input_dims;
use crate::tensor::Tensor;

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

#[test]
fn test_widening_mapping_keeps_old_indices_prefix() {
    let mut rng = rng();
    let mapping = widening_mapping(3, 6, &mut rng);

    assert_eq!(&mapping[..3], &[0, 1, 2]);
    assert_eq!(mapping.len(), 6);
    assert!(mapping.iter().all(|&idx| idx < 3));
}

#[test]
fn test_counts_of_mapping() {
    let counts = counts_of(&[0, 1, 2, 1, 0], 3);

    assert_eq!(counts, vec![2, 2, 1]);
}

#[test]
fn test_gather_along_axis_copies_columns() {
    let src = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let out = gather_along_axis(&src, 1, &[1, 0, 1]);

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.data_as_slice(), &[2.0, 1.0, 2.0, 4.0, 3.0, 4.0]);
}

#[test]
fn test_gather_along_axis_scaled_divides_duplicates() {
    let src = Tensor::new(&[2.0, 4.0], &[1, 2]);
    let out = gather_along_axis_scaled(&src, 1, &[0, 1, 1], &[1, 2]);

    assert_eq!(out.shape(), &[1, 3]);
    assert_eq!(out.data_as_slice(), &[2.0, 2.0, 2.0]);
}

#[test]
fn test_grow_hidden_uses_node_level_snapshots_without_layer_state() {
    let mut genome = NetworkGenome::minimal(4, 2);
    let mut rng = rng();
    crate::nn::evolution::mutation::InsertLayerMutation::default()
        .apply(&mut genome, &SizeConstraints::default(), &mut rng)
        .unwrap();
    repair_param_input_dims(&mut genome);

    let build1 = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build1).unwrap();
    let before = genome.node_weight_snapshots().clone();

    GrowHiddenSizeMutation
        .apply(&mut genome, &SizeConstraints::default(), &mut rng)
        .unwrap();
    repair_param_input_dims(&mut genome);

    let build2 = genome.build(&mut rng).unwrap();
    let report = genome.restore_weights(&build2).unwrap();

    assert!(genome.is_node_level());
    assert!(!before.is_empty());
    assert!(report.inherited + report.partially_inherited > 0);
}
