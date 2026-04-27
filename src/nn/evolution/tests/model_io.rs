use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::gene::*;
use crate::nn::evolution::mutation::{AddConnectionMutation, Mutation, SizeConstraints};
use crate::nn::evolution::node_ops::{find_removable_skip_connections, repair_param_input_dims};

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

#[test]
fn test_nodelevel_genome_serde_roundtrip_minimal() {
    let genome = NetworkGenome::minimal(4, 3);
    let json = serde_json::to_string(&genome).expect("序列化失败");
    let restored: NetworkGenome = serde_json::from_str(&json).expect("反序列化失败");

    assert!(restored.is_node_level());
    assert_eq!(restored.input_dim, 4);
    assert_eq!(restored.output_dim, 3);
    assert!(restored.layers().is_empty());
    assert!(restored.skip_edges().is_empty());
    assert_eq!(restored.nodes().len(), genome.nodes().len());
}

#[test]
fn test_nodelevel_genome_serde_roundtrip_with_weights() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let mut rng = rng();
    let build = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build).unwrap();

    let json = serde_json::to_string(&genome).expect("序列化失败");
    let restored: NetworkGenome = serde_json::from_str(&json).expect("反序列化失败");

    assert!(restored.is_node_level());
    assert!(restored.has_weight_snapshots());
    assert_eq!(
        restored.node_weight_snapshots().len(),
        genome.node_weight_snapshots().len()
    );
}

#[test]
fn test_nodelevel_genome_serde_roundtrip_with_connection() {
    let mut genome = NetworkGenome::minimal(4, 2);
    let mut rng = rng();
    crate::nn::evolution::mutation::InsertLayerMutation::default()
        .apply(&mut genome, &SizeConstraints::default(), &mut rng)
        .unwrap();
    repair_param_input_dims(&mut genome);
    AddConnectionMutation
        .apply(&mut genome, &SizeConstraints::default(), &mut rng)
        .expect("应能添加 NodeLevel 跨层连接");

    let json = serde_json::to_string(&genome).expect("序列化失败");
    let restored: NetworkGenome = serde_json::from_str(&json).expect("反序列化失败");

    assert!(restored.is_node_level());
    assert!(!find_removable_skip_connections(&restored).is_empty());
    assert!(
        restored
            .nodes()
            .iter()
            .any(|n| matches!(n.node_type, NodeTypeDescriptor::Add))
    );
}

#[test]
fn test_layer_level_genome_is_not_a_valid_runtime_input() {
    let mut genome = NetworkGenome {
        input_dim: 2,
        output_dim: 1,
        seq_len: None,
        input_spatial: None,
        training_config: TrainingConfig::default(),
        generated_by: "legacy".to_string(),
        repr: GenomeRepr::LayerLevel {
            layers: Vec::new(),
            skip_edges: Vec::new(),
            next_innovation: 1,
            weight_snapshots: std::collections::HashMap::new(),
        },
    };

    assert!(genome.migrate_to_node_level().is_err());
}
