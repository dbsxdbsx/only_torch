//! 通用 observation / action schema 契约测试。

use super::super::action::ActionAdapter;
use super::super::component::Components;
use super::super::runner::my_zero_mcts_config;
use super::super::schema::{ActionSchema, ObservationSchema, PolicyLayout};
use crate::rl::mcts::ActionPayload;
use approx::assert_abs_diff_eq;

#[test]
fn observation_schema_dims_cover_rect_composite_and_tokens() {
    assert_eq!(
        ObservationSchema::ImageRect {
            channels: 3,
            height: 16,
            width: 24,
        }
        .dim(),
        3 * 16 * 24
    );
    assert_eq!(
        ObservationSchema::ImageDense {
            channels: 3,
            height: 16,
            width: 24,
            aux_dim: 5,
        }
        .dim(),
        3 * 16 * 24 + 5
    );
    assert_eq!(
        ObservationSchema::Tokens {
            length: 6,
            vocab_size: 32,
            embed_dim: 8,
            pad_id: 0,
        }
        .dim(),
        12
    );
}

#[test]
fn multidiscrete_joint_id_roundtrip_and_payload() {
    let schema = ActionSchema::MultiDiscrete {
        factors: vec![4, 4, 16],
    };
    assert_eq!(schema.action_count(), 256);
    assert_eq!(
        schema.policy_layout(),
        PolicyLayout::Factorized {
            factors: vec![4, 4, 16]
        }
    );
    let id = schema.encode_joint(&[1, 2, 3]);
    assert_eq!(id, (4 + 2) * 16 + 3);
    assert_eq!(schema.decode_joint(id), vec![1, 2, 3]);
    assert_eq!(
        schema.payload(id),
        ActionPayload::MultiDiscrete(vec![1, 2, 3])
    );
    assert_eq!(schema.to_env(id), vec![1.0, 2.0, 3.0]);
}

#[test]
fn continuous_2d_uses_cartesian_bin_centers() {
    let schema = ActionSchema::ContinuousBins {
        ranges: vec![(-1.0, 1.0), (-2.0, 2.0)],
        buckets: 3,
    };
    assert_eq!(schema.action_count(), 9);
    let center = schema.encode_joint(&[1, 1]);
    let ActionPayload::Continuous(center_values) = schema.payload(center) else {
        panic!("应生成连续载荷");
    };
    assert_abs_diff_eq!(center_values[0], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(center_values[1], 0.0, epsilon = 1e-6);
    let corner = schema.to_env(schema.encode_joint(&[0, 2]));
    assert_abs_diff_eq!(corner[0], -2.0 / 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(corner[1], 4.0 / 3.0, epsilon = 1e-6);
}

#[test]
fn platform_like_hybrid_has_stable_joint_catalog() {
    let schema = ActionSchema::HybridBins {
        discrete: vec![3],
        continuous: vec![(-1.0, 1.0); 3],
        buckets: 7,
    };
    assert_eq!(schema.action_count(), 3 * 7 * 7 * 7);
    let id = schema.encode_joint(&[2, 3, 3, 3]);
    let ActionPayload::Hybrid {
        discrete,
        continuous,
    } = schema.payload(id)
    else {
        panic!("应生成 Hybrid 载荷");
    };
    assert_eq!(discrete, 2);
    for value in continuous {
        assert_abs_diff_eq!(value, 0.0, epsilon = 1e-6);
    }
    let adapter = ActionAdapter::from_schema(schema);
    assert_eq!(adapter.action_dim(), 1029);
    let env_action = adapter.to_env(id);
    assert_eq!(env_action[0], 2.0);
    for value in &env_action[1..] {
        assert_abs_diff_eq!(*value, 0.0, epsilon = 1e-6);
    }
}

#[test]
fn factorized_policy_target_and_joint_priors_are_consistent() {
    let schema = ActionSchema::MultiDiscrete {
        factors: vec![2, 3],
    };
    let mut joint_target = vec![0.0; 6];
    joint_target[schema.encode_joint(&[1, 2])] = 1.0;
    assert_eq!(
        schema.encode_policy_target(&joint_target),
        vec![0.0, 1.0, 0.0, 0.0, 1.0]
    );

    let joint = schema.joint_priors(&[0.25, 0.75, 0.2, 0.3, 0.5]);
    assert_eq!(joint.len(), 6);
    assert_abs_diff_eq!(joint.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(
        joint[schema.encode_joint(&[1, 2])],
        0.75 * 0.5,
        epsilon = 1e-6
    );
}

#[test]
fn invalid_action_schemas_fail_before_model_construction() {
    assert!(
        ActionSchema::MultiDiscrete { factors: vec![] }
            .validate()
            .is_err()
    );
    assert!(
        ActionSchema::HybridBins {
            discrete: vec![2, 3],
            continuous: vec![(-1.0, 1.0)],
            buckets: 7,
        }
        .validate()
        .is_err()
    );
    assert!(
        ActionSchema::ContinuousBins {
            ranges: vec![(-1.0, 1.0)],
            buckets: 0,
        }
        .validate()
        .is_err()
    );
}

#[test]
fn large_unknown_catalog_auto_enables_sampled_without_touching_small_path() {
    let base = Components::base();
    let small = my_zero_mcts_config(20, 1.0, 0.99, &base, 127, 0.25);
    let large = my_zero_mcts_config(20, 1.0, 0.99, &base, 128, 0.25);
    assert!(small.sampled_k.is_none());
    assert!(large.sampled_k.is_some());

    let mut gumbel = base;
    gumbel.gumbel = true;
    let large_gumbel = my_zero_mcts_config(20, 1.0, 0.99, &gumbel, 128, 0.25);
    assert!(large_gumbel.sampled_k.is_none());
}
