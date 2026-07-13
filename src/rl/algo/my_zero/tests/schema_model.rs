//! 通用 observation/action schema 与 MyZeroModel 的端到端纵切。

use super::super::network::{MyZeroModel, UnrollItem};
use super::super::schema::{ActionSchema, ObservationSchema};
use crate::nn::{Adam, Graph, Optimizer};
use crate::rl::mcts::{ActionId, Dynamics};
use approx::assert_abs_diff_eq;

#[test]
fn factorized_multidiscrete_model_infers_and_trains() {
    let graph = Graph::new_with_seed(42);
    let action_schema = ActionSchema::MultiDiscrete {
        factors: vec![2, 3],
    };
    let model = MyZeroModel::new_with_schemas(
        &graph,
        ObservationSchema::Flat(4),
        action_schema.clone(),
        16,
    )
    .unwrap();
    assert_eq!(model.action_dim, 6);
    assert_eq!(model.policy_dim, 5);

    let obs = vec![0.1, -0.2, 0.3, 0.4];
    let (latent, prior, value) = Dynamics::initial_state(&&model, &obs);
    assert_eq!(latent.len(), 16);
    assert_eq!(prior.len(), 6);
    assert!(value.is_finite());
    assert_abs_diff_eq!(prior.iter().sum::<f32>(), 1.0, epsilon = 1e-5);

    let action_id = action_schema.encode_joint(&[1, 2]);
    let payload = action_schema.payload(action_id);
    let out = Dynamics::recurrent_with_id(&&model, &latent, ActionId(action_id), &payload);
    assert_eq!(out.prior.len(), 6);
    assert!(out.reward.is_finite() && out.value.is_finite());

    let mut opt = Adam::new(&graph, &model.parameters(), 0.01);
    let mut target = vec![0.0; 6];
    target[action_id] = 1.0;
    let item = UnrollItem {
        obs_t: obs.into(),
        actions: vec![action_id],
        target_policies: vec![target.clone(), target],
        target_values: vec![0.5, 0.25],
        target_rewards: vec![0.1],
        target_continuations: vec![1.0],
        next_obs: Vec::new(),
        bc_weights: Vec::new(),
        burn_in_obs: Vec::new(),
        burn_in_actions: Vec::new(),
        burn_in_leading_action: None,
        train_prev_action: None,
    };
    let loss = model
        .train_unroll_batch(&[item], 0.0, 0.0, 1.0, false, 0.0)
        .unwrap();
    opt.zero_grad().unwrap();
    let value = loss.backward().unwrap();
    assert!(value.is_finite());
    opt.step().unwrap();
}

#[test]
fn token_model_uses_dynamic_token_input() {
    let graph = Graph::new_with_seed(7);
    let model = MyZeroModel::new_with_schemas(
        &graph,
        ObservationSchema::Tokens {
            length: 6,
            vocab_size: 32,
            embed_dim: 8,
            pad_id: 0,
        },
        ActionSchema::discrete(2),
        16,
    )
    .unwrap();

    let (latent_a, prior_a, _) = Dynamics::initial_state(
        &&model,
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let (latent_b, prior_b, _) = Dynamics::initial_state(
        &&model,
        &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    assert_eq!(prior_a.len(), 2);
    assert_eq!(prior_b.len(), 2);
    assert_ne!(latent_a, latent_b, "set_value 后 token 查表不能复用旧索引");
}

#[test]
fn rectangular_image_dense_model_runs_root_inference() {
    let graph = Graph::new_with_seed(9);
    let schema = ObservationSchema::ImageDense {
        channels: 3,
        height: 16,
        width: 24,
        aux_dim: 5,
    };
    let model =
        MyZeroModel::new_with_schemas(&graph, schema, ActionSchema::discrete(3), 16).unwrap();
    let obs: Vec<f32> = (0..schema.dim()).map(|i| (i % 31) as f32 / 31.0).collect();
    let (latent, prior, value) = Dynamics::initial_state(&&model, &obs);
    assert_eq!(latent.len(), 16);
    assert_eq!(prior.len(), 3);
    assert!(value.is_finite());
}
