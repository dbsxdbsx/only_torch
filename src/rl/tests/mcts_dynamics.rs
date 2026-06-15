//! Dynamics trait + DynamicsModel 适配器测试
//!
//! 用 mock 验证 Dynamics → MctsModel 桥接正确性和 mcts_search 兼容性。

use crate::rl::mcts::{
    ActionPayload, Dynamics, DynamicsModel, DynamicsOutput, MctsConfig, MctsModel, PuctPolicy,
    RecurrentOut, RootOut, mcts_search,
};

// ============================================================================
// Mock: 固定输出的 Dynamics 实现
// ============================================================================

struct DummyDynamics;

impl Dynamics for DummyDynamics {
    fn initial_state(&self, _obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        let latent = vec![1.0, 2.0, 3.0];
        let prior = vec![0.4, 0.3, 0.3];
        let value = 0.5;
        (latent, prior, value)
    }

    fn recurrent(&self, state: &[f32], _action: &ActionPayload) -> DynamicsOutput {
        DynamicsOutput {
            next_state: state.iter().map(|x| x + 1.0).collect(),
            reward: 0.1,
            prior: vec![0.33, 0.34, 0.33],
            value: 0.4,
            terminal: false,
        }
    }
}

fn make_model() -> DynamicsModel<DummyDynamics> {
    let actions = vec![
        ActionPayload::Discrete(0),
        ActionPayload::Discrete(1),
        ActionPayload::Discrete(2),
    ];
    DynamicsModel::new(DummyDynamics, actions, 0.99)
}

// ============================================================================
// root() 桥接
// ============================================================================

#[test]
fn test_dynamics_model_root() {
    let model = make_model();
    let out: RootOut<Vec<f32>> = model.root(&[0.0; 4]);

    assert_eq!(out.state, vec![1.0, 2.0, 3.0]);
    assert_eq!(out.prior, vec![0.4, 0.3, 0.3]);
    assert!((out.value - 0.5).abs() < 1e-6);
    assert_eq!(out.candidate_actions.len(), 3);
    assert_eq!(out.to_play, 0);
}

// ============================================================================
// recurrent() 桥接
// ============================================================================

#[test]
fn test_dynamics_model_recurrent() {
    let model = make_model();
    let state = vec![1.0, 2.0, 3.0];
    let out: RecurrentOut<Vec<f32>> = model.recurrent(&state, &ActionPayload::Discrete(0));

    assert_eq!(out.state, vec![2.0, 3.0, 4.0]);
    assert!((out.reward - 0.1).abs() < 1e-6);
    assert!((out.value - 0.4).abs() < 1e-6);
    assert_eq!(out.prior, vec![0.33, 0.34, 0.33]);
    assert_eq!(out.candidate_actions.len(), 3);
    assert!(!out.terminal);
    assert_eq!(out.to_play, 0);
    assert!((out.discount - 0.99).abs() < 1e-6);
}

// ============================================================================
// mcts_search 兼容性
// ============================================================================

#[test]
fn test_dynamics_model_mcts_search() {
    let model = make_model();
    let policy = PuctPolicy::new();
    let cfg = MctsConfig {
        num_simulations: 10,
        ..MctsConfig::default()
    };

    let result = mcts_search(&model, &policy, &[0.0; 4], &cfg);

    assert_eq!(result.children.len(), 3, "应有 3 个根子节点");
    let total_visits: u32 = result.children.iter().map(|c| c.visit_count).sum();
    assert!(total_visits >= 10, "总访问次数应 >= num_simulations");
    assert!(
        matches!(result.recommended, ActionPayload::Discrete(_)),
        "推荐动作应为离散类型"
    );
    let policy_sum: f32 = result.learn_policy.iter().sum();
    assert!(
        (policy_sum - 1.0).abs() < 1e-4,
        "learn_policy 之和应为 1.0，实际: {policy_sum}"
    );
}
