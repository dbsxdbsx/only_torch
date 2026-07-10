//! Dynamics trait + DynamicsModel 适配器测试
//!
//! 用 mock 验证 Dynamics → MctsModel 桥接正确性和 mcts_search 兼容性。

use crate::rl::algo::my_zero::ActionSchema;
use crate::rl::mcts::{
    ActionId, ActionPayload, Dynamics, DynamicsModel, DynamicsOutput, LatentState, MctsConfig,
    MctsModel, PuctPolicy, RecurrentOut, RootOut, WorldModel, WorldModelOutput, mcts_search,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

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
            continuation: 0.5,
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
    let out: RootOut<LatentState> = model.root(&[0.0; 4]);

    assert_eq!(out.state.as_ref(), &[1.0, 2.0, 3.0]);
    assert_eq!(out.candidates.policy_priors(), vec![0.4, 0.3, 0.3]);
    assert!((out.value - 0.5).abs() < 1e-6);
    assert_eq!(out.candidates.len(), 3);
    assert_eq!(out.to_play, 0);
}

// ============================================================================
// recurrent() 桥接
// ============================================================================

#[test]
fn test_dynamics_model_recurrent() {
    let model = make_model();
    let state = LatentState::new(vec![1.0, 2.0, 3.0]);
    let out: RecurrentOut<LatentState> =
        model.recurrent(&state, ActionId(0), &ActionPayload::Discrete(0));

    assert_eq!(out.state.as_ref(), &[2.0, 3.0, 4.0]);
    assert!((out.reward - 0.1).abs() < 1e-6);
    assert!((out.value - 0.4).abs() < 1e-6);
    assert_eq!(out.candidates.policy_priors(), vec![0.33, 0.34, 0.33]);
    assert_eq!(out.candidates.len(), 3);
    assert!(!out.terminal);
    assert_eq!(out.to_play, 0);
    // 非终止边用二值 gate → 恒为 γ（不再 γ·continuation），与 n-step target 口径一致。
    assert!((out.discount - 0.99).abs() < 1e-6);
}

// 终止边：discount 应为 0（γ·(1−done)），且不再产出候选。
struct TerminalDynamics;

impl Dynamics for TerminalDynamics {
    fn initial_state(&self, _obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        (vec![0.0], vec![1.0], 0.0)
    }

    fn recurrent(&self, _state: &[f32], _action: &ActionPayload) -> DynamicsOutput {
        DynamicsOutput {
            next_state: vec![0.0],
            reward: 1.0,
            prior: vec![1.0],
            value: 5.0,
            terminal: true,
            continuation: 0.0,
        }
    }
}

#[test]
fn test_terminal_edge_zeroes_discount() {
    let model = DynamicsModel::new(TerminalDynamics, vec![ActionPayload::Discrete(0)], 0.99);
    let state = LatentState::new(vec![0.0]);
    let out: RecurrentOut<LatentState> =
        model.recurrent(&state, ActionId(0), &ActionPayload::Discrete(0));
    assert!(out.terminal, "终止边 terminal 应为 true");
    assert!(
        out.discount.abs() < 1e-6,
        "终止边 discount 应为 0，实际 {}",
        out.discount
    );
    assert!(out.candidates.is_empty(), "终止边不应再产出候选");
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

    let mut rng = StdRng::seed_from_u64(42);
    let result = mcts_search(&model, &policy, &[0.0; 4], &cfg, &mut rng);

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

struct StructuredDynamics;

impl Dynamics for StructuredDynamics {
    fn initial_state(&self, _obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        (vec![0.0], vec![0.05, 0.1, 0.15, 0.2, 0.2, 0.3], 0.0)
    }

    fn recurrent(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput {
        let ActionPayload::MultiDiscrete(values) = action else {
            panic!("应为 MultiDiscrete payload");
        };
        self.recurrent_with_id(state, ActionId(values[0] * 3 + values[1]), action)
    }

    fn recurrent_with_id(
        &self,
        state: &[f32],
        action_id: ActionId,
        action: &ActionPayload,
    ) -> DynamicsOutput {
        assert_eq!(
            action,
            &ActionPayload::MultiDiscrete(vec![action_id.index() / 3, action_id.index() % 3])
        );
        DynamicsOutput {
            next_state: vec![state[0] + 1.0],
            reward: action_id.index() as f32,
            prior: vec![0.3, 0.2, 0.2, 0.15, 0.1, 0.05],
            value: 0.0,
            terminal: false,
            continuation: 1.0,
        }
    }
}

#[test]
fn structured_payload_uses_action_id_without_discrete_fallback() {
    let schema = ActionSchema::MultiDiscrete {
        factors: vec![2, 3],
    };
    let actions = (0..schema.action_count())
        .map(|id| schema.payload(id))
        .collect();
    let model = DynamicsModel::new(StructuredDynamics, actions, 0.99);
    let root = model.root(&[0.0]);
    let id = ActionId(schema.encode_joint(&[1, 2]));
    let payload = schema.payload(id.index());
    let out = model.recurrent(&root.state, id, &payload);
    assert_eq!(out.reward, 5.0);
    assert_eq!(
        out.candidates.policy_priors(),
        vec![0.3, 0.2, 0.2, 0.15, 0.1, 0.05]
    );
}

#[derive(Clone, Debug, PartialEq)]
struct RecurrentPlannerState {
    latent: Vec<f32>,
    hidden: f32,
}

struct RecurrentWorldModel;

impl WorldModel for RecurrentWorldModel {
    type State = RecurrentPlannerState;

    fn initial_state(&self, _obs: &[f32]) -> (Self::State, Vec<f32>, f32) {
        (
            RecurrentPlannerState {
                latent: vec![0.0],
                hidden: 1.0,
            },
            vec![1.0],
            0.0,
        )
    }

    fn recurrent(
        &self,
        state: &Self::State,
        _action_id: ActionId,
        _action: &ActionPayload,
    ) -> WorldModelOutput<Self::State> {
        WorldModelOutput {
            next_state: RecurrentPlannerState {
                latent: vec![state.latent[0] + 1.0],
                hidden: state.hidden + 2.0,
            },
            reward: 0.0,
            prior: vec![1.0],
            value: 0.0,
            terminal: false,
            continuation: 1.0,
        }
    }
}

#[test]
fn world_model_state_can_carry_recurrent_hidden() {
    let model = DynamicsModel::new(RecurrentWorldModel, vec![ActionPayload::Discrete(0)], 0.99);
    let root = model.root(&[0.0]);
    let next = model.recurrent(&root.state, ActionId(0), &ActionPayload::Discrete(0));
    assert_eq!(next.state.latent, vec![1.0]);
    assert_eq!(next.state.hidden, 3.0);
}
