//! State 携带 recurrent hidden 契约测试：
//! `MctsModel::State` 携带 recurrent hidden + `recurrent` 返回 value prefix 增量 reward。
//!
//! 目的：证明 EfficientZero **value prefix 忠实版**（LSTM hidden 穿过搜索树、reward 取 prefix
//! 增量）**无需改动 MCTS 内核**——现有泛型 `State` + backup 已足以承载。

use crate::rl::mcts::{
    ActionPayload, MctsConfig, MctsModel, PuctPolicy, RecurrentOut, RootOut, mcts_search,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// 携带 LSTM-like hidden + 累计 value prefix 的隐状态。
#[derive(Clone)]
struct RecurrentState {
    depth: u32,
    /// 玩具 LSTM hidden：每步累加，用于验证 hidden 确实"穿过搜索树"。
    hidden: f32,
    /// 累计 value prefix。
    prefix: f32,
}

/// 每步即时 reward 恒为 `STEP_REWARD`；`recurrent` 返回的 reward = prefix 增量（恰等于 STEP_REWARD）。
const STEP_REWARD: f32 = 1.0;

/// value prefix 忠实版 mock：State 携带 hidden + prefix，reward 取增量。
#[derive(Clone)]
struct ValuePrefixMock;

impl MctsModel for ValuePrefixMock {
    type State = RecurrentState;

    fn root(&self, _obs: &[f32]) -> RootOut<Self::State> {
        RootOut {
            state: RecurrentState {
                depth: 0,
                hidden: 0.0,
                prefix: 0.0,
            },
            prior: vec![0.5, 0.5],
            value: 0.0,
            candidate_actions: vec![ActionPayload::Discrete(0), ActionPayload::Discrete(1)],
            to_play: 0,
        }
    }

    fn recurrent(&self, state: &Self::State, _action: &ActionPayload) -> RecurrentOut<Self::State> {
        let depth = state.depth + 1;
        // LSTM hidden 累加（玩具）：体现 hidden 随 unroll 穿过搜索树
        let hidden = state.hidden + 1.0;
        // 新累计 prefix = 父 prefix + 本步 reward
        let new_prefix = state.prefix + STEP_REWARD;
        // 关键：返回的 reward 取 value prefix 增量（忠实 EZ 口径）
        let reward = new_prefix - state.prefix;
        let terminal = depth >= 3;
        RecurrentOut {
            state: RecurrentState {
                depth,
                hidden,
                prefix: new_prefix,
            },
            reward,
            value: if terminal { 0.0 } else { 0.5 },
            prior: vec![0.5, 0.5],
            candidate_actions: if terminal {
                vec![]
            } else {
                vec![ActionPayload::Discrete(0), ActionPayload::Discrete(1)]
            },
            terminal,
            to_play: 0,
            discount: 0.99,
        }
    }
}

/// 单步对照：State = u32（无 hidden），recurrent 直接返回单步 reward = STEP_REWARD。
#[derive(Clone)]
struct SingleStepMock;

impl MctsModel for SingleStepMock {
    type State = u32;

    fn root(&self, _obs: &[f32]) -> RootOut<Self::State> {
        RootOut {
            state: 0,
            prior: vec![0.5, 0.5],
            value: 0.0,
            candidate_actions: vec![ActionPayload::Discrete(0), ActionPayload::Discrete(1)],
            to_play: 0,
        }
    }

    fn recurrent(&self, state: &Self::State, _action: &ActionPayload) -> RecurrentOut<Self::State> {
        let depth = state + 1;
        let terminal = depth >= 3;
        RecurrentOut {
            state: depth,
            reward: STEP_REWARD,
            value: if terminal { 0.0 } else { 0.5 },
            prior: vec![0.5, 0.5],
            candidate_actions: if terminal {
                vec![]
            } else {
                vec![ActionPayload::Discrete(0), ActionPayload::Discrete(1)]
            },
            terminal,
            to_play: 0,
            discount: 0.99,
        }
    }
}

/// 直接验证 hidden 穿树 + reward = prefix 增量（不经搜索）。
#[test]
fn test_recurrent_state_threads_hidden_and_prefix_delta() {
    let model = ValuePrefixMock;
    let root = model.root(&[0.0]);
    assert_eq!(root.state.hidden, 0.0);
    assert_eq!(root.state.prefix, 0.0);

    let s1 = model.recurrent(&root.state, &ActionPayload::Discrete(0));
    assert_eq!(s1.state.hidden, 1.0, "hidden 应随 unroll 累加");
    assert_eq!(s1.state.prefix, STEP_REWARD, "prefix 应累计");
    assert!(
        (s1.reward - STEP_REWARD).abs() < 1e-6,
        "reward 应为 prefix 增量"
    );

    let s2 = model.recurrent(&s1.state, &ActionPayload::Discrete(1));
    assert_eq!(s2.state.hidden, 2.0, "hidden 继续累加 = 穿过搜索树");
    assert_eq!(s2.state.prefix, 2.0 * STEP_REWARD);
    assert!((s2.reward - STEP_REWARD).abs() < 1e-6);
}

/// 契约核心：prefix-delta reward 模型 ≡ 单步 reward 模型（backup 等价），
/// 证明 value prefix 忠实版无需改内核（State 类型差异不影响搜索统计）。
#[test]
fn test_prefix_delta_equivalent_to_single_step_in_search() {
    let cfg = MctsConfig {
        num_simulations: 30,
        temperature: 1.0,
        ..MctsConfig::default()
    };

    let mut rng_a = StdRng::seed_from_u64(2024);
    let res_prefix = mcts_search(
        &ValuePrefixMock,
        &PuctPolicy::new(),
        &[0.0],
        &cfg,
        &mut rng_a,
    );

    let mut rng_b = StdRng::seed_from_u64(2024);
    let res_single = mcts_search(
        &SingleStepMock,
        &PuctPolicy::new(),
        &[0.0],
        &cfg,
        &mut rng_b,
    );

    // 两模型 reward 序列相同（prefix 增量 == 单步），同 seed 下 backup 统计应逐位一致
    let v_prefix: Vec<u32> = res_prefix.children.iter().map(|c| c.visit_count).collect();
    let v_single: Vec<u32> = res_single.children.iter().map(|c| c.visit_count).collect();
    assert_eq!(
        v_prefix, v_single,
        "prefix-delta 与单步 reward 的 backup 应等价"
    );
    assert_eq!(
        res_prefix.learn_policy, res_single.learn_policy,
        "learn_policy 也应等价"
    );

    for c in &res_prefix.children {
        let q = if c.visit_count > 0 {
            c.value_sum / c.visit_count as f32
        } else {
            0.0
        };
        assert!(q.is_finite(), "Q 应为有限值");
    }
}
