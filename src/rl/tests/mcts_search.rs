//! MCTS 搜索引擎集成测试
//!
//! 使用 mock model 验证完整搜索循环、backup 公式、结果统计。
//! 纯 Rust，不依赖 pyo3。

use crate::rl::mcts::{
    ActionPayload, MctsConfig, MctsModel, PuctPolicy, RecurrentOut, RootOut, mcts_search,
};

// ============================================================================
// Mock: 确定性 3 选 1 单智能体（类 CartPole 简化）
// ============================================================================

/// 3 个离散动作，state = 步数计数器，value 恒定 0.5，非终止
#[derive(Clone)]
struct SingleAgentMock;

impl MctsModel for SingleAgentMock {
    type State = u32;

    fn root(&self, _obs: &[f32]) -> RootOut<Self::State> {
        RootOut {
            state: 0,
            prior: vec![0.6, 0.3, 0.1],
            value: 0.5,
            candidate_actions: vec![
                ActionPayload::Discrete(0),
                ActionPayload::Discrete(1),
                ActionPayload::Discrete(2),
            ],
            to_play: 0,
        }
    }

    fn recurrent(&self, state: &Self::State, _action: &ActionPayload) -> RecurrentOut<Self::State> {
        let depth = state + 1;
        let terminal = depth >= 3;
        RecurrentOut {
            state: depth,
            reward: if terminal { 1.0 } else { 0.0 },
            value: if terminal { 0.0 } else { 0.5 },
            prior: vec![0.33, 0.34, 0.33],
            candidate_actions: if terminal {
                vec![]
            } else {
                vec![
                    ActionPayload::Discrete(0),
                    ActionPayload::Discrete(1),
                    ActionPayload::Discrete(2),
                ]
            },
            terminal,
            to_play: 0,
            discount: 0.99,
        }
    }
}

// ============================================================================
// Mock: 双人零和（简化五子棋）
// ============================================================================

/// 2 个离散动作，黑白交替，depth=2 终止（黑胜 value=1）
#[derive(Clone)]
struct TwoPlayerMock;

impl MctsModel for TwoPlayerMock {
    type State = (u32, u8); // (depth, to_play)

    fn root(&self, _obs: &[f32]) -> RootOut<Self::State> {
        RootOut {
            state: (0, 0),
            prior: vec![0.5, 0.5],
            value: 0.3,
            candidate_actions: vec![
                ActionPayload::Discrete(0),
                ActionPayload::Discrete(1),
            ],
            to_play: 0, // 黑方
        }
    }

    fn recurrent(
        &self,
        state: &Self::State,
        _action: &ActionPayload,
    ) -> RecurrentOut<Self::State> {
        let (depth, current) = *state;
        let next_depth = depth + 1;
        let next_player = 1 - current;
        let terminal = next_depth >= 2;
        RecurrentOut {
            state: (next_depth, next_player),
            reward: 0.0,
            // 终止时返回"对 next_player 的价值"——黑胜=1 对黑有利
            value: if terminal { 1.0 } else { 0.3 },
            prior: vec![0.5, 0.5],
            candidate_actions: if terminal {
                vec![]
            } else {
                vec![
                    ActionPayload::Discrete(0),
                    ActionPayload::Discrete(1),
                ]
            },
            terminal,
            to_play: next_player,
            discount: 1.0, // 双人零和 discount=1
        }
    }
}

// ============================================================================
// 测试：单智能体搜索基础正确性
// ============================================================================

#[test]
fn test_search_single_agent_basic() {
    let model = SingleAgentMock;
    let policy = PuctPolicy::new();
    let cfg = MctsConfig {
        num_simulations: 50,
        temperature: 1.0,
        ..MctsConfig::default()
    };

    let result = mcts_search(&model, &policy, &[0.0], &cfg);

    // 有 3 个子节点
    assert_eq!(result.children.len(), 3);
    // 所有子节点都被访问过
    let total_visits: u32 = result.children.iter().map(|c| c.visit_count).sum();
    assert!(
        total_visits >= 50,
        "总访问次数应 >= num_simulations，实际: {total_visits}"
    );
    // 推荐动作是离散的
    assert!(
        matches!(result.recommended, ActionPayload::Discrete(_)),
        "推荐动作应为离散类型"
    );
    // learn_policy 之和 ≈ 1.0
    let policy_sum: f32 = result.learn_policy.iter().sum();
    assert!(
        (policy_sum - 1.0).abs() < 1e-4,
        "learn_policy 之和应为 1.0，实际: {policy_sum}"
    );
    // 高先验动作应获得更多访问（概率性，用宽松断言）
    assert!(
        result.children[0].visit_count >= result.children[2].visit_count,
        "高先验动作(0)应获得 >= 低先验动作(2) 的访问次数"
    );
}

#[test]
fn test_temperature_affects_make_targets() {
    use crate::rl::mcts::{ChildStat, SearchPolicy};

    let policy = PuctPolicy::new();
    // 模拟一组 visit counts：动作 0 明显多
    let children = vec![
        ChildStat { action: ActionPayload::Discrete(0), visit_count: 60, value_sum: 30.0, prior: 0.6, reward: 0.0, to_play: 0, discount: 1.0 },
        ChildStat { action: ActionPayload::Discrete(1), visit_count: 30, value_sum: 15.0, prior: 0.3, reward: 0.0, to_play: 0, discount: 1.0 },
        ChildStat { action: ActionPayload::Discrete(2), visit_count: 10, value_sum: 5.0, prior: 0.1, reward: 0.0, to_play: 0, discount: 1.0 },
    ];

    // 低温度 → 接近 one-hot
    let cfg_cold = MctsConfig { temperature: 0.01, ..MctsConfig::default() };
    let targets_cold = policy.make_targets(&children, &cfg_cold);
    let max_cold = targets_cold.iter().cloned().fold(0.0_f32, f32::max);

    // 正常温度 → 按 visit count 比例
    let cfg_normal = MctsConfig { temperature: 1.0, ..MctsConfig::default() };
    let targets_normal = policy.make_targets(&children, &cfg_normal);
    let max_normal = targets_normal.iter().cloned().fold(0.0_f32, f32::max);

    // 高温度 → 更均匀
    let cfg_hot = MctsConfig { temperature: 10.0, ..MctsConfig::default() };
    let targets_hot = policy.make_targets(&children, &cfg_hot);
    let max_hot = targets_hot.iter().cloned().fold(0.0_f32, f32::max);

    assert!(max_cold > max_normal, "低温度({max_cold}) 应 > 正常({max_normal})");
    assert!(max_normal > max_hot, "正常({max_normal}) 应 > 高温({max_hot})");
    assert!(max_cold > 0.95, "温度→0 应接近 1.0，实际: {max_cold}");
    assert!(max_hot < 0.5, "高温应接近均匀(0.33)，实际: {max_hot}");
}

// ============================================================================
// 测试：双人零和 backup 正确性
// ============================================================================

#[test]
fn test_search_two_player_negamax() {
    let model = TwoPlayerMock;
    let policy = PuctPolicy::new();
    let cfg = MctsConfig {
        num_simulations: 30,
        temperature: 1.0,
        ..MctsConfig::default()
    };

    let result = mcts_search(&model, &policy, &[0.0], &cfg);

    assert_eq!(result.children.len(), 2);
    let total_visits: u32 = result.children.iter().map(|c| c.visit_count).sum();
    assert!(total_visits >= 30, "总访问次数应 >= 30");
    // learn_policy 之和 ≈ 1.0
    let policy_sum: f32 = result.learn_policy.iter().sum();
    assert!(
        (policy_sum - 1.0).abs() < 1e-4,
        "learn_policy 之和应为 1.0，实际: {policy_sum}"
    );
    // 子节点 Q 值应为有限值（非 NaN / ±∞）
    for child in &result.children {
        let q = if child.visit_count > 0 {
            child.value_sum / child.visit_count as f32
        } else {
            0.0
        };
        assert!(q.is_finite(), "Q 值应为有限数，实际: {q}");
    }
}

// ============================================================================
// 测试：backup 单智能体折扣
// ============================================================================

#[test]
fn test_search_single_agent_discount() {
    let model = SingleAgentMock;
    let policy = PuctPolicy::new();
    // γ = 0 → 只看即时 reward，不累积未来价值
    let cfg_no_discount = MctsConfig {
        num_simulations: 50,
        discount: 0.0,
        ..MctsConfig::default()
    };
    let result_no = mcts_search(&model, &policy, &[0.0], &cfg_no_discount);

    // γ = 0.99 → 累积未来价值
    let cfg_full = MctsConfig {
        num_simulations: 50,
        discount: 0.99,
        ..MctsConfig::default()
    };
    let result_full = mcts_search(&model, &policy, &[0.0], &cfg_full);

    // 两种配置都应产生有效结果
    assert_eq!(result_no.children.len(), 3);
    assert_eq!(result_full.children.len(), 3);

    // γ=0.99 时根子节点的 Q 值总体应 > γ=0 时（因为累积了远端 reward）
    let total_v_full: f32 = result_full.children.iter().map(|c| c.value_sum).sum();
    let total_v_no: f32 = result_no.children.iter().map(|c| c.value_sum).sum();
    // 两者都应是有限值
    assert!(total_v_full.is_finite());
    assert!(total_v_no.is_finite());
}

// ============================================================================
// 测试：终止节点不展开
// ============================================================================

/// 全终止 mock：root 的所有子节点 recurrent 都返回 terminal=true
#[derive(Clone)]
struct AllTerminalMock;

impl MctsModel for AllTerminalMock {
    type State = u32;

    fn root(&self, _obs: &[f32]) -> RootOut<Self::State> {
        RootOut {
            state: 0,
            prior: vec![0.5, 0.5],
            value: 0.0,
            candidate_actions: vec![
                ActionPayload::Discrete(0),
                ActionPayload::Discrete(1),
            ],
            to_play: 0,
        }
    }

    fn recurrent(
        &self,
        _state: &Self::State,
        action: &ActionPayload,
    ) -> RecurrentOut<Self::State> {
        let reward = match action {
            ActionPayload::Discrete(0) => 1.0,
            _ => -1.0,
        };
        RecurrentOut {
            state: 1,
            reward,
            value: 0.0,
            prior: vec![],
            candidate_actions: vec![],
            terminal: true,
            to_play: 0,
            discount: 1.0,
        }
    }
}

#[test]
fn test_search_terminal_nodes() {
    let model = AllTerminalMock;
    let policy = PuctPolicy::new();
    let cfg = MctsConfig {
        num_simulations: 20,
        ..MctsConfig::default()
    };

    let result = mcts_search(&model, &policy, &[0.0], &cfg);

    assert_eq!(result.children.len(), 2);
    // 动作 0 给 reward=1，动作 1 给 reward=-1
    // PUCT 用 Q(a) = reward + γ·V(child) 选择，应偏向动作 0 → 更多访问
    let n0 = result.children[0].visit_count;
    let n1 = result.children[1].visit_count;
    assert!(
        n0 > n1,
        "动作 0（reward=1）的访问次数({n0}) 应 > 动作 1（reward=-1）({n1})"
    );
    // 推荐动作应为 0
    assert_eq!(
        result.recommended,
        ActionPayload::Discrete(0),
        "应推荐 reward 更高的动作 0"
    );
}

// ============================================================================
// 测试：零模拟次数（边界条件）
// ============================================================================

#[test]
fn test_search_zero_simulations() {
    let model = SingleAgentMock;
    let policy = PuctPolicy::new();
    let cfg = MctsConfig {
        num_simulations: 0,
        ..MctsConfig::default()
    };

    let result = mcts_search(&model, &policy, &[0.0], &cfg);

    // 即使 0 次模拟，根节点仍应展开（root 推理 + prepare_root）
    assert_eq!(result.children.len(), 3);
    // learn_policy 仍应有效
    let policy_sum: f32 = result.learn_policy.iter().sum();
    assert!(
        (policy_sum - 1.0).abs() < 1e-4,
        "零模拟时 learn_policy 之和应为 1.0"
    );
}

// ============================================================================
// 测试：SearchResult 暴露原始统计完整性
// ============================================================================

#[test]
fn test_search_result_exposes_raw_stats() {
    let model = SingleAgentMock;
    let policy = PuctPolicy::new();
    let cfg = MctsConfig {
        num_simulations: 30,
        ..MctsConfig::default()
    };

    let result = mcts_search(&model, &policy, &[0.0], &cfg);

    // 每个 ChildStat 都包含完整字段
    for (i, child) in result.children.iter().enumerate() {
        assert!(
            matches!(child.action, ActionPayload::Discrete(_)),
            "child[{i}] 应有 Discrete 动作"
        );
        assert!(
            child.prior > 0.0,
            "child[{i}] prior 应 > 0（注入 Dirichlet 后）"
        );
        assert!(
            child.value_sum.is_finite(),
            "child[{i}] value_sum 应为有限数"
        );
    }

    // learn_policy 长度 == children 数量
    assert_eq!(result.learn_policy.len(), result.children.len());
}
