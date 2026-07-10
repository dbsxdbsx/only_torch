//! `rosmo.rs` 单元测试：一步 look-ahead 改进分布、优势过滤 BC 权重、现算 bootstrap、
//! prepare 路径（Refreshed 变体 + RNG 消耗与 Borrowed 逐 bit 一致 + 不写回）。

use super::super::component::Components;
use super::super::rosmo::{
    one_step_improved_policy, one_step_improved_policy_with_actions, rosmo_refresh_window,
    validated_replay_action_id,
};
use crate::rl::mcts::{ActionPayload, Dynamics, DynamicsOutput};
use crate::rl::{GameOutcome, SelfPlayGame, SelfPlayStep};

/// 极简确定性 Dynamics：2 动作；动作 0 奖励 `r0`、动作 1 奖励 `r1`；
/// next state value 恒 `next_v`、root value 恒 `root_v`、prior 固定。
struct MockDyn {
    prior: Vec<f32>,
    root_v: f32,
    next_v: f32,
    r0: f32,
    r1: f32,
}

impl Dynamics for MockDyn {
    fn initial_state(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        (obs.to_vec(), self.prior.clone(), self.root_v)
    }
    fn recurrent(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput {
        let reward = match action {
            ActionPayload::Discrete(0) => self.r0,
            _ => self.r1,
        };
        DynamicsOutput {
            next_state: state.to_vec(),
            reward,
            prior: self.prior.clone(),
            value: self.next_v,
            terminal: false,
            continuation: 1.0,
        }
    }
}

fn make_step(obs: Vec<f32>, action: usize) -> SelfPlayStep {
    SelfPlayStep {
        obs: obs.into(),
        action: vec![action as f32],
        policy_target: vec![1.0, 0.0],
        player: 0,
        reward: 1.0,
        root_value: Some(99.0), // 故意塞离谱旧值：ROSMO 不应读它
        terminated: false,
        truncated: false,
        continuation: 1.0,
        extras: Default::default(),
    }
}

struct StructuredMock;

impl Dynamics for StructuredMock {
    fn initial_state(&self, _obs: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        (vec![0.0], vec![0.5, 0.5], 0.0)
    }

    fn recurrent(&self, state: &[f32], action: &ActionPayload) -> DynamicsOutput {
        let ActionPayload::MultiDiscrete(values) = action else {
            panic!("ROSMO 必须保留结构化 payload");
        };
        DynamicsOutput {
            next_state: state.to_vec(),
            reward: values.iter().sum::<usize>() as f32,
            prior: vec![0.5, 0.5],
            value: 0.0,
            terminal: false,
            continuation: 1.0,
        }
    }
}

#[test]
fn structured_rosmo_uses_real_payload_catalog() {
    let actions = vec![
        ActionPayload::MultiDiscrete(vec![0, 0]),
        ActionPayload::MultiDiscrete(vec![1, 2]),
    ];
    let (policy, _, advs) =
        one_step_improved_policy_with_actions(&StructuredMock, &[0.0], 0.99, &actions);
    assert!(advs[1] > advs[0]);
    assert!(policy[1] > policy[0]);
}

#[test]
#[should_panic(expected = "prior 宽度")]
fn rosmo_rejects_prior_catalog_mismatch() {
    let model = MockDyn {
        prior: vec![1.0],
        root_v: 0.0,
        next_v: 0.0,
        r0: 0.0,
        r1: 0.0,
    };
    let actions = vec![ActionPayload::Discrete(0), ActionPayload::Discrete(1)];
    let _ = one_step_improved_policy_with_actions(&model, &[0.0], 0.99, &actions);
}

#[test]
fn rosmo_rejects_malformed_replay_action_ids_before_cast() {
    for invalid in [
        vec![],
        vec![f32::NAN],
        vec![-1.0],
        vec![0.5],
        vec![0.0, 1.0],
        vec![2.0],
    ] {
        assert!(
            std::panic::catch_unwind(|| validated_replay_action_id(&invalid, 2)).is_err(),
            "非法 replay action 应被拒绝: {invalid:?}"
        );
    }
    assert_eq!(validated_replay_action_id(&[1.0], 2), 1);
}

// ---- one_step_improved_policy ----

/// 优势更高的动作应获得高于先验的概率（改进方向正确）
#[test]
fn one_step_policy_favors_positive_advantage_action() {
    let model = MockDyn {
        prior: vec![0.5, 0.5],
        root_v: 0.0,
        next_v: 0.0,
        r0: 1.0,  // adv(0) = 1.0
        r1: -1.0, // adv(1) = −1.0
    };
    let (p, root_v, advs) = one_step_improved_policy(&model, &[0.0; 4], 0.99, 2);
    assert_eq!(root_v, 0.0);
    assert!((advs[0] - 1.0).abs() < 1e-6 && (advs[1] + 1.0).abs() < 1e-6);
    assert!(
        p[0] > 0.5 && p[1] < 0.5,
        "正优势动作应被加码：p={p:?}（先验 0.5/0.5）"
    );
    assert!((p[0] + p[1] - 1.0).abs() < 1e-6, "应归一化：p={p:?}");
    // 精确值：p0 = 0.5·e^1 / (0.5·e^1 + 0.5·e^{−1})（exp 前减 max 不改比值）
    let expected = (1.0f32).exp() / ((1.0f32).exp() + (-1.0f32).exp());
    assert!((p[0] - expected).abs() < 1e-5, "p0={} vs {expected}", p[0]);
}

/// 两动作优势相同时应回到先验本身（改进算子不无中生有）
#[test]
fn one_step_policy_equal_advantage_keeps_prior() {
    let model = MockDyn {
        prior: vec![0.7, 0.3],
        root_v: 0.5,
        next_v: 1.0,
        r0: 0.2,
        r1: 0.2,
    };
    let (p, _, _) = one_step_improved_policy(&model, &[0.0; 4], 0.99, 2);
    assert!(
        (p[0] - 0.7).abs() < 1e-6 && (p[1] - 0.3).abs() < 1e-6,
        "等优势时 exp 项约去、应还原先验：p={p:?}"
    );
}

// ---- rosmo_refresh_window ----

/// 现算 value target 应使用网络 bootstrap（而非 buffer 里离谱的旧 root_value=99）
#[test]
fn refresh_window_uses_fresh_bootstrap_not_stale_root_value() {
    let model = MockDyn {
        prior: vec![0.5, 0.5],
        root_v: 0.5, // 现算 bootstrap 值
        next_v: 0.5,
        r0: 0.0,
        r1: 0.0,
    };
    // 3 步局，每步 reward=1.0；td=1 → z(0) = 1.0 + γ·v_fresh(s1)
    let steps = vec![
        make_step(vec![0.0; 4], 0),
        make_step(vec![1.0; 4], 0),
        make_step(vec![2.0; 4], 0),
    ];
    let gamma = 0.9;
    let obs_at = |pos: usize| -> Vec<f32> { steps[pos].obs.as_f32().to_vec() };
    let t = rosmo_refresh_window(&model, &steps, 0, 1, 1, gamma, 2, &obs_at);

    let expected_z0 = 1.0 + gamma * 0.5;
    assert!(
        (t.values[0] - expected_z0).abs() < 1e-6,
        "z0 应为现算 bootstrap（{expected_z0}），got {}（若 ≈{} 则误读了 stale root_value）",
        t.values[0],
        1.0 + gamma * 99.0
    );
    assert_eq!(t.policies.len(), 2, "槽位数 = actual_k+1");
    assert_eq!(t.bc_weights.len(), 1, "BC 权重数 = actual_k");
}

/// BC 权重 = 执行动作优势过滤：正优势 → 1，负优势 → 0
#[test]
fn refresh_window_bc_weights_filter_by_executed_action_advantage() {
    let model = MockDyn {
        prior: vec![0.5, 0.5],
        root_v: 0.0,
        next_v: 0.0,
        r0: 1.0,  // 动作 0 优势 > 0
        r1: -1.0, // 动作 1 优势 < 0
    };
    let steps = vec![
        make_step(vec![0.0; 4], 0), // 执行了好动作
        make_step(vec![1.0; 4], 1), // 执行了坏动作
        make_step(vec![2.0; 4], 0),
    ];
    let obs_at = |pos: usize| -> Vec<f32> { steps[pos].obs.as_f32().to_vec() };
    let t = rosmo_refresh_window(&model, &steps, 0, 2, 5, 0.99, 2, &obs_at);
    assert_eq!(t.bc_weights, vec![1.0, 0.0], "优势过滤应逐槽位判定");
}

/// 越界槽位（unroll 超出局长）：policy 回退 uniform、value=0、BC 权重 0（对齐 legacy padding）
#[test]
fn refresh_window_pads_out_of_range_slots() {
    let model = MockDyn {
        prior: vec![0.9, 0.1],
        root_v: 0.0,
        next_v: 0.0,
        r0: 1.0,
        r1: 1.0,
    };
    // terminated 局：actual_k 可超过剩余步数
    let mut steps = vec![make_step(vec![0.0; 4], 0), make_step(vec![1.0; 4], 0)];
    steps[1].terminated = true;
    steps[1].continuation = 0.0;
    let obs_at = |pos: usize| -> Vec<f32> { steps[pos].obs.as_f32().to_vec() };
    let t = rosmo_refresh_window(&model, &steps, 1, 3, 5, 0.99, 2, &obs_at);

    assert_eq!(t.policies.len(), 4);
    assert_eq!(t.policies[2], vec![0.5, 0.5], "越界槽位应为 uniform");
    assert_eq!(t.values[2], 0.0, "越界槽位 value 应为 0");
    assert_eq!(t.bc_weights[1], 0.0, "越界槽位 BC 权重应为 0");
}

// ---- prepare 路径（Refreshed 变体）----

/// ROSMO 开启时 prepare 应返回 Refreshed；采样 RNG 消耗与 Borrowed 逐 bit 一致；不写回 buffer
#[test]
fn prepare_rosmo_returns_refreshed_same_rng_no_writeback() {
    use super::super::action::ActionAdapter;
    use super::super::network::MyZeroModel;
    use super::super::runner::{PreparedBatch, prepare_train_batch};
    use crate::nn::Graph;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let graph = Graph::new_with_seed(7);
    let model = MyZeroModel::new(&graph, 4, 2, 32).unwrap();
    let adapter = ActionAdapter::discrete_for_test(2);

    let mut buf = crate::rl::ReplayBuffer::new(4);
    buf.push(SelfPlayGame {
        steps: vec![
            make_step(vec![0.0; 4], 0),
            make_step(vec![1.0; 4], 1),
            make_step(vec![2.0; 4], 0),
        ],
        outcome: GameOutcome::InProgress,
    });

    // Borrowed 参照：同 seed 的采样序列
    let mut rng_a = StdRng::seed_from_u64(11);
    let base = Components::default();
    let borrowed = prepare_train_batch(
        &buf, 2, 1, 5, false, &base, &model, &adapter, 0.99, 4, None, None, &mut rng_a,
    );
    let borrowed_items = match borrowed {
        PreparedBatch::Borrowed(v) => v,
        _ => panic!("base 应返回 Borrowed"),
    };

    // ROSMO：同 seed 应采出完全相同的 (idx, start)（模型前向不消耗 rng）
    let mut rng_b = StdRng::seed_from_u64(11);
    let c = Components {
        rosmo: true,
        ..Default::default()
    };
    let refreshed = prepare_train_batch(
        &buf, 2, 1, 5, false, &c, &model, &adapter, 0.99, 4, None, None, &mut rng_b,
    );
    match refreshed {
        PreparedBatch::Refreshed(items, targets) => {
            assert_eq!(items, borrowed_items, "采样序列应与 Borrowed 逐 bit 一致");
            assert_eq!(targets.len(), items.len());
            for (t, &(_, start)) in targets.iter().zip(&items) {
                assert!(!t.policies.is_empty());
                let p0 = &t.policies[0];
                assert!(
                    (p0.iter().sum::<f32>() - 1.0).abs() < 1e-5,
                    "现算 policy 应归一化（start={start}）：{p0:?}"
                );
            }
        }
        _ => panic!("ROSMO 应返回 Refreshed"),
    }

    // 不写回：buffer 里的 stale 标签原封不动
    let stored = buf.get_at(0).unwrap();
    assert_eq!(stored.steps[0].root_value, Some(99.0), "ROSMO 不应写回");
    assert_eq!(stored.steps[0].policy_target, vec![1.0, 0.0]);
}

/// ROSMO + BC 的完整训练一步可跑通（loss 有限、参数更新）
#[test]
fn rosmo_train_batch_end_to_end_finite() {
    use super::super::action::ActionAdapter;
    use super::super::network::MyZeroModel;
    use super::super::runner::{PreparedBatch, prepare_train_batch, train_batch};
    use crate::nn::{Adam, Graph};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let graph = Graph::new_with_seed(3);
    let model = MyZeroModel::new(&graph, 4, 2, 32).unwrap();
    let adapter = ActionAdapter::discrete_for_test(2);
    let mut opt = Adam::new(&graph, &model.parameters(), 1e-3);

    let mut buf = crate::rl::ReplayBuffer::new(4);
    buf.push(SelfPlayGame {
        steps: vec![
            make_step(vec![0.1; 4], 0),
            make_step(vec![0.2; 4], 1),
            make_step(vec![0.3; 4], 0),
            make_step(vec![0.4; 4], 1),
        ],
        outcome: GameOutcome::InProgress,
    });

    let c = Components {
        rosmo: true,
        rosmo_alpha: 0.2,
        ..Default::default()
    };

    let mut rng = StdRng::seed_from_u64(5);
    let batch = prepare_train_batch(
        &buf, 4, 2, 5, false, &c, &model, &adapter, 0.99, 4, None, None, &mut rng,
    );
    let (items, targets) = match batch {
        PreparedBatch::Refreshed(items, targets) => (items, targets),
        _ => panic!("ROSMO 应返回 Refreshed"),
    };
    let train_view: Vec<_> = items
        .iter()
        .map(|&(idx, start)| (buf.get_ref(idx).unwrap(), start))
        .collect();

    let before: Vec<Vec<f32>> = model
        .parameters()
        .iter()
        .map(|p| p.value().unwrap().unwrap().data_as_slice().to_vec())
        .collect();

    let loss = train_batch(
        &model,
        &mut opt,
        &train_view,
        Some(&targets),
        2,
        5,
        0.99,
        &c,
        None,
    )
    .unwrap();
    assert!(loss.is_finite(), "ROSMO 训练 loss 应有限，got {loss}");

    let _ = opt; // step 已在 train_batch 内完成
    let changed = model
        .parameters()
        .iter()
        .zip(&before)
        .filter(|(p, b)| {
            p.value()
                .unwrap()
                .unwrap()
                .data_as_slice()
                .iter()
                .zip(b.iter())
                .any(|(x, y)| (x - y).abs() > 1e-9)
        })
        .count();
    assert!(changed > 0, "训练一步后应有参数被更新");
}
