//! PUCT 策略测试：Dirichlet 采样合法性、高 prior 未访问优先、visit target 归一化

use crate::rl::mcts::puct::sample_dirichlet;
use crate::rl::mcts::traits::{SelectionRule, TargetRule};
use crate::rl::mcts::types::ActionPayload;
use crate::rl::mcts::{ChildStat, MctsConfig, MinMaxStats, PuctPolicy};

#[test]
fn test_dirichlet_sums_to_one() {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(42);
    let d = sample_dirichlet(0.3, 5, &mut rng);
    assert_eq!(d.len(), 5);
    let sum: f32 = d.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Dirichlet 样本之和应为 1，实际: {sum}"
    );
    for &x in &d {
        assert!(x >= 0.0, "Dirichlet 分量不应为负");
    }
}

#[test]
fn test_puct_select_prefers_high_prior_unvisited() {
    let policy = PuctPolicy::new();
    let cfg = MctsConfig::default();
    let children = vec![
        ChildStat {
            action_id: 0.into(),
            action: ActionPayload::Discrete(0),
            visit_count: 10,
            value_sum: 5.0,
            prior: 0.1,
            reward: 0.0,
            to_play: 0,
            discount: 1.0,
        },
        ChildStat {
            action_id: 1.into(),
            action: ActionPayload::Discrete(1),
            visit_count: 0,
            value_sum: 0.0,
            prior: 0.9,
            reward: 0.0,
            to_play: 0,
            discount: 1.0,
        },
    ];
    let idx = policy.select_child(10, 0, &children, &MinMaxStats::new(), &cfg);
    assert_eq!(idx, 1, "高 prior 未访问节点应被优先选择");
}

#[test]
fn test_make_targets_normalization() {
    let policy = PuctPolicy::new();
    let cfg = MctsConfig::default();
    let children = vec![
        ChildStat {
            action_id: 0.into(),
            action: ActionPayload::Discrete(0),
            visit_count: 3,
            value_sum: 1.0,
            prior: 0.5,
            reward: 0.0,
            to_play: 0,
            discount: 1.0,
        },
        ChildStat {
            action_id: 1.into(),
            action: ActionPayload::Discrete(1),
            visit_count: 7,
            value_sum: 2.0,
            prior: 0.5,
            reward: 0.0,
            to_play: 0,
            discount: 1.0,
        },
    ];
    let targets = policy.make_targets(&children, &cfg);
    assert_eq!(targets.len(), 2);
    assert!((targets[0] - 0.3).abs() < 1e-5);
    assert!((targets[1] - 0.7).abs() < 1e-5);
}
